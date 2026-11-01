"""Wan loading and sampling sized for a 12 GB card (RTX 3060) and 24 GB of RAM.

Two phases so the two big pieces are never in memory together:
  1. the UMT5-XXL text encoder (~11 GB in bf16) encodes every prompt once -> embeddings on disk
  2. the transformer (Wan2.2-TI2V-5B, ~10 GB in bf16) is stored in fp8 (~5 GB) and computed in bf16
     (diffusers layerwise casting; works on Ampere, the 3060 has no fp8 maths and does not need it)

Every condition of one prompt starts from the same initial noise, so a cheap run and the
expensive reference are two discretisations of the same flow-matching ODE trajectory.
"""
import re
import time
from contextlib import contextmanager

import numpy as np
import torch

MODELS = {
    # the smaller Wan2.2: text+image to video, 5B, VAE 16x16 spatial / 4x temporal, 48 latent channels
    "wan22-5b": dict(id="Wan-AI/Wan2.2-TI2V-5B-Diffusers", height=320, width=576, frames=33, fps=24, guidance=5.0),
    # a much faster development stand-in (Wan2.1, 1.3B, 16 latent channels)
    "wan21-1.3b": dict(id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", height=320, width=576, frames=33, fps=16, guidance=5.0),
}

NEGATIVE = ("Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
            "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
            "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
            "fused fingers, still picture, messy background, three legs, many people in the background, "
            "walking backwards")

# Sampling budgets. 'ref' is the expensive reference the cheap runs are compared to.
BUDGETS = {
    "ref": dict(steps=40),
    "s6": dict(steps=6),
    "s12": dict(steps=12),
    "s24": dict(steps=24),
    "s12c": dict(steps=12, cache=0.10),     # 12 steps + FirstBlockCache (skips blocks when step changes little)
}
FP8 = getattr(torch, "float8_e4m3fn", None)
LAST = {}      # stats of the last generate() call: share of transformer calls that ran all blocks


def parse_budget(name):
    """'ref' / 's6' / 's12c' from BUDGETS, or any 's<steps>' / 's<steps>c<threshold>' e.g. 's40c0.15'."""
    if name in BUDGETS:
        return dict(BUDGETS[name])
    m = re.fullmatch(r"s(\d+)(?:c([0-9.]+))?", name)
    if not m:
        raise ValueError(f"unknown budget {name!r}")
    b = dict(steps=int(m.group(1)))
    if m.group(2):
        b["cache"] = float(m.group(2))
    return b


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


# ------------------------------------------------------------------------- phase 1: text

def mem(tag=""):
    """One line of RAM / VRAM use, so a silent crash shows where it happened."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram = f"RAM used {psutil.Process().memory_info().rss / 2**30:.1f} GB (free {vm.available / 2**30:.1f} of {vm.total / 2**30:.1f})"
    except Exception:
        ram = "RAM ?"
    vr = ""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        vr = f", VRAM free {free / 2**30:.1f} of {total / 2**30:.1f} GB"
    print(f"[{tag}] {ram}{vr}", flush=True)


def _te_mode(mode, device):
    if mode != "auto":
        return mode
    if device != "cpu" and torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401
            return "8bit"
        except Exception:
            return "fp8"
    return "cpu"


@torch.no_grad()
def encode_prompts(model_key, prompts, device="cuda", mode="auto", max_len=512):
    """Encode prompts with Wan's UMT5-XXL text encoder, exactly as WanPipeline does.
    Returns (list of (1, max_len, 4096) fp16 CPU tensors, negative embedding).

    mode  8bit : bitsandbytes int8, streamed straight to the GPU (~6 GB VRAM, low RAM peak) [default with CUDA]
          fp8  : bf16 into RAM (~11 GB peak), stored as fp8 on the GPU
          cpu  : bf16 on the CPU (~11 GB RAM, slow but needs no GPU memory)
    """
    from transformers import AutoTokenizer, UMT5EncoderModel
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    mid = MODELS[model_key]["id"]
    mode = _te_mode(mode, device)
    print(f"text encoder mode: {mode}", flush=True)
    mem("before text encoder")
    tok = AutoTokenizer.from_pretrained(mid, subfolder="tokenizer")
    if mode == "8bit":
        from transformers import BitsAndBytesConfig
        te = UMT5EncoderModel.from_pretrained(mid, subfolder="text_encoder", dtype=torch.bfloat16,
                                              quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                                              device_map={"": 0}, low_cpu_mem_usage=True)
        dev = "cuda"
    else:
        te = UMT5EncoderModel.from_pretrained(mid, subfolder="text_encoder", dtype=torch.bfloat16, low_cpu_mem_usage=True)
        mem("text encoder in RAM")
        dev = "cpu"
        if mode == "fp8":
            from diffusers.hooks import apply_layerwise_casting
            apply_layerwise_casting(te, storage_dtype=FP8, compute_dtype=torch.bfloat16)
            te.to("cuda")
            dev = "cuda"
    te.eval()
    mem("text encoder loaded")

    def enc(text):
        t = tok([prompt_clean(text)], padding="max_length", max_length=max_len, truncation=True,
                add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        n = int(t.attention_mask.sum())
        e = te(t.input_ids.to(dev), t.attention_mask.to(dev)).last_hidden_state[0, :n].float()
        e = torch.cat([e, e.new_zeros(max_len - n, e.shape[1])])     # Wan zero-pads past the prompt
        return e[None].to("cpu", torch.float16)

    out = []
    for i, p in enumerate(prompts):
        out.append(enc(p))
        if i % 10 == 0:
            print(f"  encoded {i + 1}/{len(prompts)}", flush=True)
    neg = enc(NEGATIVE)
    del te
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mem("text encoder freed")
    return out, neg


# ------------------------------------------------------------------------- phase 2: video

def load_transformer_lowmem(source, fp8=True, local=False):
    """Load the Wan transformer one tensor at a time, casting each straight to its storage dtype.

    The Wan2.2-5B repo stores the transformer in fp32 (~20 GB). The normal route builds the whole
    bf16 model in RAM first (~10 GB, plus a shard being converted). This one builds an empty model,
    applies the same fp8-storage rule diffusers uses, and fills it tensor by tensor, so peak RAM is
    about the final size (~5.5 GB). The result is identical to
    from_pretrained(torch_dtype=bfloat16) + enable_layerwise_casting (checked in the tests)."""
    import json
    import os
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import WanTransformer3DModel
    from safetensors import safe_open
    if local:
        tdir = source
    else:
        from huggingface_hub import snapshot_download
        tdir = os.path.join(snapshot_download(source, allow_patterns=["transformer/*"]), "transformer")
    cfg = WanTransformer3DModel.load_config(tdir)
    with init_empty_weights():
        tr = WanTransformer3DModel.from_config(cfg)
    keep = tr._keep_in_fp32_modules or []
    for mname, mod in tr.named_modules():
        for pname, p in list(mod._parameters.items()):
            if p is None:
                continue
            full = f"{mname}.{pname}" if mname else pname
            dt = torch.float32 if any(k in full.split(".") for k in keep) else torch.bfloat16
            mod._parameters[pname] = torch.nn.Parameter(torch.empty_like(p, dtype=dt, device="meta"), requires_grad=False)
    if fp8 and FP8 is not None:
        tr.enable_layerwise_casting(storage_dtype=FP8, compute_dtype=torch.bfloat16)
    target = {n: p.dtype for n, p in tr.named_parameters()}
    idx = os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["diffusion_pytorch_model.safetensors"]
    seen = set()
    for fn in files:
        with safe_open(os.path.join(tdir, fn), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in target:
                    t = f.get_tensor(k)
                    if target[k] == FP8:
                        t = t.to(torch.bfloat16)      # same double rounding as the standard route
                    set_module_tensor_to_device(tr, k, "cpu", value=t.to(target[k]))
                    seen.add(k)
        mem(f"loaded {fn}")
    missing = set(target) - seen
    if missing:
        raise RuntimeError(f"{len(missing)} transformer weights missing, e.g. {sorted(missing)[:3]}")
    return tr.eval()


def load_generator(model_key, device="cuda", fp8=True, offload=False, vae_tiling=True):
    from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
    mid = MODELS[model_key]["id"]
    mem("before video model")
    vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
    mem("VAE loaded")
    tr = load_transformer_lowmem(mid, fp8=fp8)
    mem("transformer loaded" + (" (fp8 storage)" if fp8 else ""))
    pipe = WanPipeline.from_pretrained(mid, transformer=tr, vae=vae, text_encoder=None, tokenizer=None,
                                       torch_dtype=torch.bfloat16)
    if vae_tiling:
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    mem("video model ready")
    return pipe


def tiny_generator(expand_timesteps=True, seed=0):
    """A randomly initialised miniature Wan for tests (real diffusers code paths, no download)."""
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
    torch.manual_seed(seed)
    vae = AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                           temperal_downsample=[False, True, True])
    tr = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
                               in_channels=16, out_channels=16, text_dim=16, freq_dim=256, ffn_dim=32,
                               num_layers=4, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
                               rope_max_seq_len=32)
    sch = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)
    pipe = WanPipeline(tokenizer=None, text_encoder=None, transformer=tr, vae=vae, scheduler=sch,
                       expand_timesteps=expand_timesteps)
    pipe.set_progress_bar_config(disable=True)
    return pipe


TINY = dict(height=32, width=32, frames=9, fps=8, guidance=5.0, text_len=8, text_dim=16)


def noise_for(pipe, cfg, seed, device):
    nf = (cfg["frames"] - 1) // pipe.vae_scale_factor_temporal + 1
    s = pipe.vae_scale_factor_spatial
    shape = (1, pipe.transformer.config.in_channels, nf, cfg["height"] // s, cfg["width"] // s)
    g = torch.Generator("cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=g, dtype=torch.float32).to(device)


@contextmanager
def cache_enabled(pipe, threshold):
    if threshold is None:
        yield
        return
    from diffusers import FirstBlockCacheConfig
    pipe.transformer.enable_cache(FirstBlockCacheConfig(threshold=threshold))
    try:
        yield
    finally:
        pipe.transformer.disable_cache()


@torch.no_grad()
def generate(pipe, cfg, emb, neg, noise, steps, cache=None, record=False):
    """Run Wan from the given noise. Returns (final latent (normalised space), seconds, trajectory).
    trajectory (if record): list of (x entering step i, timestep t_i, sigma_i)."""
    dev = noise.device
    dt = pipe.transformer.dtype if pipe.transformer.dtype != FP8 else torch.bfloat16
    xs = []

    def cb(p, i, t, kw):
        xs.append(kw["latents"].detach().to("cpu", torch.float16))
        return {}

    calls = {"head": 0, "tail": 0}
    hs = [pipe.transformer.blocks[0].register_forward_hook(lambda *a: calls.__setitem__("head", calls["head"] + 1)),
          pipe.transformer.blocks[-1].ffn.register_forward_hook(lambda *a: calls.__setitem__("tail", calls["tail"] + 1))]
    with cache_enabled(pipe, cache):
        sync(dev)
        t0 = time.perf_counter()
        lat = pipe(prompt_embeds=emb.to(dev, dt), negative_prompt_embeds=neg.to(dev, dt),
                   latents=noise.clone(), height=cfg["height"], width=cfg["width"], num_frames=cfg["frames"],
                   num_inference_steps=steps, guidance_scale=cfg["guidance"], output_type="latent",
                   callback_on_step_end=cb if record else None).frames
        sync(dev)
        sec = time.perf_counter() - t0
    for h in hs:
        h.remove()
    LAST["computed"] = calls["tail"] / max(calls["head"], 1)
    traj = None
    if record:
        ts = pipe.scheduler.timesteps.detach().cpu()
        sig = pipe.scheduler.sigmas.detach().cpu()
        entering = [noise.detach().to("cpu", torch.float16)] + xs[:-1]
        traj = [(entering[i], float(ts[i]), float(sig[i])) for i in range(len(entering))]
    return lat.detach().float(), sec, traj


@torch.no_grad()
def decode(pipe, lat):
    """Normalised latent -> float frames (F, H, W, 3) in [0, 1] (exactly the pipeline's own decode)."""
    vae = pipe.vae
    dev = next(vae.parameters()).device
    lat = lat.to(dev, vae.dtype)
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(lat)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(lat)
    video = vae.decode(lat / inv_std + mean, return_dict=False)[0]
    v = pipe.video_processor.postprocess_video(video, output_type="np")[0]
    return np.asarray(v, dtype=np.float32)


def model_call(pipe, x, t, emb, neg, guidance):
    """One guided velocity prediction exactly as WanPipeline does it (with gradients if enabled)."""
    tr = pipe.transformer
    dt = torch.bfloat16 if tr.dtype in (FP8, torch.bfloat16) else tr.dtype
    xin = x.to(dt)
    tt = torch.tensor([t], device=x.device, dtype=torch.float32)
    if pipe.config.expand_timesteps:
        _, _, Fr, H, W = x.shape
        timestep = tt.repeat(Fr * (H // 2) * (W // 2))[None].expand(x.shape[0], -1)
    else:
        timestep = tt.expand(x.shape[0])
    vc = tr(hidden_states=xin, timestep=timestep, encoder_hidden_states=emb.to(x.device, dt), return_dict=False)[0]
    vu = tr(hidden_states=xin, timestep=timestep, encoder_hidden_states=neg.to(x.device, dt), return_dict=False)[0]
    return (vu + guidance * (vc - vu)).float()
