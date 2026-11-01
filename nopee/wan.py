"""Wan loading and sampling sized for a 12 GB card (RTX 3060) and 24 GB of RAM.

Two phases so the two big pieces are never in memory together:
  1. the UMT5-XXL text encoder (~11 GB in bf16) encodes every prompt once -> embeddings on disk
  2. the transformer (Wan2.2-TI2V-5B, ~10 GB in bf16) is stored in fp8 (~5 GB) and computed in bf16
     (diffusers layerwise casting; works on Ampere, the 3060 has no fp8 maths and does not need it)

Every condition of one prompt starts from the same initial noise, so a cheap run and the
expensive reference are two discretisations of the same flow-matching ODE trajectory.
"""
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


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


# ------------------------------------------------------------------------- phase 1: text

@torch.no_grad()
def encode_prompts(model_key, prompts, device="cuda", fp8=True, max_len=512):
    """Returns (list of prompt embeddings (1, L, 4096) fp16 on CPU, negative embedding)."""
    from diffusers import WanPipeline
    from diffusers.hooks import apply_layerwise_casting
    mid = MODELS[model_key]["id"]
    pipe = WanPipeline.from_pretrained(mid, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    te = pipe.text_encoder
    if fp8 and device != "cpu" and FP8 is not None:
        apply_layerwise_casting(te, storage_dtype=FP8, compute_dtype=torch.bfloat16)
    te.to(device)
    out = []
    for p in prompts:
        e, _ = pipe.encode_prompt(p, do_classifier_free_guidance=False, max_sequence_length=max_len, device=device)
        out.append(e.to("cpu", torch.float16))
    neg, _ = pipe.encode_prompt(NEGATIVE, do_classifier_free_guidance=False, max_sequence_length=max_len, device=device)
    del pipe, te
    if device != "cpu":
        torch.cuda.empty_cache()
    return out, neg.to("cpu", torch.float16)


# ------------------------------------------------------------------------- phase 2: video

def load_generator(model_key, device="cuda", fp8=True, offload=False, vae_tiling=True):
    from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
    mid = MODELS[model_key]["id"]
    vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
    tr = WanTransformer3DModel.from_pretrained(mid, subfolder="transformer", torch_dtype=torch.bfloat16)
    if fp8 and FP8 is not None:
        tr.enable_layerwise_casting(storage_dtype=FP8, compute_dtype=torch.bfloat16)
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

    with cache_enabled(pipe, cache):
        sync(dev)
        t0 = time.perf_counter()
        lat = pipe(prompt_embeds=emb.to(dev, dt), negative_prompt_embeds=neg.to(dev, dt),
                   latents=noise.clone(), height=cfg["height"], width=cfg["width"], num_frames=cfg["frames"],
                   num_inference_steps=steps, guidance_scale=cfg["guidance"], output_type="latent",
                   callback_on_step_end=cb if record else None).frames
        sync(dev)
        sec = time.perf_counter() - t0
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
