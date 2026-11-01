import torch

from nopee.adapters import BlockAdapters, DoG, LatentAdapter, LeakyEI, Unsharp, gaussian_2d
from nopee.data import rel_improvement
from nopee.wan import TINY, generate, model_call, noise_for, tiny_generator

torch.set_num_threads(2)


def test_adapters_start_as_identity():
    x = torch.randn(2, 16, 5, 6, 6)
    for kind in ("hc_learned", "hc_fixed", "hc_random", "free"):
        assert torch.equal(LatentAdapter(16, kind)(x), x)
    assert torch.equal(Unsharp(1.0, 0.0)(x), x)


def test_dog_is_centre_surround():
    d = DoG(4, "prior")
    k = d.kernel()
    c = k.shape[-1] // 2
    assert (k[:, c, c] > 0).all() and (k[:, 0, 0] <= k[:, c, c]).all()
    assert torch.allclose(gaussian_2d(torch.tensor([1.0])).sum(), torch.tensor(1.0))
    assert not d.log_lam.requires_grad and DoG(4, "learned").log_lam.requires_grad


def test_temporal_parts_are_causal():
    for m in (LatentAdapter(8, "hc_learned"), LatentAdapter(8, "free")):
        torch.nn.init.normal_(m.branch.proj_out.weight)
        x = torch.randn(1, 8, 6, 5, 5)
        y = x.clone()
        y[:, :, 4] += 1.0                  # change frame 4
        d = (m(y) - m(x)).abs().amax((0, 1, 3, 4))
        assert d[:4].max() == 0 and d[4] > 0


def test_leaky_ei_impulse_decays():
    ei = LeakyEI(4, "prior")
    u = torch.zeros(1, 4, 30, 1, 1)
    u[:, :, 0] = 1.0
    r = ei(u)[0, :, :, 0, 0].abs()
    assert (r[:, -1] < r[:, 0]).all()


def test_shared_noise_and_last_step_replay():
    pipe = tiny_generator()
    emb, neg = torch.randn(1, 8, 16), torch.randn(1, 8, 16)
    n = noise_for(pipe, TINY, 3, "cpu")
    a, _, traj = generate(pipe, TINY, emb, neg, n, 6, record=True)
    b, _, _ = generate(pipe, TINY, emb, neg, n, 6)
    assert torch.equal(a, b) and len(traj) == 6
    x, t, sig = traj[-1]
    v = model_call(pipe, x.float(), t, emb, neg, TINY["guidance"])
    assert (x.float() - sig * v - a).abs().max() < 1e-2        # our model call == the pipeline's


def test_inblock_hooks_zero_init_and_gradients():
    pipe = tiny_generator()
    T = pipe.transformer
    T.requires_grad_(False)
    T.enable_gradient_checkpointing()
    emb, neg = torch.randn(1, 8, 16), torch.randn(1, 8, 16)
    n = noise_for(pipe, TINY, 4, "cpu")
    a, _, _ = generate(pipe, TINY, emb, neg, n, 4)
    ad = BlockAdapters(24, (0, 2), r=4).install(T)
    b, _, _ = generate(pipe, TINY, emb, neg, n, 4)
    assert torch.equal(a, b)
    v = model_call(pipe, n, 900.0, emb, neg, 5.0)
    v.pow(2).mean().backward()
    g = ad.branches["0"].proj_out.weight.grad
    assert g is not None and g.abs().sum() > 0
    ad.remove()
    assert len(ad._handles) == 0


def test_bootstrap_sign():
    r = rel_improvement([1.0, 1.1, 0.9, 1.0], [2.0, 2.1, 1.9, 2.2])
    assert r["rel"] > 0.4 and r["lo"] > 0


def test_lowmem_loader_matches_standard(tmp_path):
    from diffusers import WanTransformer3DModel
    from nopee.wan import load_transformer_lowmem
    ref = tiny_generator().transformer
    ref.save_pretrained(tmp_path, max_shard_size="20KB")          # several fp32 shards, like the real repo
    a = WanTransformer3DModel.from_pretrained(tmp_path, torch_dtype=torch.bfloat16)
    a.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    b = load_transformer_lowmem(str(tmp_path), fp8=True, local=True)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert sa[k].dtype == sb[k].dtype, k
        assert torch.equal(sa[k].float(), sb[k].float()), k
    x = torch.randn(1, 16, 3, 4, 4)
    kw = dict(timestep=torch.tensor([500.0]), encoder_hidden_states=torch.randn(1, 8, 16).bfloat16(), return_dict=False)
    with torch.no_grad():
        assert torch.equal(a(hidden_states=x.bfloat16(), **kw)[0], b(hidden_states=x.bfloat16(), **kw)[0])
