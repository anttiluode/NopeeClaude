"""Score everything on the test prompts and compute the preregistered gates (docs/PREREG.md).

    python evaluate.py                # latent metrics + gates (no GPU needed beyond adapters)
    python evaluate.py --decode       # also decoded-frame PSNR and flicker, and side-by-side videos
    python evaluate.py --inblock      # also regenerate the test set with the stage-2 in-block adapter
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from nopee.adapters import BlockAdapters, LatentAdapter, Unsharp
from nopee.data import load_split, per_item_mse, rel_improvement, stack
from nopee.wan import MODELS, TINY, decode, generate, load_generator, noise_for, sync, tiny_generator

ROOT = Path(__file__).resolve().parent
ARMS = ("none", "unsharp", "hc_learned", "hc_fixed", "hc_random", "free")


def load_arm(arm, budget, adir, unsharp):
    if arm == "none":
        return lambda x: x
    if arm == "unsharp":
        u = unsharp[budget]
        return Unsharp(u["sigma"], u["gain"])
    ck = torch.load(adir / f"{budget}_{arm}.pt")
    m = LatentAdapter(ck["channels"], ck["kind"])
    m.load_state_dict(ck["state"])
    return m.eval()


def adapter_seconds(m, x, dev, reps=10):
    if not isinstance(m, torch.nn.Module):
        return 0.0
    m = m.to(dev)
    x = x.to(dev)
    with torch.no_grad():
        m(x)
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            m(x)
        sync(dev)
    return (time.perf_counter() - t0) / reps


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12)))


def flicker_excess(a, ref):
    """Mean |frame-to-frame change of a - that of the reference| (0 = same temporal behaviour)."""
    return float(np.mean(np.abs(np.diff(a, axis=0) - np.diff(ref, axis=0))))


def gates(L, T, primary="s6", double="s12"):
    e = lambda b, a: L[b][a]
    G = {}
    G["G0 cheap budget is really faster (ref/s6 time >= 4)"] = dict(
        ratio=T["ref"] / T[primary], passed=bool(T["ref"] / T[primary] >= 4))
    r = rel_improvement(e(primary, "hc_learned"), e(primary, "none"))
    G["G1 adapter recovers >= 10% of the error at s6"] = dict(**r, passed=bool(r["rel"] >= .10 and r["lo"] > 0))
    a, b = float(np.mean(e(primary, "hc_learned"))), float(np.mean(e(double, "none")))
    G["G2 s6 + adapter is at least as close as s12 without"] = dict(s6_adapter=a, s12_none=b, passed=bool(a <= b))
    r = rel_improvement(e(primary, "hc_learned"), e(primary, "free"))
    G["G3 structure: hypercolumn beats free kernels by >= 5%"] = dict(**r, passed=bool(r["rel"] >= .05 and r["lo"] > 0))
    r = rel_improvement(e(primary, "hc_fixed"), e(primary, "hc_random"))
    G["G4 constants: prior beats random constants by >= 5%"] = dict(**r, passed=bool(r["rel"] >= .05 and r["lo"] > 0))
    r = rel_improvement(e(primary, "hc_learned"), e(primary, "unsharp"))
    G["G5 more than sharpening: beats tuned unsharp mask by >= 5%"] = dict(**r, passed=bool(r["rel"] >= .05 and r["lo"] > 0))
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--adapters", default="adapters")
    ap.add_argument("--out", default="results")
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--inblock", action="store_true")
    ap.add_argument("--videos", type=int, default=4)
    args = ap.parse_args()
    meta, te = load_split(ROOT / args.data, "test")
    tiny = meta["model"] == "tiny"
    dev = "cuda" if torch.cuda.is_available() and not tiny else "cpu"
    adir, out = ROOT / args.adapters, ROOT / args.out
    out.mkdir(exist_ok=True)
    unsharp = json.loads((adir / "unsharp.json").read_text())
    budgets = [b for b in meta["budgets"] if b != "ref"]
    ref = stack(te, "ref")
    T = {b: float(np.median([r["sec"][b] for r in te])) for b in meta["budgets"]}

    L, AT, OUT = {}, {}, {}
    for b in budgets:
        X = stack(te, b)
        L[b], AT[b], OUT[b] = {}, {}, {}
        for arm in ARMS:
            m = load_arm(arm, b, adir, unsharp)
            with torch.no_grad():
                Y = torch.cat([m(X[i:i + 1]) for i in range(len(X))]) if arm != "none" else X
            L[b][arm] = per_item_mse(Y, ref).tolist()
            AT[b][arm] = adapter_seconds(m, X[:1], dev)
            OUT[b][arm] = Y
    res = dict(model=meta["model"], n_test=len(te), median_seconds=T, adapter_seconds=AT,
               latent_mse={b: {a: float(np.mean(v)) for a, v in L[b].items()} for b in L}, latent_mse_items=L)
    res["gates"] = gates(L, T)
    res["report_only"] = {f"{b}: hc_learned vs none": rel_improvement(L[b]["hc_learned"], L[b]["none"]) for b in budgets}
    res["report_only"].update({f"{b}: hc_learned vs free": rel_improvement(L[b]["hc_learned"], L[b]["free"]) for b in budgets})
    res["effective_speedup_s6_adapter_vs_s12"] = T["s12"] / (T["s6"] + AT["s6"]["hc_learned"]) if "s12" in T else None
    res["speedup_vs_ref"] = {b: T["ref"] / T[b] for b in budgets}

    if args.decode:
        if tiny:
            dp = tiny_generator()
        else:
            from diffusers import AutoencoderKLWan
            from diffusers.video_processor import VideoProcessor
            vae = AutoencoderKLWan.from_pretrained(MODELS[meta["model"]]["id"], subfolder="vae", torch_dtype=torch.float32).to(dev)
            dp = SimpleNamespace(vae=vae, video_processor=VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial))
        D = {}
        show = [b for b in ("s6", "s12") if b in budgets]
        refs = [decode(dp, ref[i:i + 1]) for i in range(len(te))]
        for b in show:
            D[b] = {}
            for arm in ARMS:
                vids = [decode(dp, OUT[b][arm][i:i + 1]) for i in range(len(te))]
                D[b][arm] = dict(psnr=float(np.mean([psnr(v, r) for v, r in zip(vids, refs)])),
                                 flicker_excess=float(np.mean([flicker_excess(v, r) for v, r in zip(vids, refs)])))
                if arm in ("none", "hc_learned") and b == "s6":
                    D[b][f"_{arm}_vids"] = vids[:args.videos]
        for b in budgets:
            if b not in show:
                vids = [decode(dp, OUT[b]["none"][i:i + 1]) for i in range(len(te))]
                D[b] = dict(none=dict(psnr=float(np.mean([psnr(v, r) for v, r in zip(vids, refs)]))))
        save_videos(D, refs, te, out, meta)
        res["decoded"] = {b: {a: v for a, v in d.items() if not a.startswith("_")} for b, d in D.items()}

    if args.inblock:
        cfg = dict(TINY) if tiny else dict(MODELS[meta["model"]], **meta["cfg"])
        pipe = tiny_generator() if tiny else load_generator(meta["model"], dev, fp8=meta["fp8"])
        em = torch.load(ROOT / args.data / "embeds.pt")
        rb = meta["record"]
        res["inblock"] = {}
        for f in sorted(adir.glob(f"inblock_{rb}_*.pt")):
            ck = torch.load(f)
            ad = BlockAdapters(ck["dim"], ck["blocks"], ck["rank"], ck["kind"]).to(dev)
            ad.load_state_dict(ck["state"])
            ad.install(pipe.transformer)
            errs, secs = [], []
            for i, r in enumerate(te):
                noise = noise_for(pipe, cfg, r["seed"], dev)
                lat, sec, _ = generate(pipe, cfg, em["E"][r["idx"]], em["neg"], noise, meta["budgets"][rb]["steps"])
                errs.append(float(per_item_mse(lat.cpu(), ref[i:i + 1])[0]))
                secs.append(sec)
            ad.remove()
            res["inblock"][ck["kind"]] = dict(latent_mse=float(np.mean(errs)), median_seconds=float(np.median(secs)),
                                              vs_none=rel_improvement(errs, L[rb]["none"]),
                                              vs_final_latent_adapter=rel_improvement(errs, L[rb]["hc_learned"]))

    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
    lines = [f"# results ({meta['model']}, {len(te)} test prompts)", "", "| gate | | |", "|---|---|---|"]
    for k, v in res["gates"].items():
        lines.append(f"| {k} | {'PASS' if v['passed'] else 'FAIL'} | " +
                     ", ".join(f"{a}={b:.4g}" for a, b in v.items() if a != "passed") + " |")
    lines += ["", "median seconds per video: " + ", ".join(f"{b} {t:.2f}" for b, t in T.items()), "",
              "| latent MSE to reference | " + " | ".join(ARMS) + " |", "|---" * (len(ARMS) + 1) + "|"]
    for b in budgets:
        lines.append(f"| {b} | " + " | ".join(f"{res['latent_mse'][b][a]:.5f}" for a in ARMS) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def save_videos(D, refs, te, out, meta):
    try:
        from diffusers.utils import export_to_video
    except Exception:
        return
    fps = meta["cfg"].get("fps", 16)
    for j in range(len(D.get("s6", {}).get("_none_vids", []))):
        r, a, b = refs[j], D["s6"]["_none_vids"][j], D["s6"]["_hc_learned_vids"][j]
        frames = [np.concatenate([r[f], a[f], b[f]], axis=1) for f in range(len(r))]
        try:
            export_to_video(frames, str(out / f"side_by_side_{te[j]['idx']:04d}.mp4"), fps=fps)
        except Exception as e:  # no video backend: keep a still strip instead
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.imsave(out / f"side_by_side_{te[j]['idx']:04d}.png", frames[len(frames) // 2].clip(0, 1))


if __name__ == "__main__":
    main()
