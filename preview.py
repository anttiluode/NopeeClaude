"""Decode stored latents to look at them: reference | cheap budgets, side by side.
Loads only the VAE (about 3 GB), so it can run while nothing else is on the GPU.

    python preview.py --data data_trial                 # every prompt in the set, budgets ref,s6,s12
    python preview.py --data data_trial --budgets ref,s6 --out previews
"""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import torch

from nopee.data import load_split
from nopee.wan import MODELS, decode, tiny_generator

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_trial")
    ap.add_argument("--budgets", default="ref,s6,s12")
    ap.add_argument("--out", default="previews")
    args = ap.parse_args()
    recs = []
    for split in ("test", "val", "train"):
        meta, r = load_split(ROOT / args.data, split)
        recs += r
    dev = "cuda" if torch.cuda.is_available() and meta["model"] != "tiny" else "cpu"
    if meta["model"] == "tiny":
        dp = tiny_generator()
    else:
        from diffusers import AutoencoderKLWan
        from diffusers.video_processor import VideoProcessor
        vae = AutoencoderKLWan.from_pretrained(MODELS[meta["model"]]["id"], subfolder="vae", torch_dtype=torch.float32).to(dev)
        try:
            vae.enable_tiling()
        except Exception:
            pass
        dp = SimpleNamespace(vae=vae, video_processor=__import__("diffusers.video_processor", fromlist=["VideoProcessor"]).VideoProcessor(
            vae_scale_factor=vae.config.scale_factor_spatial))
    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    budgets = [b for b in args.budgets.split(",") if all(b in r["lat"] for r in recs)]
    from diffusers.utils import export_to_video
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for r in recs:
        vids = [decode(dp, r["lat"][b].float()) for b in budgets]
        frames = [np.concatenate([v[f] for v in vids], axis=1) for f in range(len(vids[0]))]
        name = f"{r['idx']:04d}_" + "-".join(budgets)
        try:
            export_to_video(frames, str(out / f"{name}.mp4"), fps=meta["cfg"].get("fps", 16))
        except Exception as e:
            print("video export failed:", e)
        plt.imsave(out / f"{name}.png", frames[len(frames) // 2].clip(0, 1))
        ref = vids[0]
        psnr = [10 * np.log10(1 / max(np.mean((v - ref) ** 2), 1e-12)) for v in vids[1:]]
        print(f"{r['idx']:4d}  {r['prompt'][:60]:60s}  PSNR to {budgets[0]}: " +
              "  ".join(f"{b} {p:.1f} dB" for b, p in zip(budgets[1:], psnr)), flush=True)


if __name__ == "__main__":
    main()
