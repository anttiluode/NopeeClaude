"""How much does FirstBlockCache actually skip, and what does it cost in closeness to the reference?
Runs on VALIDATION prompts that already have a reference (never test prompts). Report-only; it is for
choosing the cache budget's threshold before the cache budget is generated (recorded in docs/PREREG.md).

    python make_data.py --splits val --budgets ref,s6,s12,s24     # validation prompts first
    python cache_sweep.py --steps 12,40 --thresholds 0.05,0.1,0.2,0.3
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import torch

from nopee.data import load_split
from nopee.wan import LAST, MODELS, TINY, generate, load_generator, noise_for, tiny_generator

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--steps", default="12,40")
    ap.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--n", type=int, default=3, help="how many validation prompts")
    args = ap.parse_args()
    meta, recs = load_split(ROOT / args.data, "val")
    recs = recs[:args.n]
    assert recs, "no validation prompts with a reference yet: run make_data.py --splits val first"
    tiny = meta["model"] == "tiny"
    dev = "cuda" if torch.cuda.is_available() and not tiny else "cpu"
    cfg = dict(TINY) if tiny else dict(MODELS[meta["model"]], **meta["cfg"])
    pipe = tiny_generator() if tiny else load_generator(meta["model"], dev, fp8=meta["fp8"])
    em = torch.load(ROOT / args.data / "embeds.pt")
    rows = []
    for steps in [int(s) for s in args.steps.split(",")]:
        ths = [float(t) for t in args.thresholds.split(",")]
        for th in ([None] if steps != 40 else []) + ths:      # 40 steps without cache is the reference itself
            secs, fulls, errs = [], [], []
            for r in recs:
                noise = noise_for(pipe, cfg, r["seed"], dev)
                lat, sec, _ = generate(pipe, cfg, em["E"][r["idx"]], em["neg"], noise, steps, th)
                secs.append(sec)
                fulls.append(LAST["computed"])
                errs.append(float(((lat.cpu() - r["lat"]["ref"].float()) ** 2).mean()))
            row = dict(steps=steps, threshold=th, seconds=float(np.mean(secs)), full_share=float(np.mean(fulls)),
                       mse_to_ref=float(np.mean(errs)))
            rows.append(row)
            print(f"steps {steps:3d}  cache {str(th):5s}  {row['seconds']:6.1f} s  "
                  f"{100 * row['full_share']:5.1f}% of calls ran every block  MSE to ref {row['mse_to_ref']:.5f}", flush=True)
    (ROOT / args.data / "cache_sweep.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
