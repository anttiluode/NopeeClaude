"""How far is each cheap run from the reference -- and is it the same video, just rougher?

Prints, per budget, on validation/train prompts (never test):
  mse        latent MSE to the reference (latents are normalised, so ~1 per element is "big")
  unrelated  MSE between the references of two DIFFERENT prompts: what "a different video" costs
  share      mse / unrelated: 0 = identical, 1 = as far apart as two unrelated videos
  corr       correlation of the cheap latent with the reference latent
  low        share after 4x spatial average-pooling (layout and colour blobs only)
  first/last share in the first and the last latent frame

    python inspect_data.py --data data
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nopee.data import load_split

ROOT = Path(__file__).resolve().parent


def pool(x, k=4):
    B, C, Fr, H, W = x.shape
    y = x.permute(0, 2, 1, 3, 4).reshape(B * Fr, C, H, W)
    y = F.avg_pool2d(y, k, ceil_mode=True)
    return y.reshape(B, Fr, C, *y.shape[-2:]).permute(0, 2, 1, 3, 4)


def mse(a, b):
    return float(((a - b) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    recs = []
    for split in ("val", "train"):
        meta, r = load_split(ROOT / args.data, split)
        recs += r
    refs = [r["lat"]["ref"].float() for r in recs]
    n = len(refs)
    unrel = np.mean([mse(refs[i], refs[(i + 1) % n]) for i in range(n)])
    unrel_low = np.mean([mse(pool(refs[i]), pool(refs[(i + 1) % n])) for i in range(n)])
    unrel_f0 = np.mean([mse(refs[i][:, :, 0], refs[(i + 1) % n][:, :, 0]) for i in range(n)])
    unrel_fl = np.mean([mse(refs[i][:, :, -1], refs[(i + 1) % n][:, :, -1]) for i in range(n)])
    print(f"{n} prompts; reference latent variance {np.mean([r.var().item() for r in refs]):.3f}; "
          f"two unrelated references: MSE {unrel:.3f} (pooled {unrel_low:.3f})")
    out = dict(n=n, unrelated=unrel, unrelated_low=unrel_low, budgets={})
    budgets = sorted({b for r in recs for b in r["lat"] if b != "ref"}, key=lambda b: (len(b), b))
    print(f"{'budget':10s} {'mse':>7s} {'share':>6s} {'corr':>6s} {'low':>6s} {'first':>6s} {'last':>6s}")
    for b in budgets:
        rows = [(r["lat"][b].float(), ref) for r, ref in zip(recs, refs) if b in r["lat"]]
        m = np.mean([mse(x, y) for x, y in rows])
        c = np.mean([float(torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1]) for x, y in rows])
        lo = np.mean([mse(pool(x), pool(y)) for x, y in rows]) / unrel_low
        f0 = np.mean([mse(x[:, :, 0], y[:, :, 0]) for x, y in rows]) / unrel_f0
        fl = np.mean([mse(x[:, :, -1], y[:, :, -1]) for x, y in rows]) / unrel_fl
        out["budgets"][b] = dict(mse=m, share=m / unrel, corr=c, low=lo, first=f0, last=fl, n=len(rows))
        print(f"{b:10s} {m:7.3f} {m / unrel:6.2f} {c:6.2f} {lo:6.2f} {f0:6.2f} {fl:6.2f}")
    (ROOT / args.data / "inspect.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
