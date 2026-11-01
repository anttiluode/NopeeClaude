import json
from pathlib import Path

import numpy as np
import torch


def load_split(root, split, keep_traj=False):
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())
    recs = []
    for i in meta["splits"][split]:
        f = root / split / f"{i:04d}.pt"
        if f.exists():
            r = torch.load(f)
            if not keep_traj:
                r.pop("traj", None)
            r["idx"] = i
            recs.append(r)
    return meta, recs


def stack(recs, budget):
    return torch.cat([r["lat"][budget].float() for r in recs])


def per_item_mse(a, b):
    return ((a.float() - b.float()) ** 2).flatten(1).mean(1).numpy()


def rel_improvement(err_new, err_old, n_boot=10_000, seed=0):
    """1 - mean(new)/mean(old), with a paired bootstrap 95% CI over prompts."""
    err_new, err_old = np.asarray(err_new), np.asarray(err_old)
    point = 1 - err_new.mean() / err_old.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err_new), (n_boot, len(err_new)))
    boots = 1 - err_new[idx].mean(1) / err_old[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(rel=float(point), lo=float(lo), hi=float(hi))
