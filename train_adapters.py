"""Train the final-latent adapters: cheap latent -> reference latent, one adapter per budget and kind.
Only the train split is used for fitting and the val split for early stopping / the unsharp grid.
Takes minutes: the data are cached latents, the adapters are tiny.

    python train_adapters.py [--data data] [--steps 1500]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nopee.adapters import LatentAdapter, Unsharp, n_params
from nopee.data import load_split, per_item_mse, stack

ROOT = Path(__file__).resolve().parent
KINDS = ("hc_learned", "hc_fixed", "hc_random", "free")


def fit(kind, Xtr, Ytr, Xva, Yva, steps, dev, seed=0, bs=4, lr=2e-3):
    torch.manual_seed(seed)
    m = LatentAdapter(Xtr.shape[1], kind, seed=seed).to(dev)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    Xva_d, Yva_d = Xva.to(dev), Yva.to(dev)

    def val():
        with torch.no_grad():
            return float(np.mean([((m(Xva_d[i:i + 1]) - Yva_d[i:i + 1]) ** 2).mean().item() for i in range(len(Xva_d))]))

    best, best_state, log = val(), {k: v.clone() for k, v in m.state_dict().items()}, []
    for s in range(1, steps + 1):
        b = rng.integers(0, len(Xtr), min(bs, len(Xtr)))
        x, y = Xtr[b].to(dev), Ytr[b].to(dev)
        loss = ((m(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 100 == 0 or s == steps:
            v = val()
            log.append((s, loss.item(), v))
            if v < best:
                best, best_state = v, {k: v_.clone() for k, v_ in m.state_dict().items()}
    m.load_state_dict(best_state)
    return m, best, log


def fit_unsharp(Xva, Yva):
    best = (float(per_item_mse(Xva, Yva).mean()), 1.0, 0.0)
    for sigma in (0.5, 0.75, 1.0, 1.5, 2.0):
        for gain in np.round(np.linspace(-1, 3, 41), 2):
            e = float(per_item_mse(Unsharp(sigma, gain)(Xva), Yva).mean())
            if e < best[0]:
                best = (e, sigma, float(gain))
    return dict(val_mse=best[0], sigma=best[1], gain=best[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="adapters")
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta, tr = load_split(ROOT / args.data, "train")
    _, va = load_split(ROOT / args.data, "val")
    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    report = {}
    for b in meta["budgets"]:
        if b == "ref":
            continue
        Xtr, Ytr, Xva, Yva = stack(tr, b), stack(tr, "ref"), stack(va, b), stack(va, "ref")
        rb = dict(val_none=float(per_item_mse(Xva, Yva).mean()), unsharp=fit_unsharp(Xva, Yva))
        for kind in KINDS:
            m, best, log = fit(kind, Xtr, Ytr, Xva, Yva, args.steps, dev)
            torch.save(dict(kind=kind, channels=Xtr.shape[1], state=m.cpu().state_dict()), out / f"{b}_{kind}.pt")
            rb[kind] = dict(val_mse=best, params=n_params(m), log=log)
            print(f"{b:5s} {kind:10s} val {best:.5f} (none {rb['val_none']:.5f}, unsharp {rb['unsharp']['val_mse']:.5f})"
                  f"  {n_params(m)} params", flush=True)
        report[b] = rb
    (out / "unsharp.json").write_text(json.dumps({b: r["unsharp"] for b, r in report.items()}, indent=1))
    (out / "train_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
