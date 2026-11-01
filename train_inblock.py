"""Stage 2 (exploratory): TBC's actual shape -- small parallel branches on the outputs of the first
transformer blocks -- trained by distillation from the reference.

From a state x_t on the cheap run's own trajectory (re-recorded with the adapter on every --refresh steps), the guided model predicts velocity v and so an
end point x0 = x_t - sigma * v (flow matching). The branches are trained so that this end point lands
on the reference run's final latent. Gradients flow back through the frozen 5B transformer
(gradient checkpointing, weights kept in fp8). The transformer itself is never changed.
Model selection uses the real quantity: regenerate the val prompts with the adapter on and measure
the distance of the final latent to the reference.

    python train_inblock.py [--blocks 0,1,2,3] [--rank 32] [--steps 600]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nopee.adapters import BlockAdapters, n_params
from nopee.data import load_split
from nopee.wan import TINY, MODELS, generate, load_generator, model_call, noise_for, tiny_generator

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="adapters")
    ap.add_argument("--kind", default="hc_learned", choices=("hc_learned", "hc_fixed", "hc_random", "free"))
    ap.add_argument("--blocks", default="0,1,2,3")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--weight", default="late", choices=("late", "flat"),
                    help="late: weight the end-point loss by (1 - sigma); from pure noise the end point is not knowable")
    ap.add_argument("--refresh", type=int, default=200,
                    help="every N steps re-record the training trajectories WITH the adapter on (on-policy)")
    args = ap.parse_args()

    meta, tr = load_split(ROOT / args.data, "train", keep_traj=True)
    _, va = load_split(ROOT / args.data, "val", keep_traj=True)
    tiny = meta["model"] == "tiny"
    dev = "cuda" if torch.cuda.is_available() and not tiny else "cpu"
    cfg = dict(TINY) if tiny else dict(MODELS[meta["model"]], **meta["cfg"])
    pipe = tiny_generator() if tiny else load_generator(meta["model"], dev, fp8=meta["fp8"], offload=args.offload)
    em = torch.load(ROOT / args.data / "embeds.pt")
    T = pipe.transformer
    T.requires_grad_(False)
    T.enable_gradient_checkpointing()
    dim = T.config.num_attention_heads * T.config.attention_head_dim
    ad = BlockAdapters(dim, tuple(int(b) for b in args.blocks.split(",")), args.rank, args.kind).to(dev)
    ad.install(T)
    opt = torch.optim.Adam([p for p in ad.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    rng = np.random.default_rng(0)
    g = cfg["guidance"]

    def x0_loss(r, k, grad):
        x, t, sig = r["traj"][k]
        x = x.float().to(dev)
        with torch.set_grad_enabled(grad):
            v = model_call(pipe, x, t, em["E"][r["idx"]], em["neg"], g)
            w = (1 - sig) if args.weight == "late" else 1.0
            return w * ((x - sig * v - r["lat"]["ref"].float().to(dev)) ** 2).mean()

    steps_b = meta["budgets"][meta["record"]]["steps"]

    def run(r, record=False):
        noise = noise_for(pipe, cfg, r["seed"], dev)
        return generate(pipe, cfg, em["E"][r["idx"]], em["neg"], noise, steps_b, record=record)

    def val():
        """What actually matters: regenerate the val prompts with the adapter on, distance to the reference."""
        T.eval()
        return float(np.mean([((run(r)[0].cpu() - r["lat"]["ref"].float()) ** 2).mean().item() for r in va]))

    def refresh():
        for r in tr:
            r["traj"] = run(r, record=True)[2]

    v0 = val()
    best, best_state, log = v0, {k: v.clone() for k, v in ad.state_dict().items()}, []
    print(f"in-block {args.kind}: {n_params(ad)} params ({100 * n_params(ad) / sum(p.numel() for p in T.parameters()):.4f}% "
          f"of the transformer); val latent MSE before {v0:.5f}", flush=True)
    for s in range(1, args.steps + 1):
        if args.refresh and s > 1 and (s - 1) % args.refresh == 0:
            refresh()
        r = tr[rng.integers(len(tr))]
        k = int(rng.integers(len(r["traj"])))
        loss = x0_loss(r, k, True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 100 == 0 or s == args.steps:
            v = val()
            log.append((s, loss.item(), v))
            print(f"  step {s}: train {loss.item():.5f}  val {v:.5f}", flush=True)
            if v < best:
                best, best_state = v, {k_: v_.clone() for k_, v_ in ad.state_dict().items()}
    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    torch.save(dict(kind=args.kind, dim=dim, blocks=ad.blocks, rank=args.rank, state={k: v.cpu() for k, v in best_state.items()}),
               out / f"inblock_{meta['record']}_{args.kind}.pt")
    (out / f"inblock_{meta['record']}_{args.kind}.json").write_text(json.dumps(dict(val_before=v0, val_best=best, log=log,
                                                                                     params=n_params(ad)), indent=1))


if __name__ == "__main__":
    main()
