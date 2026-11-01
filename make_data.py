"""Generate the paired data: for every prompt, the expensive reference and each cheap budget,
all from the same initial noise. Resumable (skips prompts already on disk).

    python make_data.py                      # Wan2.2-TI2V-5B, fp8 storage, 320x576, 33 frames
    python make_data.py --model wan21-1.3b   # faster stand-in
    python make_data.py --tiny               # random miniature Wan, CPU, for testing the pipeline
"""
import os

# transformers would otherwise import TensorFlow/Flax if they are installed (slow, and wastes RAM)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import faulthandler
import json
from pathlib import Path

import numpy as np
import torch

from nopee.wan import LAST, MODELS, parse_budget, TINY, encode_prompts, generate, load_generator, mem, noise_for, tiny_generator

faulthandler.enable()   # a native crash prints where it happened instead of vanishing

ROOT = Path(__file__).resolve().parent


def splits(prompts, n_train, n_val, n_test):
    perm = np.random.default_rng(0).permutation(len(prompts))
    assert n_train + n_val + n_test <= len(prompts), "not enough prompts"
    take = lambda a, b: [int(i) for i in perm[a:b]]
    return dict(test=take(0, n_test), val=take(n_test, n_test + n_val),
                train=take(n_test + n_val, n_test + n_val + n_train))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wan22-5b", choices=list(MODELS))
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--n-val", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=24)
    ap.add_argument("--budgets", default="ref,s6,s12,s24,s12c")
    ap.add_argument("--record", default="s6", help="budget whose trajectories are stored (in-block training)")
    ap.add_argument("--height", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--no-fp8", action="store_true")
    ap.add_argument("--offload", action="store_true", help="model CPU offload if fp8 on GPU does not fit")
    ap.add_argument("--te", default="auto", choices=("auto", "8bit", "fp8", "cpu"),
                    help="text encoder: 8bit (bitsandbytes, default with CUDA), fp8, or cpu")
    ap.add_argument("--only-text", action="store_true", help="stop after phase 1 (encode prompts)")
    ap.add_argument("--splits", default="val,train,test", help="which splits to generate, in this order")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() and not args.tiny else "cpu"
    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    prompts = [l.strip() for l in (ROOT / "prompts.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    sp = splits(prompts, args.n_train, args.n_val, args.n_test)
    cfg = dict(TINY) if args.tiny else dict(MODELS[args.model])
    for k in ("height", "width", "frames"):
        if getattr(args, k):
            cfg[k] = getattr(args, k)
    budgets = args.budgets.split(",")
    meta = dict(model="tiny" if args.tiny else args.model, cfg={k: v for k, v in cfg.items() if k != "id"},
                budgets={b: parse_budget(b) for b in budgets}, splits=sp, record=args.record,
                fp8=not args.no_fp8, prompts={i: prompts[i] for s in sp.values() for i in s})
    if (out / "meta.json").exists():                   # adding budgets later keeps the earlier ones
        old = json.loads((out / "meta.json").read_text())
        meta["budgets"] = {**old.get("budgets", {}), **meta["budgets"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    # phase 1: text (never in memory together with the video transformer)
    emb_path = out / "embeds.pt"
    needed = sorted({i for s in sp.values() for i in s})
    if not emb_path.exists():
        if args.tiny:
            g = torch.Generator().manual_seed(0)
            E = {i: torch.randn(1, cfg["text_len"], cfg["text_dim"], generator=g) for i in needed}
            neg = torch.randn(1, cfg["text_len"], cfg["text_dim"], generator=g)
        else:
            print("phase 1: encoding prompts", flush=True)
            embs, neg = encode_prompts(args.model, [prompts[i] for i in needed], dev, mode=args.te)
            E = dict(zip(needed, embs))
        torch.save(dict(E=E, neg=neg), emb_path)
    em = torch.load(emb_path)
    if args.only_text:
        print(f"phase 1 done: {len(em['E'])} prompts in {emb_path}")
        return

    # phase 2: video
    print("phase 2: loading the video model", flush=True)
    pipe = tiny_generator() if args.tiny else load_generator(args.model, dev, fp8=not args.no_fp8, offload=args.offload)
    first = needed[0]
    generate(pipe, cfg, em["E"][first], em["neg"], noise_for(pipe, cfg, 1, dev), 6)  # warm-up
    for split in args.splits.split(","):
        (out / split).mkdir(exist_ok=True)
        for i in sp[split]:
            f = out / split / f"{i:04d}.pt"
            rec = torch.load(f) if f.exists() else dict(prompt=prompts[i], seed=10_000 + i, lat={}, sec={})
            rec.setdefault("computed", {})
            todo = [b for b in budgets if b not in rec["lat"]]      # resumable, and budgets can be added later
            if not todo:
                continue
            noise = noise_for(pipe, cfg, 10_000 + i, dev)
            for b in todo:
                bb = parse_budget(b)
                lat, sec, traj = generate(pipe, cfg, em["E"][i], em["neg"], noise, bb["steps"], bb.get("cache"),
                                          record=(b == args.record and split != "test"))
                rec["lat"][b] = lat.to("cpu", torch.float16)
                rec["sec"][b] = sec
                rec["computed"][b] = LAST["computed"]
                if traj is not None:
                    rec["traj"] = traj
            torch.save(rec, f)
            print(f"{split} {i:4d}  " + "  ".join(
                f"{b} {rec['sec'][b]:.1f}s" + (f" ({100 * rec['computed'][b]:.0f}% full)" if "c" in b else "")
                for b in todo), flush=True)


if __name__ == "__main__":
    main()
