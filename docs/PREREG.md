# NopeeClaude — preregistration

Committed before any run on a real Wan model. The only runs before this file were on a randomly
initialised miniature Wan (`--tiny`), used to check that the code works. Its numbers say nothing
about Wan.

## The question

The Biological Computing Co. (TBC) says "mischief 1" is Wan 2.2 plus a modification adding less
than 0.1% parameters. They say it generates 5× faster at matching quality, and that the
modification is derived from recordings of living neurons. Their blog describes the Oasis
(Minecraft) prototype behind it:
- a *hypercolumn adapter*: small residual branches added at the transformer blocks, starting with
  almost no effect;
- per location, a difference-of-Gaussians spatial kernel with space constant λ and a leaky
  excitatory/inhibitory recurrence with time constant τ;
- λ and τ fitted to how evoked activity spreads and decays on a multi-electrode array.

Their speed comes from a streaming cache, a higher-order sampler (DPM++), fewer steps and serving.
In the prototype, the adapter's job is to keep quality while those make generation cheaper. No
equations, constants, loss or ablation of the biological source are published.

Our version, sized for one RTX 3060 (12 GB): **can a tiny centre-surround + E/I adapter recover
what Wan2.2-TI2V-5B loses when it is sampled cheaply? If it can, is it the structure that does
it, the constants, or plain sharpening?**

**What we cannot test:** their recordings. We have no living neurons. The "fixed" constants are
hand-chosen to look like biology:
- centre sizes 0.5–1.8 latent pixels;
- surround 3× the centre;
- τ_E 0.7–6 latent frames, with τ_I = τ_E/2.

They are not measured. Nothing here can show that neuron recordings help, or that they don't.

## Setup (fixed now)

**Model.** Wan2.2-TI2V-5B (diffusers), text-to-video, 320×576, 33 frames, guidance 5. Transformer
weights are stored in fp8 and computed in bf16. Prompts are encoded in a separate phase.

**Data.**
- 98 prompts in `prompts.txt`, split by a fixed permutation (seed 0) into 24 test, 12 validation
  and 48 training prompts.
- Every condition of a prompt starts from the same initial noise (seed 10000 + prompt index).
- Reference: 40 UniPC steps. Cheap budgets: 6, 12 and 24 steps, plus 12 steps with FirstBlockCache
  (threshold 0.10).

**Arms, all acting on the final latent of a cheap run.** One adapter is trained per budget on the
training prompts, with early stopping on validation and 1500 steps, Adam 2e-3, batch 4. The loss
is MSE to the reference latent.

| arm | what it is |
|---|---|
| `none` | the cheap latent as is |
| `unsharp` | training-free sharpening, x + g(x − G_σ x); σ and g grid-searched on validation |
| `hc_learned` | in-projection → DoG (λ, κ learned) → GELU → leaky E/I (τ_E, τ_I learned) → zero-init out-projection |
| `hc_fixed` | the same, with λ, κ, τ frozen at the hand-chosen prior (gains and projections trained) |
| `hc_random` | the same, with λ, κ, τ frozen at random log-uniform values |
| `free` | the same projections and slots, kernels unconstrained (free 7×7 spatial and length-4 causal temporal kernel per channel; about 1.4× the parameters) |

**Primary measure.** Latent MSE to the same-noise reference on the 24 test prompts. The 95% CIs
are paired bootstraps over prompts (10,000 resamples, seed 0).

## Gates

| Gate | Claim | Pass condition (budget s6) |
|---|---|---|
| **G0** | *instrument:* the cheap budget really is cheap | median time ref / s6 ≥ 4 |
| **G1** | the adapter recovers part of what cheap sampling loses | `hc_learned` vs `none`: ≥ 10% lower error, CI excludes 0 |
| **G2** | it buys a budget, like TBC's "6 steps match 10" | mean error of s6 + `hc_learned` ≤ mean error of s12 `none` |
| **G3** | the centre-surround/E-I *structure* matters | `hc_learned` vs `free`: ≥ 5% lower error, CI excludes 0 |
| **G4** | the *constants* matter | `hc_fixed` vs `hc_random`: ≥ 5% lower error, CI excludes 0 |
| **G5** | it is more than sharpening | `hc_learned` vs `unsharp`: ≥ 5% lower error, CI excludes 0 |

**Report only.**
- The same comparisons at s12, s24 and s12c.
- Decoded-frame PSNR to the reference, and excess flicker (`evaluate.py --decode`).
- The effective speedup of s6 + adapter against s12.

## Stage 2, exploratory (no gates)

This is TBC's actual shape: branches on the outputs of transformer blocks 0–3 (rank 32, about
0.8M parameters, about 0.02% of the 5B transformer), trained by distillation through the frozen
transformer.
- **Training.** From a state on the cheap run's own trajectory, the guided end point
  x0 = x_t − σv should land on the reference latent. The loss is weighted by (1 − σ). Trajectories
  are re-recorded with the adapter on every 200 steps. Training runs 600 steps at lr 3e-4.
- **Model selection.** Regenerate the validation prompts and measure final-latent MSE.
- **Disclosure.** The (1 − σ) weighting, the refresh and the learning rate were chosen on the tiny
  random model, after the unweighted version made regenerated output worse there (0.0033 →
  0.0071). The weighted version improved it (0.0033 → 0.0028).
- **Reporting.** Regenerated test latent MSE against `none` and against the final-latent adapter.

## My predictions (Claude, before any real run)

- **G0 PASS.** 40 → 6 steps should give about 6×.
- **G1 PASS.** Few-step flow sampling misses fine detail in a systematic way, and a small filter
  can learn a systematic correction.
- **G2 uncertain, leaning FAIL.** s12 is usually much closer to the reference than s6.
- **G3 FAIL.** With enough data, free kernels do at least as well as constrained ones. The
  constraint helps only if data are scarce. 48 prompts is scarce, so this could surprise.
- **G4 FAIL.** The constants are hand-chosen, and the gains and projections can compensate.
- **G5 uncertain.** If it fails, the adapter is mostly a learned sharpening filter. That would
  still be a useful thing to know about what "keeping quality with fewer steps" means.

## What a pass would and would not mean

- Matching a same-noise reference is a proxy for "matching quality". This is not FVD and not a
  human study.
- 24 test prompts at one resolution is small.
- Public few-step distillations of Wan exist (LightX2V; FastVideo's FastWan2.2-TI2V-5B, distilled
  to a few steps). Beating plain fewer-step UniPC does not beat those.

## Trial run (Nov 1 2026, timing only, recorded before the full run)

**What was run.** `make_data.py --n-train 2 --n-val 1 --n-test 2 --out data_trial` on the RTX 3060,
5 prompts. Nothing was decoded, trained or evaluated.

**Timings, seconds per prompt** (identical within ±2% across the 5 prompts):

| ref (40 steps) | s6 | s12 | s24 | s12c (12 steps + cache 0.10) |
|---|---|---|---|---|
| 90–94 | 13.6–14.0 | 27.1–27.9 | 54.2–56.0 | 27.1–28.2 |

**What this shows.**
- **Cost is linear in steps:** 2.3 s per step, i.e. about 1.15 s per transformer pass with
  guidance. ref/s6 = 6.6, so G0 should pass.
- **The cache budget as preregistered does nothing.** s12c takes the same time as s12, so
  FirstBlockCache at threshold 0.10 skipped no work at 12 steps. (The skip counter, which counts
  whether the last block's feed-forward ran, was added after this trial.)
- Peak RAM was 12.9 of 23.8 GB with the old loader. The tensor-by-tensor loader (added after) cuts
  this.

**Amendment 1 (decided now, before any full-run data).**
- The cache budget is taken out of the first full pass.
- Its threshold will be chosen with `cache_sweep.py` on 3 *validation* prompts only. The sweep
  covers thresholds 0.05–0.3 at 12 and 40 steps, and records time, the share of calls that ran
  every block, and closeness to the reference.
- The rule: the highest threshold at 12 steps whose MSE to the reference is no worse than s6's
  `none`, with at least a 20% time saving. If none qualifies, the cache budget is dropped and that
  is reported.
- The chosen budget (`s12c<threshold>`) is then added to all splits with `make_data.py --budgets`.
  It stays report-only, as before.
- The trial prompts (indices 80, 36, 20, 5, 23) are all *test* prompts of the full split. They were
  only timed, never looked at, and are not used for the sweep.

## Validation data and cache sweep (Nov 1 2026, before any train/test data or adapter)

**Generated.** All 12 validation prompts with ref, s6, s12 and s24. Timings repeat the trial:
92 / 13.9 / 27.7 / 55.4 s.

**Cache sweep** (`cache_sweep.py`; 3 validation prompts; MSE of normalised latents to the 40-step
reference):

| run | seconds | calls that ran every block | MSE to ref |
|---|---|---|---|
| 12 steps, no cache | 27.7 | 100% | 1.396 |
| 12 steps, cache 0.05 / 0.1 | 27.8 / 28.1 | 100% | 1.396 |
| 12 steps, cache 0.2 | 22.0 | 78% | 1.337 |
| 12 steps, cache 0.3 | 17.5 | 58% | 1.193 |
| 12 steps, cache 0.5 | 13.1 | 43% | 1.655 |
| 40 steps, cache 0.05 | 78.4 | 83% | 0.038 |
| 40 steps, cache 0.1 | 54.2 | 55% | 0.158 |
| 40 steps, cache 0.2 | 36.0 | 33% | 0.444 |
| 40 steps, cache 0.3 | 28.3 | 24% | 0.484 |
| 40 steps, cache 0.5 | 21.5 | 16% | 0.947 |

**Two observations that bear on the design.**
1. **The 12-step run is not a rougher copy of the reference.** An MSE of about 1.4 on normalised
   latents is a large fraction of what two unrelated videos would give. If the cheap runs land on
   *different* videos, not blurrier versions of the same one, then a small local filter trained
   with MSE cannot recover the difference. It can only shrink toward an average, which lowers MSE
   while making the video worse. G1 could then "pass" for the wrong reason. `inspect_data.py`
   measures this; the result decides Amendment 2.
2. **At matched time, caching beats cutting steps.** 40 steps with cache 0.3 (28.3 s) stays about
   3× closer to the reference than 12 steps (27.7 s). 40 steps with cache 0.5 is 4.3× faster than
   the reference and still closer than 12 steps. This is report-only, n = 3. It fits TBC's own
   account: the speed comes from caching, and the adapter keeps quality.

**Amendment 2 (pending; decided before any train/test data or adapter training).** If
`inspect_data.py` shows the step-cut runs are far from their references (share > 0.5), the
primary cheap budget for the gates changes:
- from s6 to a cached 40-step run (`s40c0.3`, about 3.2× faster);
- s12 `none` is replaced in G2 by `s40c0.1` `none`;
- s6 and s12 are kept, report-only.

The final choice and its reason will be written here before the next data are generated.

### Amendment 2: resolved (Nov 1 2026 evening): main run paused, not switched

**`inspect_data.py`, 12 validation prompts** (share = MSE to ref / MSE between unrelated
references, which is 4.19):

| budget | share | layout (4× pooled) share | correlation |
|---|---|---|---|
| s6 | 0.42 | 0.36 | 0.67 |
| s12 | 0.32 | 0.26 | 0.75 |
| s24 | 0.21 | 0.14 | 0.83 |

**Decoded previews** (4 validation prompts): PSNR to ref is 8–14.5 dB at s6, 10–17.4 dB at s12 and
12–18.6 dB at s24.

**By the rule written in advance (switch if share > 0.5), Amendment 2 does not trigger.** The
previews show two things the rule did not anticipate:
- **The cheap runs are broken, not rougher.** At s6 and s12 the subject is often missing, and
  sometimes different content appears.
- **The 40-step reference is poor at 320×576.** It is oversaturated and streaky, because the model
  was trained at 704×1280.

Neither switching the budget nor running the gates at this resolution would test what the
preregistration meant to test. **The main run is paused.** No train or test data were generated
and no adapter was trained. The gates G0–G5 are not evaluated. If the experiment resumes, it does
so under a new preregistration (a cached-40-step primary budget, at a resolution where the
reference is checked first). It does not resume under this one.
