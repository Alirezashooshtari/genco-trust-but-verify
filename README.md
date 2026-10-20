# Trust, but verify: making GENCO safe for N-k contingency screening

[![ci](https://github.com/Alirezashooshtari/genco-trust-but-verify/actions/workflows/ci.yml/badge.svg)](https://github.com/Alirezashooshtari/genco-trust-but-verify/actions/workflows/ci.yml)

Alireza Shooshtari, Institut de Recerca en Energia de Catalunya (IREC) · a.shooshtari1996@gmail.com

Entry to the [GENCO Challenge](https://huggingface.co/spaces/gridfm/genco-challenge) (GridFM Porto 2026).
**Start here:** [write-up](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/7) ·
[fine-tuned model and warning](https://huggingface.co/Alirezashooshtari/genco-texas-clustered) ·
[case explorer](https://huggingface.co/spaces/Alirezashooshtari/genco-contingency-verifier) ·
[notebooks and results](notebooks/README.md)

GENCO predicts the AC power flow of a contingency in one forward pass, which makes it attractive for screening
thousands of N-k cases. The cases that matter most are the rare, severe ones, and there a confident wrong answer is the
worst outcome. This project measures where GENCO fails on the released Texas 2,000-bus grid, wraps it in a verification
step that always ends in an AC solution or an explicit alarm, and extends GENCO with fine-tuning and a warning head.

![The verified workflow](figures/fig1_workflow.png)

## Results

On outages clustered in one area of the grid, of the kind storms and wildfires cause, GENCO's voltage errors grow 4 to
17 times compared with random outages of the same size. On 33 of 120 clustered N-20 cases no Newton start converges at the
actual load, yet GENCO returns a full state for every one. With verification, every case ends in an AC decision or an
alarm:

| Frozen test | Texas 2,000-bus | IEEE 118 |
|---|---|---|
| Contingencies | 870 | 708 |
| Resolved by a classical solver: AC decision from the workflow | 748 of 748 | 679 of 679 |
| Not resolved: alarm | 122 | 29 |
| False alarms¹ | 0 | 0 |
| GENCO alone: resolved cases with at least one missed decision | 260 (35 %) | 356 (52 %) |
| GENCO's voltage error on clustered outages, relative to random ones² | ×4.0 (N-10), ×16.5 (N-20) | ×2.6 (N-3 and N-6) |
| GENCO's voltage drop under uniform load growth, relative to the AC drop³ | 0.60 | 0.83 |

¹ Cases the workflow marks unresolved although a classical solver finds an AC solution.
² Median worst-bus voltage error. ³ Median slope over the operating points.

On IEEE 118 the model is the one the authors released for that grid, and the operating points come from their test split.

Fine-tuning the released checkpoint on 6,737 solver-labelled contingencies cuts missed decisions on unseen clustered
outages from 6.13 to 4.79 per case (−22 %; −20 % with a second training seed). The fine-tuned model is a specialist, so
a rule on the outage alone (at least 5 lines out, spread over at most 11 hops) sends clustered outages to it and leaves
ordinary ones with the released model. Replaying ordinary scenarios during fine-tuning did not remove the trade-off:
none of its 20 checkpoints kept ordinary accuracy within 10 % of the released model (`notebooks/finetune_replay.ipynb`).
A warning head on GENCO's embeddings flags the cases the solver will not resolve, with AUROC 0.96.

## Quick start

```bash
git clone https://github.com/Alirezashooshtari/genco-trust-but-verify.git
cd genco-trust-but-verify
pip install -r requirements.txt && pip install -e .

pytest                                # tests of the verification module (no data needed)
python scripts/headline_numbers.py    # every number in the write-up, recomputed from results/
python scripts/make_figures.py        # figures 2 to 5 (figure 1 is a draw.io diagram)
python scripts/build_demo.py          # the case explorer, demo/index.html
```

`headline_numbers.py` writes `results/headline_numbers.csv`, one row per claim with the file it comes from. CI recomputes
it on every push and fails if any value changes.

## Using the verification module

```python
from genco_verify import Verifier, case_from_tables

case = case_from_tables(bus, branch, removed_idx=lines_out)     # one scenario of the released tables, lines removed
result = Verifier(case).run(genco_vm=vm, genco_va_deg=va)        # vm, va: GENCO's predicted state

result.status             # "verified_safe" | "verified_violation" | "unresolved"
result.limits             # overloaded lines and voltage violations of the AC state
result.unresolved_reason  # why no start converged, when unresolved
result.history            # every start tried: iterations, mismatch, time
```

`Verifier(case, {"enforce_q_limits": True})` enforces generator reactive limits (pass the generator table to
`case_from_tables`), and `{"branch_check": True}` runs every start and reports when they reach different AC solutions.
`policy_name="classical"` runs the same check without GENCO. `notebooks/frozen_test.ipynb` shows how the tables and
GENCO's state of a released scenario are loaded. Newton-Raphson uses lightsim2grid when it is installed
(`pip install -e ".[fast]"`) and a SciPy implementation otherwise.

## Layout

```
src/genco_verify/   the verification workflow (numpy, scipy; lightsim2grid optional)
tests/              tests of every outcome and option on a three-bus grid
notebooks/          the Colab notebooks that produced the results (see notebooks/README.md)
results/            what the numbers and figures need from each notebook run, plus the analyses in results/analysis/
scripts/            headline numbers, figures, demo, warning ablation, Hugging Face upload
figures/            the five figures of the write-up (figure 1 is a draw.io file)
demo/               the case explorer (also the Hugging Face Space)
environment/        Colab setup, pinned upstream revisions, data and checkpoint hashes
release/            the model card of the fine-tuned model
```

## Reproducing from scratch

Each notebook runs on a fresh Colab runtime from the public data and checkpoint, with gridfm-graphkit pinned to commit
`95b90ee` and `gridfm/genco-pf-contingency` to revision `5c625b0`. `environment/colab.md` describes the setup and what
reproduces exactly; `notebooks/README.md` maps each notebook to its results folder and figure. The warning ablation also
needs the feature matrices published with the model:

```bash
python scripts/warning_ablation.py --features-dir <folder with features_released.npz and features_finetuned.npz>
```

## Scope

The check is a steady-state AC power flow; dynamics, protection and remedial actions are outside it. *Unresolved* means
that none of the tested Newton starts converged, not that no AC solution exists; for the storms, continuation places the
fold of the AC solution below the actual load in 32 of 33 cases. On Texas, GENCO does not make screening faster (86 ms
per case with verification, 13 ms for the classical solve, on a T4 GPU with a one-thread KLU solver; see
`results/frozen_test/timing.json`). Within 5 % of the fold the warning's AUROC falls to 0.57, so the solver still runs
on every case.

## Credits and licence

GENCO, gridfm-graphkit and the Texas test data are by the GridFM team (`gridfm/genco-pf-contingency`, Apache-2.0).
lightsim2grid is by RTE France; PYPOWER, a port of MATPOWER, is used for the independent check. The code in this repository
is MIT licensed; `results/` holds numbers derived from the public data, not the data itself.

If you use this work, please cite it as described in `CITATION.cff`.
