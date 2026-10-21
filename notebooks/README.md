# Notebooks

Each notebook is self-contained and runs on a fresh Colab runtime (see `../environment/colab.md`). It writes the
`results/` folder listed below.

| Notebook | What it shows | Runtime | Results |
|---|---|---|---|
| `clustered_outages` | GENCO on random and clustered N-10/N-20 outages (Figure 3a) | T4, ~40 min | `clustered_outages` |
| `load_response` | GENCO's response to single loads and to system-wide load growth (Figure 3b) | T4, ~25 min | `load_response` |
| `storms` | 120 clustered N-20 storms, load sweeps to the fold, what GENCO shows (Figure 2) | T4, ~35 min | `storms` |
| `fold_check` | The fold of every storm by continuation | CPU, ~15 min | `fold_check` |
| `overload_margin` | Conformal overload margin for GENCO and DC power flow | CPU, ~30 min | `overload_margin` |
| `frozen_test` | The frozen test: 870 cases, four workflows, timing (Figure 4) | T4, ~50 min | `frozen_test` |
| `branch_check` | The branch check on the frozen-test cases: which cases have two AC solutions | T4, ~20 min | `branch_check` |
| `finetune` | Fine-tuning GENCO on solver-labelled contingencies, one run per `TRAIN_SEED` (seed 0 is published) | T4, ~2 h | `finetune_seed0`, `finetune_seed1` |
| `finetune_as_run` | The first fine-tuning run, before the gradient-accumulation fix | T4, ~2 h | `finetune_as_run` |
| `evaluate_finetuned` | Paired test of a fine-tuned run and the unresolved-case warning (Figure 5) | T4, ~1 h | `evaluate_finetuned_seed0`, `_seed1`, `_as_run` |
| `finetune_replay` | `finetune` with replay of released N-2 scenarios in every step and selection that keeps ordinary accuracy; decision rule in its header | T4, ~1.7 h | `finetune_replay_seed0` |
| `evaluate_finetuned_replay` | `evaluate_finetuned` reading the replay run | T4, ~30 min | `evaluate_finetuned_replay_seed0` |
| `finetune_loadgrowth` | `finetune` with a load-growth response term: each case paired with the same case at 2 % more load; decision rule in its header | T4, ~2.5 h | `finetune_loadgrowth_seed0`, `_seed1` |
| `evaluate_finetuned_loadgrowth` | `evaluate_finetuned` reading the load-growth runs | T4, ~45 min | `evaluate_finetuned_loadgrowth_seed0`, `_seed1` |
| `load_response_finetuned` | `load_response`, on its 129 published load-growth scenarios, with a fine-tuned model (Figure 3b) | T4, ~25 min | `load_response_seed0`, `load_response_loadgrowth_seed0`, `_seed1` |
| `second_grid` | The verified workflow, GENCO alone and the load-growth probe on IEEE 118, with its released GENCO | CPU, ~15 min | `second_grid` |
| `independent_check` | PYPOWER built from the raw network parameters as an independent check | CPU, ~40 min | `independent_check` |

Inside the notebooks the data sets keep the short tags used while the work was done: `F1` is the frozen test, `K1` the
storms, `T1` the fine-tuning set and `C1` the overload-margin calibration. The tags also appear in some result files.

Some names in the result files date from the first runs and are kept so that the files stay as the notebooks wrote
them: `solvable_at_1`, `unsolvable` and `no solution` there mean that none of the tested Newton starts converged at
that load, not that no AC solution exists. Where a fold is reported, it comes from the continuation in `fold_check`.
