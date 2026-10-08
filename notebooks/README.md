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
| `finetune` | Fine-tuning GENCO on solver-labelled contingencies | T4, ~2 h | `finetune` |
| `finetune_as_run` | The exact code that produced the published fine-tuned checkpoint | T4, ~2 h | `finetune` |
| `evaluate_finetuned` | Paired test of the fine-tuned model and the unresolved-case warning (Figure 5) | T4, ~1 h | `evaluate_finetuned` |
| `independent_check` | PYPOWER built from the raw network parameters as an independent check | CPU, ~40 min | `independent_check` |

Inside the notebooks the data sets keep the short tags used while the work was done: `F1` is the frozen test, `K1` the
storms, `T1` the fine-tuning set and `C1` the overload-margin calibration. The tags also appear in some result files.
