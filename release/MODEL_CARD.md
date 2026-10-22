---
license: apache-2.0
base_model: gridfm/genco-pf-contingency
tags: [power-systems, power-flow, contingency-analysis, graph-neural-network, gridfm, genco]
---

# GENCO fine-tuned for clustered contingencies, with an unresolved-case warning

The released GENCO checkpoint (`gridfm/genco-pf-contingency`, `model/training_bs_16`, revision `5c625b0`) fine-tuned with
its own training objective (gridfm-graphkit `MixedLoss`) on 6,737 contingencies of the Texas 2,000-bus grid labelled by a
verified AC solver: random, nearest-line and corridor outages from N-5 to N-20, at the actual load and with loads scaled by
0.85 to 1.35. One outage construction was held out for testing. These are the weights of training seed 0. A second seed
was trained on the same cases (its weights are in `seed1/`); an earlier run, made before a fix to gradient accumulation, is kept for the record.

Code, notebooks and results: [github.com/Alirezashooshtari/genco-trust-but-verify](https://github.com/Alirezashooshtari/genco-trust-but-verify) ·
Demo: [genco-contingency-verifier](https://huggingface.co/spaces/Alirezashooshtari/genco-contingency-verifier)

Alireza Shooshtari, Institut de Recerca en Energia de Catalunya (IREC)

## Files

| File | Content |
|---|---|
| `model.safetensors` | fine-tuned weights for gridfm-graphkit's `GNS_heterogeneous` (`caseTexas` configuration) |
| `normalizer_stats.pt` | the normalizer of the base model, unchanged |
| `config.json` | base model, code revision, training settings and file hashes |
| `warning/warning_released.joblib`, `warning/warning_finetuned.joblib` | the warning head (scikit-learn `StandardScaler` + `LogisticRegression`) for each GENCO model |
| `warning/warning_released.json`, `warning/warning_finetuned.json` | the same heads as plain numbers: scaler, coefficients, intercept, threshold |
| `warning/features.json` | the 777 input features in order (9 GENCO outputs, 384 + 384 embedding mean and max) |
| `warning/scores.csv` | the score and label of every evaluated case |
| `data/features_released.npz`, `data/features_finetuned.npz` | features and labels of every training, validation and test case |
| `training/finetune.ipynb` | the notebook that produced the weights (`TRAIN_SEED = 0`) |
| `loadgrowth/model.safetensors` | the same fine-tuning plus a load-growth response term (see below) |
| `training/finetune_loadgrowth.ipynb` | the notebook that produced those weights (`TRAIN_SEED = 0`) |
| `seed1/model.safetensors`, `loadgrowth/seed1/model.safetensors` | the second training seed (`TRAIN_SEED = 1`) of each, with the results reported below |

The `.joblib` heads were saved with scikit-learn 1.6.1 (the Colab version at the time); with other versions use the
JSON files, which hold the same parameters as plain numbers.

`model.safetensors` SHA-256: `a20b05e4dd4f1739fd8170596fc71b89eed305bc4ca9383f59dba5e31da75352`. It is converted from the notebook's `genco_finetuned_state_dict.pt`
(SHA-256 `cf705d23b87cc0fd1b6ede6a9028925584bd9b00b936a76244a90a75fcb353a2`).

Loading follows the base model: build the gridfm-graphkit task from the `caseTexas` configuration and its normalizer,
then load these weights in place of the base checkpoint. In outline:

```python
from safetensors.torch import load_file
# args, normalizer: the caseTexas configuration and normalizer of the base model (gridfm-graphkit, commit 95b90ee)
task = get_task(args, [normalizer])
task.model.load_state_dict(load_file("model.safetensors"))
```

`notebooks/evaluate_finetuned.ipynb` in the code repository does this end to end, from installation to predictions.

## Results

Paired against the base model on cases neither model saw, with the protocol fixed before the run:

| | Base model | Fine-tuned (seed 0) | Seed 1 / first run, before the fix |
|---|---|---|---|
| Decisions missed per case, clustered outages (401 cases) | 6.13 | 4.79 (95 % CI of the difference −1.72 to −0.97) | 4.88 / 4.70 |
| Decisions missed per case, held-out construction (153 cases) | 2.01 | 1.56 (−0.66 to −0.24) | 1.59 / 1.58 |
| Decisions missed per case, random outages (347 cases) | 0.14 | 0.14 (−0.03 to +0.04) | 0.15 / 0.18 |
| Median worst-bus error, released N-2 scenarios (p.u.) | 0.0013 | 0.0036 | 0.0043 / 0.0035 |

The fine-tuned model misses fewer decisions on clustered and held-out outages and has a lower worst-bus voltage error on
clustered outages (median 0.017 to 0.012 p.u.), but it is less accurate on ordinary scenarios: a specialist, not a
replacement for the base checkpoint. A rule that needs only the outage itself picks the right model: use this one when
at least 5 lines are out and the buses they touch are at most 11 hops apart, the base model otherwise. On the frozen test
it sends 400 of the 401 clustered outages here and none of the 347 random ones, and on random N-2, N-10 and N-20 outages
it picked this model in none of 2,000 draws each.

The warning predicts whether the classical Newton-Raphson policy (pre-contingency, DC and flat starts; tolerance 1e-8 p.u.;
30 iterations) will leave a case unresolved. AUROC 0.96 with the base model and 0.98 with the fine-tuned one on the
870-case frozen test; at its threshold it catches 95 to 98 % of unresolved cases with 10 to 16 % false alarms. It predicts
solver non-convergence, not the absence of an AC solution, and it is close to chance within 5 % of the fold of the tracked AC solution.

## With a load-growth response term (`loadgrowth/`)

The base model's voltage drops under system-wide load growth are only 0.60 of the AC ones, which is why it glides past
the fold of the AC solution. `loadgrowth/model.safetensors` is fine-tuned on the same cases with the same objective, plus
one term: every training case is paired with the same case at 2 % more load (loads only), and the difference between
GENCO's and the AC voltage change at the PQ buses is penalised. The decision rule was fixed before the run; both seeds
meet it.

| | Base model | Fine-tuned (`model.safetensors`) | Load-growth, seed 0 (`loadgrowth/`) | Load-growth, seed 1 |
|---|---|---|---|---|
| Load-growth response on 129 held-out scenarios (AC = 1) | 0.60 | 0.78 | 0.99 | 1.01 |
| Decisions missed per case, clustered outages | 6.13 | 4.79 | 4.45 (95 % CI of the difference −2.23 to −1.23) | 4.22 |
| Decisions missed per case, held-out construction | 2.01 | 1.56 | 1.35 | 1.24 |
| Decisions missed per case, random outages | 0.14 | 0.14 | 0.21 | 0.14 |
| Median worst-bus error, released N-2 scenarios (p.u.) | 0.0013 | 0.0036 | 0.0052 | 0.0056 |

It is the stronger specialist and responds to load growth like the AC solution, but it is further from the base model on
ordinary scenarios, so the same routing rule applies: with it, missed decisions per clustered case fall to 4.45 and those
on random outages stay at 0.14. Near the fold it still follows only part of the AC collapse (14 % of the AC voltage drop
over the last 5 % of load, against 5 % for the base model), and Newton never converges beyond a fold from its state.

`loadgrowth/model.safetensors` SHA-256: `584b15d658fa7dd97b606c5d78ade0f40ae6bbbf6d8f5fe631fbe072c40763eb`, converted from
the notebook's `genco_finetuned_state_dict.pt` (SHA-256 `9d3f671bf4c171b9327b52e9f100b01c140f76f096fdc35e1318aa3b5401c8b6`).
It loads in the same way as `model.safetensors`.

The second seeds load in the same way: `seed1/model.safetensors` SHA-256
`20143b739aef9dca29733a7f92ec6362dd048450adfe53d0d4793463d33552c3` (from `544185154b000f39bbfa53cf7ed1ac6e92fdebf47092f4b864973e418c38d14b`),
`loadgrowth/seed1/model.safetensors` SHA-256 `cc855f15be3f90257d9850f621ca8b81867a7ed5cc960e414de9d92505b81a09`
(from `7bf1ad7b445f7431d7d69e9c7e573cd256e3fad7bd3ce8ebb823761d6130e37c`).

## Intended use

Research. Every security decision must come from an AC solver: the model proposes a state and the warning orders the work.
