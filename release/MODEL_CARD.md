---
license: apache-2.0
base_model: gridfm/genco-pf-contingency
tags: [power-systems, power-flow, contingency-analysis, graph-neural-network, gridfm, genco]
---

# GENCO fine-tuned for clustered contingencies, with an unresolved-case warning

The released GENCO checkpoint (`gridfm/genco-pf-contingency`, `model/training_bs_16`, revision `5c625b0`) fine-tuned with
its own training objective (gridfm-graphkit `MixedLoss`) on 6,737 contingencies of the Texas 2,000-bus grid labelled by a
verified AC solver: random, nearest-line and corridor outages from N-5 to N-20, at the actual load and with loads scaled by
0.85 to 1.35. One outage construction was held out for testing.

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
| `training/finetune.ipynb` | the notebook run that produced the weights |

`model.safetensors` SHA-256: `{SAFETENSORS_SHA256}`. It is converted from the notebook's `genco_finetuned_state_dict.pt`
(SHA-256 `7e9e3c635ea6eca6d3b1dc7009c2ba9e2aa05890fbd9a6f76ab59084d75c57e6`).

```python
from safetensors.torch import load_file
task = get_task(args, [normalizer])        # gridfm-graphkit, caseTexas configuration, as for the base model
task.model.load_state_dict(load_file("model.safetensors"))
```

## Results

Paired against the base model on cases neither model saw, with the protocol fixed before the run:

| | Base model | Fine-tuned |
|---|---|---|
| Decisions missed per case, clustered outages (401 cases) | 6.13 | 4.70 (95 % CI of the difference −1.88 to −1.02) |
| Decisions missed per case, held-out construction (153 cases) | 2.01 | 1.58 (−0.68 to −0.21) |
| Median worst-bus error, released N-2 scenarios (p.u.) | 0.0013 | 0.0035 |

The fine-tuned model is better on clustered and stressed cases and worse on ordinary ones: a specialist, not a replacement
for the base checkpoint.

The warning predicts whether the classical Newton-Raphson policy (pre-contingency, DC and flat starts; tolerance 1e-8 p.u.;
30 iterations) will leave a case unresolved. AUROC 0.96 with the base model and 0.97 with the fine-tuned one on the
870-case frozen test; at its threshold it catches 94 to 95 % of unresolved cases with 11 to 16 % false alarms. It predicts
solver non-convergence, not the absence of an AC solution, and it is close to chance within 5 % of a voltage-collapse point.

## Intended use

Research. Every security decision must come from an AC solver: the model proposes a state and the warning orders the work.
