"""Near the fold: can a warning trained on continuation data tell storms it has never seen apart?

Carries out results/near_fold_warning/protocol.json, written before any result of this analysis was computed. The
published warning is near chance on load points within 5 % of the fold (AUROC 0.572). Here the same model is trained on
those points, with whole storms held out in cross-validation.

    python scripts/near_fold_warning.py --features-dir <folder with features_released.npz>

The feature file is published with the model: https://huggingface.co/Alirezashooshtari/genco-texas-clustered (data/).
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/near_fold_warning"
SET = "K1 points within 5 % of the fold"
N_OUTPUT_FEATURES = 9
FOLDS, BOOT, SEED = 5, 2000, 0
PUBLISHED_AUROC = 0.572


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000))


def out_of_fold(X, y, groups):
    """Scores of every point from a model trained on the other storms (GroupKFold by storm)."""
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=FOLDS).split(X, y, groups):
        s[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    assert not np.isnan(s).any()
    return s


def storm_bootstrap(y, groups, scores):
    """AUROC of each score vector on BOOT resamples of storms (the same resamples for every vector)."""
    rng = np.random.default_rng(SEED)
    storms = np.unique(groups)
    idx = {g: np.nonzero(groups == g)[0] for g in storms}
    out = {k: [] for k in scores}
    for _ in range(BOOT):
        take = np.concatenate([idx[g] for g in rng.choice(storms, len(storms), replace=True)])
        if y[take].min() == y[take].max():
            continue
        for k, s in scores.items():
            out[k].append(roc_auc_score(y[take], s[take]))
    return {k: np.array(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True, type=Path)
    a = ap.parse_args()
    protocol = json.loads((OUT / "protocol.json").read_text())
    f = a.features_dir / "features_released.npz"
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    assert sha == protocol["data"]["sha256"], f"feature file {sha[:12]} is not the one in the protocol"
    z = np.load(f, allow_pickle=False)
    X, y, ids = z[SET + "__X"], z[SET + "__y"].astype(int), z[SET + "__ids"]
    groups = np.array([i.split("@")[0] for i in ids])
    lam = np.array([float(i.split("@")[1]) for i in ids]).reshape(-1, 1)
    assert X.shape == (1226, 777) and len(np.unique(groups)) == 111 and y.sum() == 558

    published = json.loads(
        (ROOT / "results/evaluate_finetuned_seed0/unresolved_warning/warning_released.json").read_text()
    )
    z_pub = (X - np.array(published["scaler_mean"])) / np.array(published["scaler_scale"])
    s_pub = 1 / (1 + np.exp(-(z_pub @ np.array(published["coef"]) + published["intercept"])))
    assert abs(roc_auc_score(y, s_pub) - PUBLISHED_AUROC) < 5e-4

    scores = {
        "published warning (unchanged)": s_pub,
        "trained near the fold: outputs + embeddings (primary)": out_of_fold(X, y, groups),
        "trained near the fold: outputs only": out_of_fold(X[:, :N_OUTPUT_FEATURES], y, groups),
        "trained near the fold: embeddings only": out_of_fold(X[:, N_OUTPUT_FEATURES:], y, groups),
        "trained near the fold: load factor only": out_of_fold(lam, y, groups),
    }
    primary = "trained near the fold: outputs + embeddings (primary)"
    boot = storm_bootstrap(y, groups, scores)
    rows = []
    for k, s in scores.items():
        d = boot[k] - boot[primary]
        rows.append(
            {
                "model": k,
                "auroc": roc_auc_score(y, s),
                "ci_lo": np.percentile(boot[k], 2.5),
                "ci_hi": np.percentile(boot[k], 97.5),
                "diff_vs_primary": roc_auc_score(y, s) - roc_auc_score(y, scores[primary]),
                "diff_ci_lo": np.percentile(d, 2.5),
                "diff_ci_hi": np.percentile(d, 97.5),
            }
        )
    R = pd.DataFrame(rows)
    p = R.set_index("model").loc[primary]
    met = bool(p.auroc >= 0.70 and p.ci_lo > PUBLISHED_AUROC)
    OUT.mkdir(exist_ok=True)
    R.to_csv(OUT / "auroc.csv", index=False, float_format="%.4f")
    pd.DataFrame(
        {"point": ids, "storm": groups, "load_factor": lam[:, 0], "unresolved": y, **{k: v for k, v in scores.items()}}
    ).to_csv(OUT / "scores.csv", index=False, float_format="%.6f")
    summary = {
        "protocol": "results/near_fold_warning/protocol.json",
        "feature_file_sha256": sha,
        "points": int(len(y)),
        "storms": int(len(np.unique(groups))),
        "unresolved": int(y.sum()),
        "bootstrap_resamples_used": int(len(boot[primary])),
        "primary_auroc": float(p.auroc),
        "primary_ci95": [float(p.ci_lo), float(p.ci_hi)],
        "published_auroc": PUBLISHED_AUROC,
        "success_rule": protocol["success"],
        "rule_met": met,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(R.round(3).to_string(index=False))
    print(f"\nrule ({protocol['success']}): {'MET' if met else 'NOT MET'}")


if __name__ == "__main__":
    main()
