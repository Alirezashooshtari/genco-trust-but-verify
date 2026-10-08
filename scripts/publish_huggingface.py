"""Creates the Hugging Face model repository and the demo Space (private unless --public).

    pip install huggingface_hub safetensors torch
    huggingface-cli login
    python scripts/publish_huggingface.py --user <account> --finetune finetune.zip --evaluation evaluate_finetuned.zip

--finetune and --evaluation are the zip files written by notebooks/finetune.ipynb and notebooks/evaluate_finetuned.ipynb.
The weights are checked against the SHA-256 recorded in results/finetune/summary.json, converted to safetensors and
uploaded together with the warning heads, the feature matrices, the model card and the training notebook.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
BASE_REPO = "gridfm/genco-pf-contingency"
BASE_REVISION = "5c625b00f5b79c8a7d4d27ecc133d4ecdf573af7"
GRAPHKIT_COMMIT = "95b90eecc7f5c2c86a9dd776f0c3f40434def22f"

WARNING_FILES = {  # file written by the evaluation notebook -> path in the model repository
    "warning_released.joblib": "warning/warning_released.joblib",
    "warning_fine_tuned.joblib": "warning/warning_finetuned.joblib",
    "warning_released.json": "warning/warning_released.json",
    "warning_fine_tuned.json": "warning/warning_finetuned.json",
    "feature_definitions.json": "warning/features.json",
    "per_case_scores.csv": "warning/scores.csv",
    "features_released.npz": "data/features_released.npz",
    "features_fine_tuned.npz": "data/features_finetuned.npz",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unzip(path, into):
    with zipfile.ZipFile(path) as z:
        z.extractall(into)
    return into


def build_model_folder(finetune_zip, evaluation_zip, out):
    tmp = Path(tempfile.mkdtemp())
    ft = unzip(finetune_zip, tmp / "finetune")
    ev = unzip(evaluation_zip, tmp / "evaluation")
    summary = json.loads((ROOT / "results/finetune/summary.json").read_text())

    weights = next(ft.rglob("genco_finetuned_state_dict.pt"))
    if sha256(weights) != summary["checkpoint_sha256"]:
        raise SystemExit(f"{weights.name}: SHA-256 does not match results/finetune/summary.json")
    state = torch.load(weights, map_location="cpu")
    save_file({k: v.contiguous() for k, v in state.items()}, out / "model.safetensors")

    warning_dir = next(ev.rglob("unresolved_warning"))
    for src, dst in WARNING_FILES.items():
        if not (warning_dir / src).exists():
            raise SystemExit(f"{src} is missing from {evaluation_zip}")
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(warning_dir / src, out / dst)

    normalizer = hf_hub_download(BASE_REPO, "model/training_bs_16/normalizer_stats.pt", revision=BASE_REVISION)
    shutil.copy(normalizer, out / "normalizer_stats.pt")
    (out / "training").mkdir()
    shutil.copy(ROOT / "notebooks/finetune_as_run.ipynb", out / "training/finetune.ipynb")

    protocol = summary["frozen_protocol"]
    config = {
        "base_model": {"repo": BASE_REPO, "revision": BASE_REVISION, "checkpoint": "model/training_bs_16"},
        "code": {
            "repo": "https://github.com/gridfm/gridfm-graphkit",
            "branch": "genco-paper-repro",
            "commit": GRAPHKIT_COMMIT,
        },
        "model_class": "gridfm_graphkit GNS_heterogeneous, caseTexas configuration (unchanged from the base model)",
        "objective": protocol["objective"],
        "optimizer": protocol["optimiser"],
        "training": {
            "cases": summary["cases"]["regression_targets_train"],
            "validation_cases": summary["cases"]["regression_targets_val"],
            "optimizer_steps": summary["optimizer_steps"],
            "selected_step": summary["selected_step"],
            "minutes_on_t4": round(summary["training_minutes"]),
        },
        "sha256": {
            "genco_finetuned_state_dict.pt": summary["checkpoint_sha256"],
            "model.safetensors": sha256(out / "model.safetensors"),
        },
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    card = (
        (ROOT / "release/MODEL_CARD.md")
        .read_text()
        .replace("{SAFETENSORS_SHA256}", config["sha256"]["model.safetensors"])
    )
    (out / "README.md").write_text(card)
    shutil.rmtree(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Hugging Face account")
    ap.add_argument("--finetune", required=True, help="zip written by notebooks/finetune.ipynb")
    ap.add_argument("--evaluation", required=True, help="zip written by notebooks/evaluate_finetuned.ipynb")
    ap.add_argument("--model-name", default="genco-texas-clustered")
    ap.add_argument("--space-name", default="genco-contingency-verifier")
    ap.add_argument("--public", action="store_true", help="create public repositories (default: private)")
    args = ap.parse_args()

    api = HfApi()
    private = not args.public
    model_repo = f"{args.user}/{args.model_name}"
    space_repo = f"{args.user}/{args.space_name}"

    model_dir = Path(tempfile.mkdtemp())
    build_model_folder(args.finetune, args.evaluation, model_dir)
    api.create_repo(model_repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=model_repo,
        repo_type="model",
        commit_message="Fine-tuned GENCO weights and unresolved-case warning",
    )

    space_dir = Path(tempfile.mkdtemp())
    shutil.copy(ROOT / "demo/index.html", space_dir / "index.html")
    shutil.copy(ROOT / "demo/README.md", space_dir / "README.md")
    api.create_repo(space_repo, repo_type="space", space_sdk="static", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(space_dir), repo_id=space_repo, repo_type="space", commit_message="Case explorer")

    print(f"model: https://huggingface.co/{model_repo}")
    print(f"space: https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()
