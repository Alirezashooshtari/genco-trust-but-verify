"""Creates the Hugging Face model repository and the demo Space (private unless --public).

    pip install huggingface_hub safetensors torch
    huggingface-cli login
    python scripts/publish_huggingface.py --user <account> \
        --finetune finetune_seed0.zip --evaluation evaluate_finetuned_seed0.zip \
        --loadgrowth finetune_loadgrowth_seed0.zip

--finetune and --evaluation are the zip files written by notebooks/finetune.ipynb and notebooks/evaluate_finetuned.ipynb
with TRAIN_SEED = 0, the published run. The weights are checked against the SHA-256 recorded in
results/finetune_seed0/summary.json and the evaluation against the same hash, then the weights are converted to
safetensors and uploaded together with the warning heads, the feature matrices, the model card and the training notebook.
--loadgrowth is the zip written by notebooks/finetune_loadgrowth.ipynb with TRAIN_SEED = 0; its weights are checked the same
way against results/finetune_loadgrowth_seed0/summary.json and published under loadgrowth/. --finetune-seed1 and
--loadgrowth-seed1 are the TRAIN_SEED = 1 runs of the two notebooks, checked against results/finetune_seed1 and
results/finetune_loadgrowth_seed1 and published under seed1/ and loadgrowth/seed1/.
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
RUN = "seed0"
LOADGROWTH_RUN = "loadgrowth_seed0"

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


def second_seed(zip_path, run, dst, tmp, out):
    """Checks a TRAIN_SEED = 1 checkpoint against results/<run>/summary.json and converts it to out/<dst>."""
    lsum = json.loads((ROOT / f"results/{run}/summary.json").read_text())
    weights = next(unzip(zip_path, tmp / run).rglob("genco_finetuned_state_dict.pt"))
    if sha256(weights) != lsum["checkpoint_sha256"]:
        raise SystemExit(f"{zip_path}: SHA-256 does not match results/{run}/summary.json")
    if lsum["frozen_protocol"]["train_seed"] != 1:
        raise SystemExit(f"{zip_path} is not a TRAIN_SEED = 1 run")
    (out / dst).parent.mkdir(parents=True, exist_ok=True)
    state = torch.load(weights, map_location="cpu")
    save_file({k: v.contiguous() for k, v in state.items()}, out / dst)
    return {
        "run": run,
        "train_seed": 1,
        "genco_finetuned_state_dict.pt": lsum["checkpoint_sha256"],
        dst: sha256(out / dst),
    }


def build_model_folder(
    finetune_zip, evaluation_zip, out, loadgrowth_zip=None, finetune_seed1=None, loadgrowth_seed1=None
):
    tmp = Path(tempfile.mkdtemp())
    ft = unzip(finetune_zip, tmp / "finetune")
    ev = unzip(evaluation_zip, tmp / "evaluation")
    summary = json.loads((ROOT / f"results/finetune_{RUN}/summary.json").read_text())

    weights = next(ft.rglob("genco_finetuned_state_dict.pt"))
    if sha256(weights) != summary["checkpoint_sha256"]:
        raise SystemExit(f"{weights.name}: SHA-256 does not match results/finetune_{RUN}/summary.json")
    evaluated = json.loads(next(ev.rglob("summary.json")).read_text())["checkpoint_sha256"]
    if evaluated != summary["checkpoint_sha256"]:
        raise SystemExit(f"{evaluation_zip} evaluates a different checkpoint")
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
    shutil.copy(ROOT / "notebooks/finetune.ipynb", out / "training/finetune.ipynb")

    loadgrowth = None
    if loadgrowth_zip:
        lsum = json.loads((ROOT / f"results/finetune_{LOADGROWTH_RUN}/summary.json").read_text())
        lweights = next(unzip(loadgrowth_zip, tmp / "loadgrowth").rglob("genco_finetuned_state_dict.pt"))
        if sha256(lweights) != lsum["checkpoint_sha256"]:
            raise SystemExit(f"{lweights.name}: SHA-256 does not match results/finetune_{LOADGROWTH_RUN}/summary.json")
        (out / "loadgrowth").mkdir()
        lstate = torch.load(lweights, map_location="cpu")
        save_file({k: v.contiguous() for k, v in lstate.items()}, out / "loadgrowth/model.safetensors")
        shutil.copy(ROOT / "notebooks/finetune_loadgrowth.ipynb", out / "training/finetune_loadgrowth.ipynb")
        loadgrowth = {
            "what": "fine-tuned as model.safetensors, plus a load-growth response term (notebooks/finetune_loadgrowth.ipynb)",
            "train_seed": lsum["frozen_protocol"]["train_seed"],
            "optimizer_steps": lsum["optimizer_steps"],
            "selected_step": lsum["selected_step"],
            "minutes_on_t4": round(lsum["training_minutes"]),
            "sha256": {
                "genco_finetuned_state_dict.pt": lsum["checkpoint_sha256"],
                "loadgrowth/model.safetensors": sha256(out / "loadgrowth/model.safetensors"),
            },
        }

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
            "train_seed": protocol["train_seed"],
            "minutes_on_t4": round(summary["training_minutes"]),
        },
        "sha256": {
            "genco_finetuned_state_dict.pt": summary["checkpoint_sha256"],
            "model.safetensors": sha256(out / "model.safetensors"),
        },
    }
    if loadgrowth:
        config["loadgrowth"] = loadgrowth
    seeds = []
    if finetune_seed1:
        seeds.append(second_seed(finetune_seed1, "finetune_seed1", "seed1/model.safetensors", tmp, out))
    if loadgrowth_seed1:
        seeds.append(
            second_seed(loadgrowth_seed1, "finetune_loadgrowth_seed1", "loadgrowth/seed1/model.safetensors", tmp, out)
        )
    if seeds:
        config["second_seeds"] = seeds
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if config["sha256"]["model.safetensors"] not in (ROOT / "release/MODEL_CARD.md").read_text():
        raise SystemExit("release/MODEL_CARD.md does not state the SHA-256 of the converted model.safetensors")
    if (
        loadgrowth
        and loadgrowth["sha256"]["loadgrowth/model.safetensors"] not in (ROOT / "release/MODEL_CARD.md").read_text()
    ):
        raise SystemExit("release/MODEL_CARD.md does not state the SHA-256 of loadgrowth/model.safetensors")
    for sd in seeds:
        dst = next(k for k in sd if k.endswith(".safetensors"))
        if sd[dst] not in (ROOT / "release/MODEL_CARD.md").read_text():
            raise SystemExit(f"release/MODEL_CARD.md does not state the SHA-256 of {dst}")
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
    ap.add_argument("--loadgrowth", help="zip written by notebooks/finetune_loadgrowth.ipynb (optional)")
    ap.add_argument("--finetune-seed1", help="zip written by notebooks/finetune.ipynb with TRAIN_SEED = 1 (optional)")
    ap.add_argument(
        "--loadgrowth-seed1", help="zip written by notebooks/finetune_loadgrowth.ipynb with TRAIN_SEED = 1 (optional)"
    )
    ap.add_argument("--model-name", default="genco-texas-clustered")
    ap.add_argument("--space-name", default="genco-contingency-verifier")
    ap.add_argument("--public", action="store_true", help="create public repositories (default: private)")
    args = ap.parse_args()

    api = HfApi()
    private = not args.public
    model_repo = f"{args.user}/{args.model_name}"
    space_repo = f"{args.user}/{args.space_name}"

    model_dir = Path(tempfile.mkdtemp())
    build_model_folder(
        args.finetune, args.evaluation, model_dir, args.loadgrowth, args.finetune_seed1, args.loadgrowth_seed1
    )
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
