# Running the notebooks

Every notebook runs top to bottom on a fresh Google Colab runtime. It installs what it needs, downloads the public data
and the released checkpoint from Hugging Face (`gridfm/genco-pf-contingency`), and ends by zipping its output folder
(for example `storms.zip`). The folders in `results/` hold the parts of those outputs that the numbers and figures
need; large files (checkpoints, feature matrices) are left out, and the fine-tuned model's are on Hugging Face.

1. Open the notebook in Colab. For a notebook that needs a GPU (see `notebooks/README.md`) choose *Runtime → Change
   runtime type → T4 GPU*.
2. *Runtime → Run all*. The first cells print the download plan and the expected run time.
3. Download the zip it produces and compare it with the matching folder in `results/`.

`finetune` and `evaluate_finetuned` take a training seed (`TRAIN_SEED` and `FINETUNE_SEED` in their first cell; the
published model is seed 0). `evaluate_finetuned` needs `finetune_seed<N>.zip` from `finetune`: upload it into the Colab
file pane, or leave it in `MyDrive/genco_runs/`. Both notebooks mount Google Drive and copy their zip there as well, so a
long run is not lost if the runtime disconnects.

## Pinned versions

| Component | Version |
|---|---|
| gridfm-graphkit | branch `genco-paper-repro`, commit `95b90eecc7f5c2c86a9dd776f0c3f40434def22f` |
| Hugging Face `gridfm/genco-pf-contingency` | revision `5c625b00f5b79c8a7d4d27ecc133d4ecdf573af7` (Apache-2.0) |
| Released checkpoint `model/training_bs_16` | SHA-256 `86124547e21dd380d654495a073c493a11806651d4556036c1f62e8f5b4ceccf` |
| Fine-tuned checkpoint, published (seed 0) | SHA-256 `cf705d23b87cc0fd1b6ede6a9028925584bd9b00b936a76244a90a75fcb353a2` |
| Fine-tuned checkpoints, seed 1 and first run | SHA-256 `544185154b000f39bbfa53cf7ed1ac6e92fdebf47092f4b864973e418c38d14b`, `7e9e3c635ea6eca6d3b1dc7009c2ba9e2aa05890fbd9a6f76ab59084d75c57e6` |
| Colab runtime (October 2026) | Python 3.13, torch 2.11.0+cu130, numpy 2.1.3, pandas 2.2.3, lightsim2grid 1.1.0; full list in `colab_pip_freeze.txt` |

`data_sha256.csv` and `checkpoint_sha256.csv` list the server-side SHA-256 of every file of the Hugging Face repository.

Every notebook checks out this commit and downloads this revision. The Texas notebooks first ran against the head of the
branch and the latest revision; these were the same versions throughout (the commit was the branch head from 1 to
14 October 2026, the revision the latest from 1 October), and the runs that record them show this commit. The pins were
added afterwards so that a new run uses the same versions. `finetune_as_run.ipynb` is kept unpinned, as it ran.
`second_grid.ipynb` pins its own model and data revisions.

## Installing gridfm-graphkit on Colab

At the pinned commit gridfm-graphkit declares `python <3.13` and `torch <2.9`, which Colab no longer provides, so `pip`
refuses to install it. The notebooks clone it instead and import it from the clone:

```bash
git clone -b genco-paper-repro https://github.com/gridfm/gridfm-graphkit.git
git -C gridfm-graphkit checkout 95b90eecc7f5c2c86a9dd776f0c3f40434def22f
pip install torch_geometric lightning huggingface_hub pyarrow scikit-learn lightsim2grid
```

`torch_scatter` has no wheel for torch 2.11; the notebooks use a small pure-PyTorch replacement, which reproduces the
released predictions to 1.8e-6 p.u.

## What reproduces exactly

Counts, statuses and decisions reproduce exactly. GPU arithmetic is not bit-deterministic, so GENCO's predicted voltages can
differ in the last digits between runs; a second full run of `evaluate_finetuned` gave identical
verdicts and every endpoint within 1e-6.
Timings depend on the machine.

## Module versions

Each notebook embeds the version of the verification module it ran with. `src/genco_verify` is the same code organised as a
package; on 1,200 randomised cases covering every policy and option it returns results identical to the embedded versions.
`finetune.ipynb` produced the published checkpoint (seed 0) and the second seed. `finetune_as_run.ipynb` is the code of
the first run, before the gradient-accumulation fix; its results are kept in `results/finetune_as_run` and
`results/evaluate_finetuned_as_run` as a third run.
