# FOMO26 - Team volBrain

[![CI](https://github.com/EloiNavet/FOMO26/actions/workflows/ci.yaml/badge.svg)](https://github.com/EloiNavet/FOMO26/actions/workflows/ci.yaml)
[![Pipeline tests](https://github.com/EloiNavet/FOMO26/actions/workflows/pipeline_tests.yml/badge.svg)](https://github.com/EloiNavet/FOMO26/actions/workflows/pipeline_tests.yml)
[![Python 3.11-3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Official source-code release of **Team volBrain's submission to the FOMO26 challenge**.

Our submitted model uses a 3D **ResEnc-B** backbone pretrained on heterogeneous brain MRI with an **AMAES masked-reconstruction objective**. The same pretrained parent is then fine-tuned for Tasks 1-5 and used as a frozen representation extractor for Tasks 6-7.

### Submitted configuration

| Component | Released contract |
|---|---|
| Backbone | `resenc_unet_b_ssl` (ResEnc-B) |
| Objective | AMAES masked reconstruction only |
| Reconstruction loss | masked MSE, weight `10.0` |
| Input patch | `160 × 160 × 160` voxels |
| Mask ratio | `0.6` |
| Pretraining corpus | `FOMO300K_curated_iso1p0_v1` |
| Curated input count | 33,336 preprocessed MRI tensors |
| Geometry | 1.0 mm isotropic, RAS orientation |
| Global batch size | 32 across four devices |
| Scheduler horizon | 187,500 optimizer steps / 6 million samples |
| Submitted milestone | 96,000 optimizer steps |
| Authoritative recipe | [`resenc_amaes_p2_96k.yaml`](configs/projects/fomo26/safety/pretrain/resenc_amaes_p2_96k.yaml) |

The 187,500-step value is the scheduler horizon. The released model is the explicitly stopped **96,000-step milestone**; the two values are not interchangeable.

## Challenge tasks

| Task | Problem | Input modalities | Output | Primary metric |
|---|---|---|---|---|
| 1 | Infarct-presence classification | FLAIR, ADC, DWI b1000, T2\*/SWI | probability in `.txt` | AUROC |
| 2 | Meningioma segmentation | FLAIR, DWI b1000, T2\*/SWI | binary `.nii.gz` mask | DSC |
| 3 | Brain-age regression | T1 MRI | age estimate in `.txt` | MAE |
| 4 | Nerve/vessel segmentation | T2 MRI | multiclass `.nii.gz` mask | DSC |
| 5 | Polymicrogyria classification | T1 MRI | probability in `.txt` | AUROC |
| 6-7 | Hidden downstream probing | one 3D MRI volume | frozen 1-D `.npy` embedding | organizer evaluation |

Tasks 1-5 use task-specific fine-tuning and inference entrypoints. Tasks 6-7 consume the **frozen pretrained representation**: no Task 1-5 fine-tuned checkpoint is admissible. The machine-readable task contract is [`task_definitions.json`](finetuning/fomo26_inference/task_definitions.json).

## What is included

- the ResEnc-B/AMAES implementation and model configuration;
- the preprocessing geometry contract used to build the curated corpus;
- fine-tuning configurations for Tasks 1-5;
- task-specific inference entrypoints for Tasks 1-7;
- checkpoint loading, embedding extraction, geometry, TTA, and runtime contracts;
- container build and validation tooling;
- CPU, DDP, release, and pipeline test coverage;
- environment, citation, and licensing records.

## What is not included

- MRI data, labels, participant tables, or dataset manifests;
- trained pretraining or downstream checkpoints;
- submitted Apptainer/Singularity images;
- private experiment tracking or cluster orchestration;
- historical candidate-selection machinery;
- development architectures and objectives outside the released method;
- third-party validator source code.

The repository can be installed, inspected, composed, and tested without private data. Full training, fine-tuning, and prediction require the corresponding prepared datasets and checkpoints.

## Installation

### Requirements

- Linux;
- Python 3.11 or 3.12;
- [`uv==0.12.7`](https://docs.astral.sh/uv/);
- CUDA only for GPU training or inference.

Clone the repository and create the locked environment:

```bash
git clone https://github.com/EloiNavet/FOMO26.git
cd FOMO26

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "uv==0.12.7"

uv sync --locked
```

For development and all test dependencies:

```bash
uv sync --locked --all-extras --all-groups
```

The public runtime pins PyTorch 2.7.0, Lightning 2.4.0, Torchvision 0.22.0, and `gardening_tools` 0.3.2. See [`docs/ENVIRONMENTS.md`](docs/ENVIRONMENTS.md) for the distinction between the public runtime and the recorded historical SIF environment.

## Environment variables

Create a `.env` file at the repository root or export the variables in your shell:

```dotenv
ASPARAGUS_CONFIGS=/absolute/path/to/FOMO26/configs
ASPARAGUS_DATA=/absolute/path/to/prepared/data
ASPARAGUS_MODELS=/absolute/path/to/checkpoints
ASPARAGUS_RESULTS=/absolute/path/to/results
ASPARAGUS_RAW_LABELS=/absolute/path/to/raw/labels
```

The entrypoints load `.env` automatically. Use absolute paths and never commit operator-specific paths, credentials, or personal data.

For offline logging, you may additionally set:

```dotenv
WANDB_MODE=offline
WANDB_ENTITY=test
```

## Walkthrough

### 1. Inspect the submitted configuration

Hydra can print the fully composed configuration without starting training:

```bash
uv run asp_pretrain \
  --config-name projects/fomo26/safety/pretrain/resenc_amaes_p2_96k \
  --cfg job
```

The resolved recipe must select ResEnc-B, AMAES reconstruction only, the curated dataset contract, and the 96,000-step stop milestone.

### 2. Preprocess the MRI corpus

The P0 preprocessing contract canonicalizes the selected images to 1 mm isotropic RAS geometry while preserving the full field of view. Its implementation is [`p0_kernel.py`](asparagus/preprocessing/asparagus_preprocessing/p0_kernel.py).

Reconstructing the training corpus requires access to the source MRI collections and the release-pinned manifests. They are not distributed here. The pretraining launcher validates the expected dataset and tensor-manifest digests before training, so an alternate corpus cannot silently replace the released one.

### 3. Launch model pretraining

Once the prepared corpus, manifest contract, and run-provenance variables are available:

```bash
uv run asp_pretrain \
  --config-name projects/fomo26/safety/pretrain/resenc_amaes_p2_96k
```

This command launches the public model configuration. It is not, by itself, a reproduction of the historical run: that additionally requires the original data, four-device execution environment, complete schedule, and run metadata.

### 4. Fine-tune Tasks 1-5

Point every task to the same compatible checkpoint:

```bash
export FOMO26_RESENC_CHECKPOINT=/absolute/path/to/checkpoint.ckpt
```

Then select the entrypoint matching the downstream problem:

```bash
# Task 1 - infarct-presence classification
uv run asp_finetune_cls \
  --config-name projects/fomo26/finetune/task1_presence

# Task 2 - meningioma segmentation
uv run asp_finetune_seg \
  --config-name projects/fomo26/finetune/task2_lesion

# Task 3 - brain-age regression
uv run asp_finetune_reg \
  --config-name projects/fomo26/finetune/task3_age

# Task 4 - nerve/vessel segmentation
uv run asp_finetune_seg \
  --config-name projects/fomo26/finetune/task4_multiclass

# Task 5 - polymicrogyria classification
uv run asp_finetune_cls \
  --config-name projects/fomo26/finetune/task5_ppmr
```

These are the ResEnc-B configurations. Similarly named `_ft` configurations represent separate retained experiments and are not the model walkthrough.

### 5. Run task-specific inference

Each task has a dedicated fail-closed entrypoint. Inspect the exact interface with:

```bash
uv run python -m finetuning.container.predict_task1 --help
uv run python -m finetuning.container.predict_task2 --help
uv run python -m finetuning.container.predict_task3 --help
uv run python -m finetuning.container.predict_task4 --help
uv run python -m finetuning.container.predict_task5 --help
uv run python -m finetuning.container.predict_task6_7 --help
```

Example for Task 1:

```bash
uv run python -m finetuning.container.predict_task1 \
  --flair /path/to/flair.nii.gz \
  --adc /path/to/adc.nii.gz \
  --dwi /path/to/dwi_b1000.nii.gz \
  --swi /path/to/swi.nii.gz \
  --manifest /path/to/task1_manifest.json \
  --checkpoint-name best \
  --output /path/to/prediction.txt
```

Pass either `--t2s` or `--swi` for the susceptibility image. The manifest defines the eligible checkpoint members and is validated before inference.

### 6. Extract frozen embeddings for Tasks 6-7

```bash
uv run python -m finetuning.container.predict_task6_7 \
  --input /path/to/volume.nii.gz \
  --output /path/to/embedding.npy \
  --checkpoint /path/to/checkpoint.ckpt \
  --architecture resenc_b \
  --ssl-objective amaes
```

The output is a one-dimensional floating-point NumPy array extracted from the frozen pretrained representation.

## Testing

Install the development dependencies and ensure the environment variables above point to existing directories. The default tier is CPU-only and does not require NATTEN or a GPU.

```bash
# Default CPU-fast suite
uv run python tests/run_test_tier.py fast

# Container and release contracts
uv run python tests/run_test_tier.py release

# Two-rank CPU/Gloo distributed contract
uv run python tests/run_test_tier.py ddp
```

The tier manifest in [`tests/test_tier_contract.py`](tests/test_tier_contract.py) assigns every tracked test module exactly once and fails when a test is silently omitted.

Focused checks for the released method:

```bash
# Model configuration and AMAES-only objective
uv run pytest tests/test_safety_resenc_amaes.py -v

# Portable numerical regression of the AMAES optimization step
uv run pytest tests/test_amaes_step_numerics.py -v

# Model-configuration and factory contracts
uv run pytest tests/test_model_config_contract.py -v

# Public packaging and repository surface
uv run pytest tests/test_publication_surfaces.py -v
```

## Container and external validator

The repository contains task-specific container entrypoints and a fail-closed build contract. The build tooling checks that runtime files are staged, rejects operator paths and credentials, and prevents the container runscript from accessing the network.

The official FOMO26 container validator is **not redistributed** because its upstream repository does not declare a licence. When explicitly requested by the release tooling, it is acquired only at the pinned commit and verified against [`validator_manifest.json`](finetuning/container/validator_manifest.json) before use. Its absence or a digest mismatch is an error, never a passing validation.

No container image is distributed here. The submitted environment inputs are recorded in [`historical_sif_profile.json`](finetuning/container/historical_sif_profile.json), but no binary-identical SIF rebuild is claimed.

## Repository layout

```text
asparagus/                         core models, training, metrics, and preprocessing
configs/projects/fomo26/           released pretraining and fine-tuning recipes
finetuning/fomo26_inference/       task definitions and inference policies
finetuning/container/              task entrypoints and container contracts
tests/                             numerical, runtime, DDP, and release tests
docs/                              environment and supporting documentation
CITATION.cff                       citation metadata
NOTICE                             third-party and redistribution notices
```

## Team

Authors are listed in the official challenge-submission order.

| | Author | Affiliation |
|---|---|---|
| 1 | Éloi Navet | Univ. Bordeaux, CNRS, Bordeaux INP, LaBRI, UMR 5800 |
| 2 | Rémi Giraud | Univ. Bordeaux, CNRS, Bordeaux INP, IMS, UMR 5218 |
| 3 | Boris Mansencal | Univ. Bordeaux, CNRS, Bordeaux INP, LaBRI, UMR 5800 |
| 4 | Pierrick Coupé | Univ. Bordeaux, CNRS, Bordeaux INP, LaBRI, UMR 5800 |

## Citation

If you use this software, please cite the repository using [`CITATION.cff`](CITATION.cff) or GitHub's **Cite this repository** menu.

## Upstream framework

This project is built on [Asparagus](https://github.com/Sllambias/asparagus), originally developed by Sebastian Llambias using PyTorch, Lightning, and Hydra.

The `asparagus` Python namespace and `asp_*` command-line entrypoints are retained for configuration and checkpoint compatibility. The FOMO26 method, configurations, task contracts, inference rail, and release validation published here were developed and maintained by Team volBrain.

## Licence

This repository is released under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for upstream attribution, dependency information, and redistribution notices.
