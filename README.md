<div align="center">
  <h1> GramSR: Visual Feature Conditioning for
Diffusion-Based Super-Resolution </h1>
</div>

<div align="center">
  
[![Paper](https://img.shields.io/badge/Paper-arxiv.2511.22715-B31B1B.svg)](https://arxiv.org/abs/2511.22715)
[![HF Collection](https://img.shields.io/badge/🤗-HF%20Collection-yellow.svg)]()
</div>

<p align="center">
  <img src="./assets/model.png" alt="GramSR" width="840" />
</p>

## 📢 Latest Updates
  - [2026/05/11] 🤗 Inference code and **weights** release

## Overview

GramSR is a
## Table of Contents

- [Environment Setup](#environment-setup)
  - [Inference environment](#1-create-the-inference-environment-venv)
  - [EVQA evaluation environment](#2-create-the-evqa-evaluation-environment-evqa-eval)
- [Inference](#inference)
  - [Slurm (optional)](#slurm-optional)
  - [What the scripts do](#what-the-scripts-do)
- [Outputs](#outputs)
- [Citation](#citation)


## Environment Setup

### 1) Create the inference environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

If you use `conda`, you can install the packages from [requirements.txt](requirements.txt) in an activated conda environment.

## Inference

Once datasets and indexes are in place, unzip all archives and update the paths in the `.sh` scripts to match your local filesystem. We provide two ready-to-use scripts:

- EVQA: [retrieval_evqa.sh](src/retrieval_module/retrieval_evqa.sh)
- Infoseek: [retrieval_infoseek.sh](src/retrieval_module/retrieval_infoseek.sh)

These scripts are written as Slurm jobs. For local runs, remove the Slurm directives and `srun` prefix while keeping the rest of the command unchanged.

### Slurm (optional)

If you are on an HPC cluster with SLURM, submit the scripts directly after editing the asset paths:

```bash
sbatch retrieval_evqa.sh
sbatch retrieval_infoseek.sh
```

### What the scripts do

Both scripts invoke [retrieval.py](src/retrieval_module/retrieval.py) with a common set of flags representing the **full end-to-end ReAG pipeline**:

- `--model_name "aimagelab/ReAG-3B"`
- `--top_k 20`
- `--force_reasoning` + `--extract_reasoning`
- `--crop_query_img`
- `--critic_model_name "aimagelab/ReAG-Critic"`
- ` --eval_passages` + `--yes_prob_thr 0.1`

You can freely add or remove flags inside the `.sh` files to configure your experiment. Notable options include:

- `--few_shots` — Infoseek few-shot prompting
- `--eval_passages` — critic-based passage filtering
- `--use_google_lens` or `--use_oracle` — EVQA retrieval variants

## Outputs

Results are written under the `--output_root` directory. The script automatically constructs an experiment folder name based on the dataset, model, retrieval setup, and active flags.

## Citation

If you use this code, please cite our CVPR 2026 paper:

```bibtex
@inproceedings{compagnoni2026reag,
  title={{ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering}},
  author={Compagnoni, Alberto and Morini, Marco and Sarto, Sara and Cocchi, Federico and Caffagni, Davide and Cornia, Marcella and Baraldi, Lorenzo and Cucchiara, Rita},
  booktitle={Proceedings of the IEEE/CVF Computer Vision and Pattern Recognition Conference},
  year={2026}
}
```
