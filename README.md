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
- [Inference](#inference)
- [Outputs](#outputs)
- [Citation](#citation)


## Environment Setup

### 1) Create the environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

If you use `conda`, you can install the packages from [requirements.txt](requirements.txt) in an activated conda environment.

### 2) Models & weights



**Models & weights**
- Stable Diffusion 2.1 weights (required): download the Stable Diffusion 2.1 model weights and provide their folder path as `--pretrained_model_path` when running the scripts.
- The DINO embedder in the code defaults to the HF model id `facebook/dinov3-vitb16-pretrain-lvd1689m` and will be downloaded automatically if not present in the local cache.
- LoRA adapters and other checkpoints are read from the  `weights/checkpoints` directory in the repo; filenames in use by the code follow the pattern used in `weights/checkpoints` (e.g. `adapter_53001.pth`, `model_12501.pkl`).

The files `adapter_53001.pth` and `lora_weights_dino_53001.safetensors` are already included in the repository.
However, `model_12501.pkl` is not hosted on GitHub because it exceeds the file size limit (25 MB). It must be downloaded separately from the provided external link and placed into `weights/checkpoints/`.

Save any checkpoints you download from the provided link into `weights/checkpoints/` so the code can find them.



## Inference

```bash
python inference.py \
  --pretrained_model_path /path/to/sd2.1_model \
  --lora_dir ./checkpoints \
  --output_dir output/pred
```

## Outputs

Results are written under the `--output_root` directory. The script automatically constructs an experiment folder name based on the dataset, model, retrieval setup, and active flags.

## Citation

If you use this code, please cite our ICPR 2026 paper:

```bibtex
@inproceedings{fdoronzio2026gramsr,
  title={{GramSR: Visual Feature Conditioning for Diffusion-Based Super-Resolution}},
  author={D'Oronzio, Fabio and Putamorsi, Federico and Zini, Leonardo and Cornia, Marcella and Baraldi, Lorenzo},
  booktitle={International Conference on Pattern Recognition},
  year={2026}
}
```
