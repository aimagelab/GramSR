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
