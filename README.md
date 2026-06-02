# SeDID

Code for the paper "Exposing the Fake: Effective Diffusion-Generated Images Detection" (AdvML Frontiers Workshop at ICML 2023).

This repository contains:

- `main.py`: training and evaluation entrypoint for DDPM backbones used by the paper
- `mia_evals/SeDID.py`: statistical SeDID detector
- `mia_evals/SeDID_nn.py`: neural-network post-processor utilities for SeDID features

## Environment

The original project was written around Python 3.9 and PyTorch 2.0, but the maintenance fixes in this repository also work with newer PyTorch environments.

Install the basic dependencies with:

```bash
pip install -r requirements.txt
```

If `tensorboardX` is not installed, `main.py` will fall back to `torch.utils.tensorboard`.

## Dataset Layout

By default, the project looks for datasets under:

```text
./datasets
```

Examples:

- `./datasets/cifar10`
- `./datasets/cifar100`
- `./datasets/celeba`
- `./datasets/svhn`
- `./datasets/stl10`
- `./datasets/tiny-imagenet-200/train`

Member split files are stored in:

```text
./mia_evals/member_splits
```

You can override paths with environment variables when needed:

```bash
export SEDID_DATASET_ROOT=/path/to/datasets
export SEDID_MEMBER_SPLIT_ROOT=/path/to/member_splits
```

For synthetic non-member samples used by `mia_evals/dataset_utils.py`, you can also override:

```bash
export SEDID_CIFAR10_SYNTHETIC_CKPT=/path/to/ckpt-step1200000_samples.pt
export SEDID_CIFAR10_SYNTHETIC_DIR=/path/to/generated_images/cifar10
export SEDID_TINYIN_SYNTHETIC_CKPT=/path/to/ckpt-step1200000_samples.pt
export SEDID_TINYIN_SYNTHETIC_DIR=/path/to/generated_images/TINY-IN
```

## Training

Train a DDPM model:

```bash
python main.py --train --dataset CIFAR10 --logdir ./logs/DDPM_CIFAR10_EPS
```

The helper script `train.sh` shows one minimal example.

## Evaluation

Evaluate a saved checkpoint:

```bash
python main.py --eval --dataset CIFAR10 --img_size 32 --logdir ./logs/DDPM_CIFAR10_EPS
```

`main.py` currently expects the checkpoint file name `ckpt-step800000.pt` inside `--logdir`, matching the original project convention.

## Running SeDID

The standalone statistical script requires the checkpoint path and training flagfile path:

```bash
export SEDID_CKPT=/path/to/ckpt-step800000.pt
export SEDID_FLAGFILE=/path/to/flagfile.txt
python -m mia_evals.SeDID
```

The neural post-processor utilities can be imported from:

```python
from mia_evals.SeDID_nn import run_NNs
```

## Maintenance Notes

This repository has been lightly maintained to improve reproducibility:

- fixed package import paths so module entrypoints run more reliably
- removed several machine-specific default paths from the main workflow
- fixed a threshold-range bug in `mia_evals/SeDID.py`
- added a minimal dependency file and ignored Python cache artifacts

The paper-specific experiment scripts under `mia_evals/secmia_my_details*.py` still contain historical hard-coded experiment paths and are best treated as archival scripts unless you adapt them to your environment.
