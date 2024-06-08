# README for Exposing the Fake: Effective Diffusion-Generated Images Detection

This repository contains the code associated with the research paper "Exposing the Fake: Effective Diffusion-Generated Images Detection".

## Requirements

- Python version 3.9.1
- PyTorch version 2.0.0

Ensure that you are running these versions of Python and PyTorch. If you need to install PyTorch, please follow the official installation instructions at https://pytorch.org.

## Files Description

The code is organized as follows:

- `mia_evals\main.py`: The main script to train the diffusion model. The main tasks are defined here, including training and evaluation. Use the `--train` flag for training and the `--eval` flag for evaluation.

- `mia_evals\SeDID_my_details.py`: The script to launch the Stepwise Error for Diffusion-generated Image Detection (SeDID) detection method. 

- The `mia_evals` directory contains the following additional scripts:
  - `attack_naive_rand.py`: Implements the naive random attack method.
  - `attack_sum.py`: Implements the summation attack method.
  - `components.py`: Contains various components used across the project.
  - `count.py`: Implements the count method.
  - `dataset_utils.py`: Contains utilities for handling datasets.
  - `diffusion.py`: Contains the implementation of the diffusion process.
  - `load_plk.py`: Utilities for loading pickle files.
  - `load.py`: Contains various loading utilities.
  - `model.py`: Contains the model architecture and related functions.
  - `resnet.py`: Contains the ResNet model implementation.
  - `SeDID_my_details.py`: The main script for the SeDID method.
  - `SeDID_nn.py`: The script for the SeDID method with neural networks.
  - `SeDID.py`: The main script for the SeDID method.

## Usage

To train the diffusion model, run:

```python
python mia_evals\main.py --train
```

To evaluate the model, run:

```python
python mia_evals\main.py --eval
```

To launch the SeDID detection method, run:

```python
python mia_evals\SeDID_my_details.py
```

Note: Make sure you are in the right Python environment and have all the necessary dependencies installed.
