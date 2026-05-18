# `FedDeblur`: Federated Multi-Observation Image Deblurring via Consensus ADMM with Partial Overlapping Views

## Quick Start

```bash
conda create -n feddeblur python=3.13 -y
conda activate feddeblur
pip install torch torchvision torchmetrics numpy
```

The PnP regularizer requires the DRUNet checkpoint at `model_zoo/`. This checkpoint is tracked with Git LFS. If it is missing after cloning, run `git lfs pull`.

### Example usage

FedDeblur with TV regularization under complete observation:

```bash
python main.py --algorithm FedDeblur --regularizer TV --observation complete --n_clients 5 --noise_std 1.0
```

FedDeblur with TV regularization under partial observation:

```bash
python main.py --algorithm FedDeblur --regularizer TV --observation partial --blur_type motion --noise_std 1.0
```

FedDeblur with PnP regularization under complete observation:

```bash
python main.py --algorithm FedDeblur --regularizer PnP --observation complete --n_clients 5 --noise_std 1.0
```

FedDeblur with PnP regularization under partial observation:

```bash
python main.py --algorithm FedDeblur --regularizer PnP --observation partial --blur_type synthesis --noise_std 1.0
```
