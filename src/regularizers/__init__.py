"""
Regularizers module for image deblurring algorithms.

Available regularizers:
- TVRegularizer: Total Variation regularization using soft thresholding
- PnPRegularizer: Plug-and-Play regularization using DR-UNet
"""

from .base_regularizer import BaseRegularizer
from .tv_regularizer import TVRegularizer
from .pnp_regularizer import PnPRegularizer

__all__ = ["BaseRegularizer", "TVRegularizer", "PnPRegularizer"]


def create_regularizer(regularizer_type: str, params: dict, device):
    if regularizer_type == "TV":
        return TVRegularizer(params, device)
    elif regularizer_type == "PnP":
        return PnPRegularizer(params, device)
    else:
        raise ValueError(f"Unsupported regularizer type: {regularizer_type}.")
