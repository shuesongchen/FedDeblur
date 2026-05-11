"""
Regularizers module for image deblurring algorithms.

This module provides regularization methods including TV (Total Variation)
and PnP (Plug-and-Play) for image restoration tasks.

Available regularizers:
- TVRegularizer: Total Variation regularization using soft thresholding
- PnPRegularizer: Plug-and-Play regularization using DR-UNet (grayscale only)
"""

from .base_regularizer import BaseRegularizer
from .tv_regularizer import TVRegularizer
from .pnp_regularizer import PnPRegularizer

__all__ = [
    'BaseRegularizer',
    'TVRegularizer',
    'PnPRegularizer'
]


def create_regularizer(regularizer_type: str, params: dict, device):
    """
    Factory function to create regularizer instances.

    Args:
        regularizer_type: Type of regularizer ("TV" or "PnP")
        params: Parameter dictionary for the regularizer
        device: PyTorch device

    Returns:
        Regularizer instance

    Raises:
        ValueError: For unsupported regularizer types
    """
    if regularizer_type == "TV":
        return TVRegularizer(params, device)
    elif regularizer_type == "PnP":
        return PnPRegularizer(params, device)
    else:
        raise ValueError(
            f"Unsupported regularizer type: {regularizer_type}. "
            f"Supported types: 'TV', 'PnP'"
        )