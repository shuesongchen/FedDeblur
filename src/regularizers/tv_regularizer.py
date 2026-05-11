"""
TV (Total Variation) regularizer for image deblurring.

This module implements the TV regularization method using the soft thresholding
function. The implementation is precisely extracted from the original batch
processing scripts to ensure numerical equivalence.
"""

import torch
from .base_regularizer import BaseRegularizer
from ..utils.math_ops import Dx, Dy, shrink_soft_2
from typing import Tuple


class TVRegularizer(BaseRegularizer):
    def __init__(self, params: dict, device):
        super().__init__(params, device)

        required_params = ["eta", "rho"]
        for param in required_params:
            if param not in params:
                raise ValueError(f"TV regularizer requires '{param}' parameter")

        self.eta = params["eta"]
        self.rho = params["rho"]

    def step(
        self, x: torch.Tensor, mu_x: torch.Tensor, mu_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_x = Dx(x)
        grad_y = Dy(x)

        z_x_temp = grad_x - mu_x / self.rho
        z_y_temp = grad_y - mu_y / self.rho

        z_x, z_y = shrink_soft_2(z_x_temp, z_y_temp, self.eta / self.rho)

        return z_x, z_y

    def validate_parameters(self):
        super().validate_parameters()

        if self.eta <= 0:
            raise ValueError(f"'eta' must be positive, got {self.eta}")

        if self.rho <= 0:
            raise ValueError(f"'rho' must be positive, got {self.rho}")

    def get_regularization_strength(self) -> float:
        return self.eta / self.rho
