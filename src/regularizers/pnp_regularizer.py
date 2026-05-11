import torch
from .base_regularizer import BaseRegularizer
from ..utils.pnp_denoiser import DRUNetDenoiser


class PnPRegularizer(BaseRegularizer):
    def __init__(self, params: dict, device):
        super().__init__(params, device)

        required_params = ["sigma", "rho"]
        for param in required_params:
            if param not in params:
                raise ValueError(f"PnP regularizer requires '{param}' parameter")

        self.sigma = params["sigma"]
        self.rho = params["rho"]

        self.denoiser = DRUNetDenoiser(
            model_type="gray", weights_dir="model_zoo", device=self.device
        )

    def step(self, consensus_x: torch.Tensor) -> torch.Tensor:
        if consensus_x.dim() == 3:
            raise ValueError("Regularization only supports grayscale images. ")
        elif consensus_x.dim() != 2:
            raise ValueError(
                f"expects 2D grayscale image tensor, got {consensus_x.dim()}D tensor with shape {consensus_x.shape}"
            )

        denoised = self.denoiser.denoise(consensus_x, self.sigma)

        return denoised

    def validate_image_compatibility(self, image_shape: tuple):
        if len(image_shape) == 3:
            raise ValueError("Invalid image shape")
        elif len(image_shape) != 2:
            raise ValueError("Invalid image shape")

    def validate_parameters(self):
        super().validate_parameters()

        # Check parameter ranges
        if self.sigma <= 0:
            raise ValueError(f"'sigma' must be positive, got {self.sigma}")

        if self.rho <= 0:
            raise ValueError(f"'rho' must be positive, got {self.rho}")

        if self.sigma > 255:
            raise ValueError(f"'sigma' should be in [0, 255] range, got {self.sigma}")

    def get_denoising_strength(self) -> float:
        return self.sigma

    def is_color_supported(self) -> bool:
        return False
