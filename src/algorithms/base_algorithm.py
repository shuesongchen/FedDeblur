"""
Base algorithm class.

FedDeblur, CenDeblur, FedAvgDeblur, and LocDeblur.
"""

from abc import ABC, abstractmethod
import torch
from typing import Dict, Tuple, List, Union
from ..regularizers import TVRegularizer, PnPRegularizer
from ..core.metrics import calculate_image_metrics, format_metrics_for_display
from ..utils.math_ops import setup_all, Dx, Dy, Dxt, Dyt, psf2otf


class BaseAlgorithm(ABC):
    """
    Abstract base class for image deblurring algorithms.

    This class provides the common ADMM framework and manages the interaction
    between different algorithms and regularizers. Each concrete algorithm
    implements its specific ADMM iteration logic.
    """

    def __init__(
        self, params: Dict, regularizer: Union[TVRegularizer, PnPRegularizer], device
    ):
        """
        Initialize base algorithm.

        Args:
            params: Algorithm parameters dictionary
            regularizer: Regularizer instance (TV or PnP)
            device: PyTorch device for computations
        """
        self.params = params
        self.regularizer = regularizer
        self.device = device

        # Extract common parameters
        self.max_iter = params.get("max_iter", 500)
        self.tol = params.get("tol", 1e-5)
        self.rho = params["rho"]
        self.gamma = 1.05

    def run(self, data: Dict) -> Tuple[torch.Tensor, int, Dict]:
        """
        Run the deblurring algorithm.

        Args:
            data: Dictionary containing data

        Returns:
            Tuple
        """
        # Validate image type
        self._validate_pnp_compatibility(data)

        # Initialize algorithm variables
        x, aux_vars, otfs = self._initialize(data)

        # Main ADMM iteration
        x_old = x.clone()
        iteration = 0

        for iteration in range(self.max_iter):
            # Perform one ADMM iteration
            x, aux_vars = self._admm_iteration(x, aux_vars, otfs, data)

            # Check convergence
            crit = torch.norm(x - x_old, "fro") / torch.norm(x, "fro")
            if crit < self.tol:
                break

            x_old = x.clone()

        # Clamp to valid range
        x_final = torch.clamp(x, 0, 255)

        # Calculate metrics if reference available
        metrics = {}
        if "clean_image" in data and data["clean_image"] is not None:
            metrics = calculate_image_metrics(x_final, data["clean_image"])

        return x_final, iteration + 1, metrics

    def _validate_pnp_compatibility(self, data: Dict):
        """
        Validate PnP regularizer compatibility with image data.

        Args:
            data: Data dictionary

        Raises:
            ValueError: When PnP is used with incompatible data
        """
        if isinstance(self.regularizer, PnPRegularizer):
            # Get sample image to check dimensionality
            if "blurred_images" in data:
                sample_image = data["blurred_images"][0]
            elif "observations" in data:
                observations = data["observations"]
                if isinstance(observations, list):
                    sample_image = observations[0]
                else:
                    sample_image = observations
            else:
                raise ValueError("No image data found in input")

            # Validate image compatibility
            self.regularizer.validate_image_compatibility(sample_image.shape)

    @abstractmethod
    def _initialize(self, data: Dict) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Initialize algorithm-specific variables.
        """
        pass

    @abstractmethod
    def _admm_iteration(
        self, x: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Perform one ADMM iteration.
        """
        pass

    def _compute_otfs(
        self, kernels: List[torch.Tensor], img_shape: Tuple
    ) -> List[torch.Tensor]:
        """
        Compute optical transfer functions for blur kernels.

        Args:
            kernels: List of blur kernel tensors
            img_shape: Image shape tuple

        Returns:
            List of OTF tensors
        """
        otf_kernels = []
        original_is_color = len(img_shape) == 3 and img_shape[2] == 3

        for kernel in kernels:
            if original_is_color:
                otf_k = psf2otf(kernel, img_shape[:2])
                otf_k = otf_k.unsqueeze(2).repeat(1, 1, img_shape[2])
            else:
                otf_k = psf2otf(kernel, img_shape)
            otf_kernels.append(otf_k)

        return otf_kernels

    def _compute_gradient_otfs(self, img_shape: Tuple) -> torch.Tensor:
        """
        Compute gradient operator OTFs (DTD matrix).

        Args:
            img_shape: Image shape tuple

        Returns:
            DTD matrix tensor
        """
        original_is_color = len(img_shape) == 3 and img_shape[2] == 3

        # Create finite difference kernels
        dx_kernel = torch.tensor([[1, -1]], dtype=torch.float32, device=self.device)
        dy_kernel = torch.tensor([[1], [-1]], dtype=torch.float32, device=self.device)

        if original_is_color:
            otfDx = psf2otf(dx_kernel, img_shape[:2])
            otfDy = psf2otf(dy_kernel, img_shape[:2])
            DTD = torch.abs(otfDx) ** 2 + torch.abs(otfDy) ** 2
            DTD = DTD.unsqueeze(2).repeat(1, 1, img_shape[2])
        else:
            otfDx = psf2otf(dx_kernel, img_shape)
            otfDy = psf2otf(dy_kernel, img_shape)
            DTD = torch.abs(otfDx) ** 2 + torch.abs(otfDy) ** 2

        return DTD

    def _get_image_shape(self, data: Dict) -> Tuple:
        """
        Extract image shape from data.

        Args:
            data: Data dictionary

        Returns:
            Image shape tuple
        """
        if "blurred_images" in data:
            return data["blurred_images"][0].shape
        elif "observations" in data:
            observations = data["observations"]
            if isinstance(observations, list):
                return observations[0].shape
            else:
                return observations.shape
        else:
            raise ValueError("Cannot determine image shape from data")

    def _is_color_image(self, img_shape: Tuple) -> bool:
        """
        Check if image is color based on shape.

        Args:
            img_shape: Image shape tuple

        Returns:
            True if color image, False if grayscale
        """
        return len(img_shape) == 3 and img_shape[2] == 3

    def get_algorithm_name(self) -> str:
        """
        Get the algorithm name.

        Returns:
            Algorithm name string
        """
        return self.__class__.__name__

    def get_regularizer_type(self) -> str:
        """
        Get the regularizer type.

        Returns:
            "TV" or "PnP"
        """
        if isinstance(self.regularizer, TVRegularizer):
            return "TV"
        elif isinstance(self.regularizer, PnPRegularizer):
            return "PnP"
        else:
            return "Unknown"
