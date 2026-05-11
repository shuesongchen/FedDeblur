"""
CenDeblur (Centralized Deblurring)
"""

import torch
from typing import Dict, Tuple, List
from .base_algorithm import BaseAlgorithm
from ..regularizers import TVRegularizer, PnPRegularizer
from ..utils.math_ops import Dx, Dy, Dxt, Dyt
from ..utils.patch_utils import average_overlapping_patches_with_masks


class CenDeblur(BaseAlgorithm):
    """
    Centralized Deblurring algorithm implementation.
    """

    def __init__(self, params: Dict, regularizer, device):
        super().__init__(params, regularizer, device)

    def _initialize(self, data: Dict) -> Tuple[torch.Tensor, Dict, Dict]:
        img_shape = self._get_image_shape(data)
        is_color = self._is_color_image(img_shape)
        is_partial = "observations" in data

        if is_partial:
            return self._initialize_partial(data, img_shape, is_color)
        else:
            return self._initialize_complete(data, img_shape, is_color)

    def _initialize_complete(
        self, data: Dict, img_shape: Tuple, is_color: bool
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        kernels = data["kernels"]
        blurred_images = data["blurred_images"]
        n_kernels = len(kernels)

        # Initialize x with the average of all blurred images
        x = torch.stack(blurred_images).mean(dim=0).to(self.device)

        if isinstance(self.regularizer, TVRegularizer):
            aux_vars = {
                "z_x": torch.zeros_like(x),
                "z_y": torch.zeros_like(x),
                "mu_x": torch.zeros_like(x),
                "mu_y": torch.zeros_like(x),
            }
        else:  # PnP regularizer
            aux_vars = {"z": torch.zeros_like(x), "mu": torch.zeros_like(x)}

        # Store blurred images and kernels info
        aux_vars["blurred_images"] = blurred_images
        aux_vars["n_kernels"] = n_kernels

        # Pre-compute OTFs
        otf_kernels = self._compute_otfs(kernels, img_shape)
        DTD = self._compute_gradient_otfs(img_shape)

        # Pre-compute sum of |F(k_k)|^2 for all kernels
        sum_kernel_otf_squared = torch.zeros_like(DTD)
        for otf_k in otf_kernels:
            sum_kernel_otf_squared += torch.abs(otf_k) ** 2

        otfs = {
            "otf_kernels": otf_kernels,
            "DTD": DTD,
            "sum_kernel_otf_squared": sum_kernel_otf_squared,
            "n_kernels": n_kernels,
        }

        return x, aux_vars, otfs

    def _initialize_partial(
        self, data: Dict, img_shape: Tuple, is_color: bool
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        kernels = data["kernels"]
        observations = data["observations"]
        masks = data["masks"]
        n_kernels = len(kernels)

        if isinstance(observations, list):
            x = average_overlapping_patches_with_masks(observations, masks)
            r_k = [torch.zeros_like(obs).to(self.device) for obs in observations]

        if isinstance(self.regularizer, TVRegularizer):
            aux_vars = {
                "r_k": r_k,
                "z_x": torch.zeros_like(x),
                "z_y": torch.zeros_like(x),
                "mu_x": torch.zeros_like(x),
                "mu_y": torch.zeros_like(x),
                "beta_k": [
                    torch.zeros_like(x).to(self.device) for _ in range(n_kernels)
                ],
                "masks": masks,
                "observations": observations,
            }
        else:  # PnP regularizer
            aux_vars = {
                "r_k": r_k,
                "z": torch.zeros_like(x),
                "mu": torch.zeros_like(x),
                "beta_k": [
                    torch.zeros_like(x).to(self.device) for _ in range(n_kernels)
                ],
                "masks": masks,
                "observations": observations,
            }

        aux_vars["n_kernels"] = n_kernels

        # Pre-compute OTFs
        otf_kernels = self._compute_otfs(kernels, img_shape)
        DTD = self._compute_gradient_otfs(img_shape)

        # Pre-compute sum of |OTF|^2 for masked case
        if is_color:
            sum_otf_squared = torch.zeros_like(DTD)
            for k in range(n_kernels):
                for c in range(img_shape[2]):
                    sum_otf_squared[:, :, c] += torch.abs(otf_kernels[k][:, :, c]) ** 2
        else:
            sum_otf_squared = torch.zeros_like(DTD)
            for k in range(n_kernels):
                sum_otf_squared += torch.abs(otf_kernels[k]) ** 2

        otfs = {
            "otf_kernels": otf_kernels,
            "DTD": DTD,
            "sum_otf_squared": sum_otf_squared,
            "n_kernels": n_kernels,
        }

        return x, aux_vars, otfs

    def _admm_iteration(
        self, x: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Perform one CenDeblur ADMM iteration.

        Args:
            x: Current image estimate
            aux_vars: Auxiliary variables
            otfs: Optical transfer functions
            data: Input data

        Returns:
            Updated (x, aux_vars)
        """
        if isinstance(self.regularizer, TVRegularizer):
            return self._tv_admm_iteration(x, aux_vars, otfs, data)
        else:  # PnP regularizer
            return self._pnp_admm_iteration(x, aux_vars, otfs, data)

    def _tv_admm_iteration(
        self, x: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """TV-specific ADMM iteration for CenDeblur (2-block ADMM)."""
        img_shape = self._get_image_shape(data)
        is_color = self._is_color_image(img_shape)
        is_partial = "observations" in data

        otf_kernels = otfs["otf_kernels"]
        DTD = otfs["DTD"]
        n_kernels = otfs["n_kernels"]

        mu_x = aux_vars["mu_x"]
        mu_y = aux_vars["mu_y"]

        z_x, z_y = self.regularizer.step(x, mu_x, mu_y)
        aux_vars["z_x"] = z_x
        aux_vars["z_y"] = z_y

        if is_partial:
            x = self._solve_x_partial_tv(
                x, z_x, z_y, mu_x, mu_y, aux_vars, otfs, is_color, img_shape
            )
        else:
            x = self._solve_x_complete_tv(
                x, z_x, z_y, mu_x, mu_y, aux_vars, otfs, is_color, img_shape
            )

        grad_x = Dx(x)
        grad_y = Dy(x)

        aux_vars["mu_x"] = mu_x + self.gamma * self.rho * (z_x - grad_x)
        aux_vars["mu_y"] = mu_y + self.gamma * self.rho * (z_y - grad_y)

        if "beta_k" in aux_vars:
            is_partial = "observations" in data
            if is_partial:
                img_shape = self._get_image_shape(data)
                is_color = self._is_color_image(img_shape)
                otf_kernels = otfs["otf_kernels"]
                observations = aux_vars["observations"]
                masks = aux_vars["masks"]
                beta_k = aux_vars["beta_k"]
                r_k = aux_vars["r_k"]

                for k in range(otfs["n_kernels"]):
                    if isinstance(observations, list):
                        y_k = observations[k].to(self.device)
                    otf_k = otf_kernels[k]

                    if is_color:
                        k_k_conv_x_new = torch.zeros_like(x)
                        for c in range(img_shape[2]):
                            k_k_conv_x_new[:, :, c] = torch.real(
                                torch.fft.ifft2(
                                    otf_k[:, :, c] * torch.fft.fft2(x[:, :, c])
                                )
                            )
                    else:
                        k_k_conv_x_new = torch.real(
                            torch.fft.ifft2(otf_k * torch.fft.fft2(x))
                        )

                    aux_vars["beta_k"][k] = beta_k[k] + self.gamma * self.rho * (
                        r_k[k] - (k_k_conv_x_new - y_k)
                    )

        return x, aux_vars

    def _pnp_admm_iteration(
        self, x: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """PnP-specific ADMM iteration for CenDeblur."""
        img_shape = self._get_image_shape(data)
        is_color = self._is_color_image(img_shape)
        is_partial = "observations" in data

        otf_kernels = otfs["otf_kernels"]
        n_kernels = otfs["n_kernels"]

        mu = aux_vars["mu"]

        z_input = x - mu / self.rho
        z = self.regularizer.step(z_input)
        aux_vars["z"] = z

        if is_partial:
            x = self._solve_x_partial_pnp(x, z, mu, aux_vars, otfs, is_color, img_shape)
        else:
            x = self._solve_x_complete_pnp(
                x, z, mu, aux_vars, otfs, is_color, img_shape
            )

        aux_vars["mu"] = mu + self.gamma * self.rho * (z - x)

        if "beta_k" in aux_vars:
            is_partial = "observations" in data
            if is_partial:
                img_shape = self._get_image_shape(data)
                is_color = self._is_color_image(img_shape)
                otf_kernels = otfs["otf_kernels"]
                observations = aux_vars["observations"]
                masks = aux_vars["masks"]
                beta_k = aux_vars["beta_k"]
                r_k = aux_vars["r_k"]

                for k in range(otfs["n_kernels"]):
                    if isinstance(observations, list):
                        y_k = observations[k].to(self.device)
                    otf_k = otf_kernels[k]

                    if is_color:
                        k_k_conv_x_new = torch.zeros_like(x)
                        for c in range(img_shape[2]):
                            k_k_conv_x_new[:, :, c] = torch.real(
                                torch.fft.ifft2(
                                    otf_k[:, :, c] * torch.fft.fft2(x[:, :, c])
                                )
                            )
                    else:
                        k_k_conv_x_new = torch.real(
                            torch.fft.ifft2(otf_k * torch.fft.fft2(x))
                        )

                    aux_vars["beta_k"][k] = beta_k[k] + self.gamma * self.rho * (
                        r_k[k] - (k_k_conv_x_new - y_k)
                    )

        return x, aux_vars

    def _solve_x_complete_tv(
        self,
        x: torch.Tensor,
        z_x: torch.Tensor,
        z_y: torch.Tensor,
        mu_x: torch.Tensor,
        mu_y: torch.Tensor,
        aux_vars: Dict,
        otfs: Dict,
        is_color: bool,
        img_shape: Tuple,
    ) -> torch.Tensor:
        """Solve x subproblem for complete observation with TV regularization."""
        otf_kernels = otfs["otf_kernels"]
        DTD = otfs["DTD"]
        sum_kernel_otf_squared = otfs["sum_kernel_otf_squared"]
        n_kernels = otfs["n_kernels"]
        blurred_images = aux_vars["blurred_images"]

        sum_kernel_data = torch.zeros_like(x, dtype=torch.complex64)

        if is_color:
            for c in range(img_shape[2]):
                channel_sum = torch.zeros(
                    img_shape[:2], dtype=torch.complex64, device=self.device
                )
                for k in range(n_kernels):
                    otf_k = otf_kernels[k][:, :, c]
                    y_k = (
                        blurred_images[k][:, :, c]
                        if blurred_images[k].dim() == 3
                        else blurred_images[k]
                    )
                    channel_sum += torch.conj(otf_k) * torch.fft.fft2(y_k)
                sum_kernel_data[:, :, c] = channel_sum
        else:
            for k in range(n_kernels):
                otf_k = otf_kernels[k]
                y_k = blurred_images[k]
                sum_kernel_data += torch.conj(otf_k) * torch.fft.fft2(y_k)

        if is_color:
            x_new = torch.zeros_like(x)
            for c in range(img_shape[2]):
                z_x_c = z_x[:, :, c] if z_x.dim() == 3 else z_x
                z_y_c = z_y[:, :, c] if z_y.dim() == 3 else z_y
                mu_x_c = mu_x[:, :, c] if mu_x.dim() == 3 else mu_x
                mu_y_c = mu_y[:, :, c] if mu_y.dim() == 3 else mu_y

                grad_transpose_term = Dxt(z_x_c + mu_x_c / self.rho) + Dyt(
                    z_y_c + mu_y_c / self.rho
                )

                numerator = (2.0 / n_kernels) * sum_kernel_data[
                    :, :, c
                ] + torch.fft.fft2(self.rho * grad_transpose_term)
                denominator = (2.0 / n_kernels) * sum_kernel_otf_squared[
                    :, :, c
                ] + self.rho * DTD[:, :, c]
                x_new[:, :, c] = torch.real(torch.fft.ifft2(numerator / denominator))
            return x_new
        else:
            grad_transpose_term = Dxt(z_x + mu_x / self.rho) + Dyt(
                z_y + mu_y / self.rho
            )

            numerator = (2.0 / n_kernels) * sum_kernel_data + torch.fft.fft2(
                self.rho * grad_transpose_term
            )
            denominator = (2.0 / n_kernels) * sum_kernel_otf_squared + self.rho * DTD
            return torch.real(torch.fft.ifft2(numerator / denominator))

    def _solve_x_complete_pnp(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        mu: torch.Tensor,
        aux_vars: Dict,
        otfs: Dict,
        is_color: bool,
        img_shape: Tuple,
    ) -> torch.Tensor:
        """Solve x subproblem for complete observation with PnP regularization."""
        otf_kernels = otfs["otf_kernels"]
        sum_kernel_otf_squared = otfs["sum_kernel_otf_squared"]
        n_kernels = otfs["n_kernels"]
        blurred_images = aux_vars["blurred_images"]

        sum_kernel_data = torch.zeros_like(x, dtype=torch.complex64)

        if is_color:
            for c in range(img_shape[2]):
                channel_sum = torch.zeros(
                    img_shape[:2], dtype=torch.complex64, device=self.device
                )
                for k in range(n_kernels):
                    otf_k = otf_kernels[k][:, :, c]
                    y_k = (
                        blurred_images[k][:, :, c]
                        if blurred_images[k].dim() == 3
                        else blurred_images[k]
                    )
                    channel_sum += torch.conj(otf_k) * torch.fft.fft2(y_k)
                sum_kernel_data[:, :, c] = channel_sum
        else:
            for k in range(n_kernels):
                otf_k = otf_kernels[k]
                y_k = blurred_images[k]
                sum_kernel_data += torch.conj(otf_k) * torch.fft.fft2(y_k)

        if is_color:
            x_new = torch.zeros_like(x)
            for c in range(img_shape[2]):
                z_c = z[:, :, c] if z.dim() == 3 else z
                mu_c = mu[:, :, c] if mu.dim() == 3 else mu

                numerator = (2.0 / n_kernels) * sum_kernel_data[
                    :, :, c
                ] + torch.fft.fft2(self.rho * z_c + mu_c)
                denominator = (2.0 / n_kernels) * sum_kernel_otf_squared[
                    :, :, c
                ] + self.rho
                x_new[:, :, c] = torch.real(torch.fft.ifft2(numerator / denominator))
            return x_new
        else:
            numerator = (2.0 / n_kernels) * sum_kernel_data + torch.fft.fft2(
                self.rho * z + mu
            )
            denominator = (2.0 / n_kernels) * sum_kernel_otf_squared + self.rho
            return torch.real(torch.fft.ifft2(numerator / denominator))

    def _solve_x_partial_tv(
        self,
        x: torch.Tensor,
        z_x: torch.Tensor,
        z_y: torch.Tensor,
        mu_x: torch.Tensor,
        mu_y: torch.Tensor,
        aux_vars: Dict,
        otfs: Dict,
        is_color: bool,
        img_shape: Tuple,
    ) -> torch.Tensor:
        """Solve x subproblem for partial observation with TV regularization."""
        r_k = aux_vars["r_k"]
        beta_k = aux_vars["beta_k"]
        observations = aux_vars["observations"]
        masks = aux_vars["masks"]
        otf_kernels = otfs["otf_kernels"]
        DTD = otfs["DTD"]
        sum_otf_squared = otfs["sum_otf_squared"]
        n_kernels = otfs["n_kernels"]

        for k in range(n_kernels):
            if isinstance(observations, list):
                y_k = observations[k].to(self.device)
            M_k = masks[k]
            otf_k = otf_kernels[k]

            if is_color:
                k_k_conv_x = torch.zeros_like(x)
                for c in range(img_shape[2]):
                    k_k_conv_x[:, :, c] = torch.real(
                        torch.fft.ifft2(otf_k[:, :, c] * torch.fft.fft2(x[:, :, c]))
                    )
            else:
                k_k_conv_x = torch.real(torch.fft.ifft2(otf_k * torch.fft.fft2(x)))

            if is_color:
                for c in range(img_shape[2]):
                    numerator = (
                        self.rho * (k_k_conv_x[:, :, c] - y_k[:, :, c])
                        - beta_k[k][:, :, c]
                    )
                    denominator = (2.0 / n_kernels) * M_k + self.rho
                    r_k[k][:, :, c] = numerator / (denominator + 1e-10)
            else:
                numerator = self.rho * (k_k_conv_x - y_k) - beta_k[k]
                denominator = (2.0 / n_kernels) * M_k + self.rho
                r_k[k] = numerator / (denominator + 1e-10)

        if is_color:
            numerator = (
                torch.zeros_like(torch.fft.fft2(x[:, :, 0]))
                .unsqueeze(2)
                .repeat(1, 1, img_shape[2])
            )

            for k in range(n_kernels):
                for c in range(img_shape[2]):
                    term = (
                        r_k[k][:, :, c]
                        + (observations[k][:, :, c])
                        + beta_k[k][:, :, c] / self.rho
                    )
                    numerator[:, :, c] += torch.conj(
                        otf_kernels[k][:, :, c]
                    ) * torch.fft.fft2(term)

            for c in range(img_shape[2]):
                z_x_c = z_x[:, :, c] if z_x.dim() == 3 else z_x
                z_y_c = z_y[:, :, c] if z_y.dim() == 3 else z_y
                mu_x_c = mu_x[:, :, c] if mu_x.dim() == 3 else mu_x
                mu_y_c = mu_y[:, :, c] if mu_y.dim() == 3 else mu_y

                grad_term = Dxt(z_x_c + mu_x_c / self.rho) + Dyt(
                    z_y_c + mu_y_c / self.rho
                )
                numerator[:, :, c] += torch.fft.fft2(grad_term)

            denominator = sum_otf_squared + DTD

            x_new = torch.zeros_like(x)
            for c in range(img_shape[2]):
                x_new[:, :, c] = torch.real(
                    torch.fft.ifft2(numerator[:, :, c] / denominator[:, :, c])
                )
            return x_new

        else:
            numerator = torch.zeros_like(torch.fft.fft2(x))

            for k in range(n_kernels):
                term = r_k[k] + (observations[k]) + beta_k[k] / self.rho
                numerator += torch.conj(otf_kernels[k]) * torch.fft.fft2(term)

            grad_term = Dxt(z_x + mu_x / self.rho) + Dyt(z_y + mu_y / self.rho)
            numerator += torch.fft.fft2(grad_term)

            denominator = sum_otf_squared + DTD

            return torch.real(torch.fft.ifft2(numerator / denominator))

    def _solve_x_partial_pnp(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        mu: torch.Tensor,
        aux_vars: Dict,
        otfs: Dict,
        is_color: bool,
        img_shape: Tuple,
    ) -> torch.Tensor:
        """Solve x subproblem for partial observation with PnP regularization."""
        r_k = aux_vars["r_k"]
        beta_k = aux_vars["beta_k"]
        observations = aux_vars["observations"]
        masks = aux_vars["masks"]
        otf_kernels = otfs["otf_kernels"]
        sum_otf_squared = otfs["sum_otf_squared"]
        n_kernels = otfs["n_kernels"]

        for k in range(n_kernels):
            if isinstance(observations, list):
                y_k = observations[k].to(self.device)
            M_k = masks[k]
            otf_k = otf_kernels[k]

            if is_color:
                k_k_conv_x = torch.zeros_like(x)
                for c in range(img_shape[2]):
                    k_k_conv_x[:, :, c] = torch.real(
                        torch.fft.ifft2(otf_k[:, :, c] * torch.fft.fft2(x[:, :, c]))
                    )
            else:
                k_k_conv_x = torch.real(torch.fft.ifft2(otf_k * torch.fft.fft2(x)))

            if is_color:
                for c in range(img_shape[2]):
                    numerator = (
                        self.rho * (k_k_conv_x[:, :, c] - y_k[:, :, c])
                        - beta_k[k][:, :, c]
                    )
                    denominator = (2.0 / n_kernels) * M_k + self.rho
                    r_k[k][:, :, c] = numerator / (denominator + 1e-10)
            else:
                numerator = self.rho * (k_k_conv_x - y_k) - beta_k[k]
                denominator = (2.0 / n_kernels) * M_k + self.rho
                r_k[k] = numerator / (denominator + 1e-10)

        if is_color:
            numerator = (
                torch.zeros_like(torch.fft.fft2(x[:, :, 0]))
                .unsqueeze(2)
                .repeat(1, 1, img_shape[2])
            )

            for k in range(n_kernels):
                for c in range(img_shape[2]):
                    term = (
                        r_k[k][:, :, c]
                        + (observations[k][:, :, c])
                        + beta_k[k][:, :, c] / self.rho
                    )
                    numerator[:, :, c] += torch.conj(
                        otf_kernels[k][:, :, c]
                    ) * torch.fft.fft2(term)

            for c in range(img_shape[2]):
                z_c = z[:, :, c] if z.dim() == 3 else z
                mu_c = mu[:, :, c] if mu.dim() == 3 else mu
                pnp_term = z_c + mu_c / self.rho
                numerator[:, :, c] += torch.fft.fft2(pnp_term)

            denominator = sum_otf_squared + 1

            x_new = torch.zeros_like(x)
            for c in range(img_shape[2]):
                x_new[:, :, c] = torch.real(
                    torch.fft.ifft2(numerator[:, :, c] / denominator[:, :, c])
                )
            return x_new

        else:
            numerator = torch.zeros_like(torch.fft.fft2(x))

            for k in range(n_kernels):
                term = r_k[k] + (observations[k]) + beta_k[k] / self.rho
                numerator += torch.conj(otf_kernels[k]) * torch.fft.fft2(term)

            pnp_term = z + mu / self.rho
            numerator += torch.fft.fft2(pnp_term)

            denominator = sum_otf_squared + 1

            return torch.real(torch.fft.ifft2(numerator / denominator))
