"""
FedAvgDeblur (Federated Averaging Deblurring)
"""

import torch
from typing import Dict, Tuple, List
from .base_algorithm import BaseAlgorithm
from ..regularizers import TVRegularizer, PnPRegularizer
from ..utils.math_ops import Dx, Dy, Dxt, Dyt
from ..utils.patch_utils import (
    extract_patch_from_masked_observation,
    place_patch_in_full_image,
    average_overlapping_patches,
)


class FedAvgDeblur(BaseAlgorithm):
    """
    Federated Averaging Deblurring algorithm implementation.
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

        x_bar = torch.stack(blurred_images).mean(dim=0).to(self.device)

        if isinstance(self.regularizer, TVRegularizer):
            mu_x_list = [
                torch.zeros_like(x_bar).to(self.device) for _ in range(n_kernels)
            ]
            mu_y_list = [
                torch.zeros_like(x_bar).to(self.device) for _ in range(n_kernels)
            ]
            aux_vars = {
                "mu_x_list": mu_x_list,
                "mu_y_list": mu_y_list,
                "blurred_images": blurred_images,
                "n_kernels": n_kernels,
            }
        else:
            mu_list = [
                torch.zeros_like(x_bar).to(self.device) for _ in range(n_kernels)
            ]
            aux_vars = {
                "mu_list": mu_list,
                "blurred_images": blurred_images,
                "n_kernels": n_kernels,
            }

        # Pre-compute OTFs
        otf_kernels = self._compute_otfs(kernels, img_shape)
        DTD = self._compute_gradient_otfs(img_shape)

        otfs = {"otf_kernels": otf_kernels, "DTD": DTD, "n_kernels": n_kernels}

        return x_bar, aux_vars, otfs

    def _initialize_partial(
        self, data: Dict, img_shape: Tuple, is_color: bool
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        kernels = data["kernels"]
        observations = data["observations"]
        masks = data["masks"]
        metadata = data["metadata"]
        n_kernels = len(kernels)

        patch_coords = metadata["patch_coords"]
        patch_size = metadata["patch_size"]

        x_bar = torch.rand(img_shape, device=self.device) * 255

        if isinstance(self.regularizer, TVRegularizer):
            mu_x_patches = []
            mu_y_patches = []
            for _ in range(n_kernels):
                if is_color:
                    mu_x_patches.append(
                        torch.zeros(
                            (patch_size, patch_size, img_shape[2]), device=self.device
                        )
                    )
                    mu_y_patches.append(
                        torch.zeros(
                            (patch_size, patch_size, img_shape[2]), device=self.device
                        )
                    )
                else:
                    mu_x_patches.append(
                        torch.zeros((patch_size, patch_size), device=self.device)
                    )
                    mu_y_patches.append(
                        torch.zeros((patch_size, patch_size), device=self.device)
                    )

            aux_vars = {
                "mu_x_patches": mu_x_patches,
                "mu_y_patches": mu_y_patches,
                "patch_coords": patch_coords,
                "patch_size": patch_size,
                "observations": observations,
                "masks": masks,
                "n_kernels": n_kernels,
            }
        else:
            mu_patches = []
            for _ in range(n_kernels):
                if is_color:
                    mu_patches.append(
                        torch.zeros(
                            (patch_size, patch_size, img_shape[2]), device=self.device
                        )
                    )
                else:
                    mu_patches.append(
                        torch.zeros((patch_size, patch_size), device=self.device)
                    )

            aux_vars = {
                "mu_patches": mu_patches,
                "patch_coords": patch_coords,
                "patch_size": patch_size,
                "observations": observations,
                "masks": masks,
                "n_kernels": n_kernels,
            }

        patch_shape = (
            (patch_size, patch_size, img_shape[2])
            if is_color
            else (patch_size, patch_size)
        )
        otf_kernels = self._compute_otfs(kernels, patch_shape)
        DTD = self._compute_gradient_otfs(patch_shape)

        otfs = {"otf_kernels": otf_kernels, "DTD": DTD, "n_kernels": n_kernels}

        return x_bar, aux_vars, otfs

    def _admm_iteration(
        self, x: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        if isinstance(self.regularizer, TVRegularizer):
            return self._tv_admm_iteration(x, aux_vars, otfs, data)
        else:
            return self._pnp_admm_iteration(x, aux_vars, otfs, data)

    def _tv_admm_iteration(
        self, x_bar: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        img_shape = self._get_image_shape(data)
        is_color = self._is_color_image(img_shape)
        is_partial = "observations" in data

        # Handle complete vs partial observation differently
        if is_partial:
            return self._tv_admm_iteration_partial(x_bar, aux_vars, otfs, data)
        else:
            return self._tv_admm_iteration_complete(x_bar, aux_vars, otfs, data)

    def _tv_admm_iteration_complete(
        self, x_bar: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        img_shape = self._get_image_shape(data)
        is_color = self._is_color_image(img_shape)

        blurred_images = aux_vars["blurred_images"]
        mu_x_list = aux_vars["mu_x_list"]
        mu_y_list = aux_vars["mu_y_list"]
        n_kernels = aux_vars["n_kernels"]
        otf_kernels = otfs["otf_kernels"]
        DTD = otfs["DTD"]

        x_k_list = []

        for k in range(n_kernels):
            x_k = x_bar.clone()

            f = blurred_images[k].to(self.device)
            mu_x = mu_x_list[k]
            mu_y = mu_y_list[k]
            otfH = otf_kernels[k]

            z_x, z_y = self.regularizer.step(x_k, mu_x, mu_y)

            Denom = torch.abs(otfH) ** 2 + self.rho * DTD

            if is_color:
                x_k = torch.zeros_like(f)
                for c in range(img_shape[2]):
                    z_x_c = z_x[:, :, c] if z_x.dim() == 3 else z_x
                    z_y_c = z_y[:, :, c] if z_y.dim() == 3 else z_y
                    mu_x_c = mu_x[:, :, c] if mu_x.dim() == 3 else mu_x
                    mu_y_c = mu_y[:, :, c] if mu_y.dim() == 3 else mu_y

                    grad_transpose_term = Dxt(z_x_c + mu_x_c / self.rho) + Dyt(
                        z_y_c + mu_y_c / self.rho
                    )

                    otfH_c = otfH[:, :, c] if otfH.dim() == 3 else otfH
                    Denom_c = Denom[:, :, c] if Denom.dim() == 3 else Denom

                    Nomin = torch.conj(otfH_c) * torch.fft.fft2(
                        f[:, :, c]
                    ) + self.rho * torch.fft.fft2(grad_transpose_term)
                    x_k[:, :, c] = torch.real(torch.fft.ifft2(Nomin / Denom_c))
            else:
                grad_transpose_term = Dxt(z_x + mu_x / self.rho) + Dyt(
                    z_y + mu_y / self.rho
                )

                Nomin = torch.conj(otfH) * torch.fft.fft2(
                    f
                ) + self.rho * torch.fft.fft2(grad_transpose_term)
                x_k = torch.real(torch.fft.ifft2(Nomin / Denom))

            grad_x_new = Dx(x_k)
            grad_y_new = Dy(x_k)

            aux_vars["mu_x_list"][k] = mu_x + self.gamma * self.rho * (z_x - grad_x_new)
            aux_vars["mu_y_list"][k] = mu_y + self.gamma * self.rho * (z_y - grad_y_new)

            x_k_list.append(x_k)

        x_bar = torch.stack(x_k_list).mean(dim=0)
        x_bar = torch.clamp(x_bar, 0, 255)

        return x_bar, aux_vars

    def _tv_admm_iteration_partial(
        self, x_bar: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        img_shape = self._get_image_shape(data)
        is_color = self._is_color_image(img_shape)

        mu_x_patches = aux_vars["mu_x_patches"]
        mu_y_patches = aux_vars["mu_y_patches"]
        patch_coords = aux_vars["patch_coords"]
        patch_size = aux_vars["patch_size"]
        observations = aux_vars["observations"]
        masks = aux_vars["masks"]
        n_kernels = aux_vars["n_kernels"]
        otf_kernels = otfs["otf_kernels"]
        DTD = otfs["DTD"]

        patch_results = []

        for k in range(n_kernels):
            x_k = extract_patch_from_masked_observation(
                x_bar, masks[k], patch_coords[k], patch_size
            )

            if isinstance(observations, list):
                y_k = extract_patch_from_masked_observation(
                    observations[k], masks[k], patch_coords[k], patch_size
                )

            mu_x = mu_x_patches[k]
            mu_y = mu_y_patches[k]

            otf_k = otf_kernels[k]

            z_x, z_y = self.regularizer.step(x_k, mu_x, mu_y)

            Denom = torch.abs(otf_k) ** 2 + self.rho * DTD

            if is_color:
                x_k = torch.zeros_like(y_k)
                for c in range(y_k.shape[2]):
                    z_x_c = z_x[:, :, c] if z_x.dim() == 3 else z_x
                    z_y_c = z_y[:, :, c] if z_y.dim() == 3 else z_y
                    mu_x_c = mu_x[:, :, c] if mu_x.dim() == 3 else mu_x
                    mu_y_c = mu_y[:, :, c] if mu_y.dim() == 3 else mu_y

                    grad_transpose_term = Dxt(z_x_c + mu_x_c / self.rho) + Dyt(
                        z_y_c + mu_y_c / self.rho
                    )

                    otf_k_c = otf_k[:, :, c] if otf_k.dim() == 3 else otf_k
                    Denom_c = Denom[:, :, c] if Denom.dim() == 3 else Denom

                    Nomin = torch.conj(otf_k_c) * torch.fft.fft2(
                        y_k[:, :, c]
                    ) + self.rho * torch.fft.fft2(grad_transpose_term)
                    x_k[:, :, c] = torch.real(torch.fft.ifft2(Nomin / Denom_c))
            else:
                grad_transpose_term = Dxt(z_x + mu_x / self.rho) + Dyt(
                    z_y + mu_y / self.rho
                )

                Nomin = torch.conj(otf_k) * torch.fft.fft2(
                    y_k
                ) + self.rho * torch.fft.fft2(grad_transpose_term)
                x_k = torch.real(torch.fft.ifft2(Nomin / Denom))

            grad_x_new = Dx(x_k)
            grad_y_new = Dy(x_k)

            aux_vars["mu_x_patches"][k] = mu_x + self.gamma * self.rho * (
                z_x - grad_x_new
            )
            aux_vars["mu_y_patches"][k] = mu_y + self.gamma * self.rho * (
                z_y - grad_y_new
            )

            x_k_full = place_patch_in_full_image(
                x_k, patch_coords[k], img_shape, patch_size
            )
            patch_results.append(x_k_full)

        x_bar = average_overlapping_patches(patch_results)
        x_bar = torch.clamp(x_bar, 0, 255)

        return x_bar, aux_vars

    def _pnp_admm_iteration(
        self, x_bar: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        img_shape = self._get_image_shape(data)
        is_color = self._is_color_image(img_shape)
        is_partial = "observations" in data

        if is_partial:
            return self._pnp_admm_iteration_partial(x_bar, aux_vars, otfs, data)
        else:
            return self._pnp_admm_iteration_complete(x_bar, aux_vars, otfs, data)

    def _pnp_admm_iteration_complete(
        self, x_bar: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        img_shape = self._get_image_shape(data)
        is_color = self._is_color_image(img_shape)

        blurred_images = aux_vars["blurred_images"]
        mu_list = aux_vars["mu_list"]
        n_kernels = aux_vars["n_kernels"]
        otf_kernels = otfs["otf_kernels"]

        x_k_list = []

        for k in range(n_kernels):
            x_k = x_bar.clone()

            f = blurred_images[k].to(self.device)
            mu = mu_list[k]
            otfH = otf_kernels[k]

            z_input = x_k - mu / self.rho
            z = self.regularizer.step(z_input)

            if is_color:
                x_k = torch.zeros_like(f)
                for c in range(img_shape[2]):
                    z_c = z[:, :, c] if z.dim() == 3 else z
                    mu_c = mu[:, :, c] if mu.dim() == 3 else mu
                    otfH_c = otfH[:, :, c] if otfH.dim() == 3 else otfH

                    Nomin = 2.0 * torch.conj(otfH_c) * torch.fft.fft2(
                        f[:, :, c]
                    ) + torch.fft.fft2(self.rho * z_c + mu_c)
                    Denom = 2.0 * torch.abs(otfH_c) ** 2 + self.rho
                    x_k[:, :, c] = torch.real(torch.fft.ifft2(Nomin / Denom))
            else:
                Nomin = 2.0 * torch.conj(otfH) * torch.fft.fft2(f) + torch.fft.fft2(
                    self.rho * z + mu
                )
                Denom = 2.0 * torch.abs(otfH) ** 2 + self.rho
                x_k = torch.real(torch.fft.ifft2(Nomin / Denom))

            aux_vars["mu_list"][k] = mu + self.gamma * self.rho * (z - x_k)

            x_k_list.append(x_k)

        x_bar = torch.stack(x_k_list).mean(dim=0)
        x_bar = torch.clamp(x_bar, 0, 255)

        return x_bar, aux_vars

    def _pnp_admm_iteration_partial(
        self, x_bar: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        img_shape = self._get_image_shape(data)
        is_color = self._is_color_image(img_shape)

        mu_patches = aux_vars["mu_patches"]
        patch_coords = aux_vars["patch_coords"]
        patch_size = aux_vars["patch_size"]
        observations = aux_vars["observations"]
        masks = aux_vars["masks"]
        n_kernels = aux_vars["n_kernels"]
        otf_kernels = otfs["otf_kernels"]

        patch_results = []

        for k in range(n_kernels):
            x_k = extract_patch_from_masked_observation(
                x_bar, masks[k], patch_coords[k], patch_size
            )

            if isinstance(observations, list):
                y_k = extract_patch_from_masked_observation(
                    observations[k], masks[k], patch_coords[k], patch_size
                )

            mu = mu_patches[k]

            otf_k = otf_kernels[k]

            z_input = x_k - mu / self.rho
            z = self.regularizer.step(z_input)

            if is_color:
                x_k = torch.zeros_like(y_k)
                for c in range(y_k.shape[2]):
                    z_c = z[:, :, c] if z.dim() == 3 else z
                    mu_c = mu[:, :, c] if mu.dim() == 3 else mu
                    otf_k_c = otf_k[:, :, c] if otf_k.dim() == 3 else otf_k

                    Nomin = 2.0 * torch.conj(otf_k_c) * torch.fft.fft2(
                        y_k[:, :, c]
                    ) + torch.fft.fft2(self.rho * z_c + mu_c)
                    Denom = 2.0 * torch.abs(otf_k_c) ** 2 + self.rho
                    x_k[:, :, c] = torch.real(torch.fft.ifft2(Nomin / Denom))
            else:
                Nomin = 2.0 * torch.conj(otf_k) * torch.fft.fft2(y_k) + torch.fft.fft2(
                    self.rho * z + mu
                )
                Denom = 2.0 * torch.abs(otf_k) ** 2 + self.rho
                x_k = torch.real(torch.fft.ifft2(Nomin / Denom))

            aux_vars["mu_patches"][k] = mu + self.gamma * self.rho * (z - x_k)

            x_k_full = place_patch_in_full_image(
                x_k, patch_coords[k], img_shape, patch_size
            )
            patch_results.append(x_k_full)

        x_bar = average_overlapping_patches(patch_results)
        x_bar = torch.clamp(x_bar, 0, 255)

        return x_bar, aux_vars
