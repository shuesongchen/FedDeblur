"""
LocDeblur (Local Deblurring)
"""

import torch
from typing import Dict, Tuple, List
from .base_algorithm import BaseAlgorithm
from ..regularizers import TVRegularizer, PnPRegularizer
from ..utils.math_ops import Dx, Dy, Dxt, Dyt, psf2otf
from ..utils.patch_utils import (
    extract_patch_from_masked_observation,
    place_patch_in_full_image,
    average_overlapping_patches,
)


class LocDeblur(BaseAlgorithm):
    """
    LocDeblur algorithm implementation.
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

        x = torch.zeros(img_shape, device=self.device)

        per_kernel_vars = []
        for k in range(n_kernels):
            f_k = blurred_images[k].to(self.device)

            if isinstance(self.regularizer, TVRegularizer):
                kernel_vars = {
                    "x_k": f_k.clone(),
                    "z_x": torch.zeros_like(f_k),
                    "z_y": torch.zeros_like(f_k),
                    "mu_x": torch.zeros_like(f_k),
                    "mu_y": torch.zeros_like(f_k),
                    "f_k": f_k,
                }
            else:
                kernel_vars = {
                    "x_k": f_k.clone(),
                    "z_k": torch.zeros_like(f_k),
                    "mu_k": torch.zeros_like(f_k),
                    "f_k": f_k,
                }
            per_kernel_vars.append(kernel_vars)

        aux_vars = {
            "per_kernel_vars": per_kernel_vars,
            "blurred_images": blurred_images,
            "n_kernels": n_kernels,
        }

        otf_kernels = self._compute_otfs(kernels, img_shape)
        if isinstance(self.regularizer, TVRegularizer):
            DTD = self._compute_gradient_otfs(img_shape)
        else:
            DTD = None

        otfs = {"otf_kernels": otf_kernels, "DTD": DTD, "n_kernels": n_kernels}

        return x, aux_vars, otfs

    def _initialize_partial(
        self, data: Dict, img_shape: Tuple, is_color: bool
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        kernels = data["kernels"]
        observations = data["observations"]
        masks = data["masks"]
        patch_coords = data["patch_coords"]
        patch_size = data["patch_size"]
        n_kernels = len(kernels)

        x = torch.zeros(img_shape, device=self.device)

        per_kernel_vars = []
        for k in range(n_kernels):
            if isinstance(observations, list):
                obs_k = observations[k].to(self.device)

            if isinstance(masks, list):
                mask_k = masks[k].to(self.device)

            coords_k = patch_coords[k]

            patch_k = extract_patch_from_masked_observation(
                obs_k, mask_k, coords_k, patch_size
            )

            if isinstance(self.regularizer, TVRegularizer):
                kernel_vars = {
                    "x_k": patch_k.clone(),
                    "z_x": torch.zeros_like(patch_k),
                    "z_y": torch.zeros_like(patch_k),
                    "mu_x": torch.zeros_like(patch_k),
                    "mu_y": torch.zeros_like(patch_k),
                    "obs_patch": patch_k.clone(),
                    "coords": coords_k,
                    "patch_size": patch_size,
                }
            else:
                kernel_vars = {
                    "x_k": patch_k.clone(),
                    "z_k": torch.zeros_like(patch_k),
                    "mu_k": torch.zeros_like(patch_k),
                    "obs_patch": patch_k.clone(),
                    "coords": coords_k,
                    "patch_size": patch_size,
                }
            per_kernel_vars.append(kernel_vars)

        aux_vars = {
            "per_kernel_vars": per_kernel_vars,
            "observations": observations,
            "masks": masks,
            "patch_coords": patch_coords,
            "patch_size": patch_size,
            "n_kernels": n_kernels,
        }

        patch_shape = (
            (patch_size, patch_size, img_shape[2])
            if is_color
            else (patch_size, patch_size)
        )
        otf_kernels = self._compute_otfs(kernels, patch_shape)
        if isinstance(self.regularizer, TVRegularizer):
            DTD = self._compute_gradient_otfs(patch_shape)
        else:
            DTD = None

        otfs = {"otf_kernels": otf_kernels, "DTD": DTD, "n_kernels": n_kernels}

        return x, aux_vars, otfs

    def _admm_iteration(
        self, x: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        if isinstance(self.regularizer, TVRegularizer):
            return self._tv_admm_iteration(x, aux_vars, otfs, data)
        else:
            return self._pnp_admm_iteration(x, aux_vars, otfs, data)

    def _tv_admm_iteration(
        self, x: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        img_shape = self._get_image_shape(data)
        is_color = self._is_color_image(img_shape)
        is_partial = "observations" in data

        per_kernel_vars = aux_vars["per_kernel_vars"]
        n_kernels = aux_vars["n_kernels"]
        otf_kernels = otfs["otf_kernels"]
        DTD = otfs["DTD"]

        for k in range(n_kernels):
            kernel_vars = per_kernel_vars[k]
            x_k = kernel_vars["x_k"]

            if is_partial:
                z_x = kernel_vars["z_x"]
                z_y = kernel_vars["z_y"]
                mu_x = kernel_vars["mu_x"]
                mu_y = kernel_vars["mu_y"]
                obs_patch = kernel_vars["obs_patch"]

                z_x, z_y = self.regularizer.step(x_k, mu_x, mu_y)
                kernel_vars["z_x"] = z_x
                kernel_vars["z_y"] = z_y

                otfH = otf_kernels[k]
                Denom = torch.abs(otfH) ** 2 + self.rho * DTD

                if is_color:
                    x_k_new = torch.zeros_like(x_k)
                    for c in range(x_k.shape[2]):
                        grad_transpose_term = Dxt(
                            z_x[:, :, c] + mu_x[:, :, c] / self.rho
                        ) + Dyt(z_y[:, :, c] + mu_y[:, :, c] / self.rho)

                        Nomin = torch.conj(otfH[:, :, c]) * torch.fft.fft2(
                            obs_patch[:, :, c]
                        ) + self.rho * torch.fft.fft2(grad_transpose_term)
                        x_k_new[:, :, c] = torch.real(
                            torch.fft.ifft2(Nomin / Denom[:, :, c])
                        )
                    x_k = x_k_new
                else:
                    grad_transpose_term = Dxt(z_x + mu_x / self.rho) + Dyt(
                        z_y + mu_y / self.rho
                    )

                    Nomin = torch.conj(otfH) * torch.fft.fft2(
                        obs_patch
                    ) + self.rho * torch.fft.fft2(grad_transpose_term)
                    x_k = torch.real(torch.fft.ifft2(Nomin / Denom))

                kernel_vars["x_k"] = x_k

                grad_x_new = Dx(x_k)
                grad_y_new = Dy(x_k)

                kernel_vars["mu_x"] = mu_x + self.gamma * self.rho * (z_x - grad_x_new)
                kernel_vars["mu_y"] = mu_y + self.gamma * self.rho * (z_y - grad_y_new)

            else:
                z_x = kernel_vars["z_x"]
                z_y = kernel_vars["z_y"]
                mu_x = kernel_vars["mu_x"]
                mu_y = kernel_vars["mu_y"]
                f_k = kernel_vars["f_k"]

                z_x, z_y = self.regularizer.step(x_k, mu_x, mu_y)
                kernel_vars["z_x"] = z_x
                kernel_vars["z_y"] = z_y

                otfH = otf_kernels[k]
                Denom = torch.abs(otfH) ** 2 + self.rho * DTD

                if is_color:
                    x_k_new = torch.zeros_like(x_k)
                    for c in range(x_k.shape[2]):
                        grad_transpose_term = Dxt(
                            z_x[:, :, c] + mu_x[:, :, c] / self.rho
                        ) + Dyt(z_y[:, :, c] + mu_y[:, :, c] / self.rho)

                        Nomin = torch.conj(otfH[:, :, c]) * torch.fft.fft2(
                            f_k[:, :, c]
                        ) + self.rho * torch.fft.fft2(grad_transpose_term)
                        x_k_new[:, :, c] = torch.real(
                            torch.fft.ifft2(Nomin / Denom[:, :, c])
                        )
                    x_k = x_k_new
                else:
                    grad_transpose_term = Dxt(z_x + mu_x / self.rho) + Dyt(
                        z_y + mu_y / self.rho
                    )

                    Nomin = torch.conj(otfH) * torch.fft.fft2(
                        f_k
                    ) + self.rho * torch.fft.fft2(grad_transpose_term)
                    x_k = torch.real(torch.fft.ifft2(Nomin / Denom))

                kernel_vars["x_k"] = x_k

                grad_x_new = Dx(x_k)
                grad_y_new = Dy(x_k)

                kernel_vars["mu_x"] = mu_x + self.gamma * self.rho * (z_x - grad_x_new)
                kernel_vars["mu_y"] = mu_y + self.gamma * self.rho * (z_y - grad_y_new)

        if is_partial:
            patch_results = []
            for kernel_vars in per_kernel_vars:
                x_k = kernel_vars["x_k"]
                coords = kernel_vars["coords"]
                patch_size = kernel_vars["patch_size"]

                x_k_full = place_patch_in_full_image(x_k, coords, img_shape, patch_size)
                patch_results.append(x_k_full)

            x = average_overlapping_patches(patch_results)
        else:
            x_list = [kernel_vars["x_k"] for kernel_vars in per_kernel_vars]
            x = torch.stack(x_list).mean(dim=0)

        return x, aux_vars

    def _pnp_admm_iteration(
        self, x: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        img_shape = self._get_image_shape(data)
        is_color = self._is_color_image(img_shape)
        is_partial = "observations" in data

        per_kernel_vars = aux_vars["per_kernel_vars"]
        n_kernels = aux_vars["n_kernels"]
        otf_kernels = otfs["otf_kernels"]

        for k in range(n_kernels):
            kernel_vars = per_kernel_vars[k]
            x_k = kernel_vars["x_k"]

            if is_partial:
                z_k = kernel_vars["z_k"]
                mu_k = kernel_vars["mu_k"]
                obs_patch = kernel_vars["obs_patch"]

                denoiser_input = x_k - mu_k / self.rho
                z_k = self.regularizer.step(denoiser_input)
                kernel_vars["z_k"] = z_k

                otfH = otf_kernels[k]

                if is_color:
                    x_k_new = torch.zeros_like(x_k)
                    for c in range(x_k.shape[2]):
                        numerator = 2.0 * torch.conj(otfH[:, :, c]) * torch.fft.fft2(
                            obs_patch[:, :, c]
                        ) + torch.fft.fft2(self.rho * z_k[:, :, c] + mu_k[:, :, c])
                        denominator = 2.0 * torch.abs(otfH[:, :, c]) ** 2 + self.rho
                        x_k_new[:, :, c] = torch.real(
                            torch.fft.ifft2(numerator / denominator)
                        )
                    x_k = x_k_new
                else:
                    numerator = 2.0 * torch.conj(otfH) * torch.fft.fft2(
                        obs_patch
                    ) + torch.fft.fft2(self.rho * z_k + mu_k)
                    denominator = 2.0 * torch.abs(otfH) ** 2 + self.rho
                    x_k = torch.real(torch.fft.ifft2(numerator / denominator))

                kernel_vars["x_k"] = x_k

                kernel_vars["mu_k"] = mu_k + self.gamma * self.rho * (z_k - x_k)

            else:
                z_k = kernel_vars["z_k"]
                mu_k = kernel_vars["mu_k"]
                f_k = kernel_vars["f_k"]

                denoiser_input = x_k - mu_k / self.rho
                z_k = self.regularizer.step(denoiser_input)
                kernel_vars["z_k"] = z_k

                otfH = otf_kernels[k]

                if is_color:
                    x_k_new = torch.zeros_like(x_k)
                    for c in range(x_k.shape[2]):
                        numerator = 2.0 * torch.conj(otfH[:, :, c]) * torch.fft.fft2(
                            f_k[:, :, c]
                        ) + torch.fft.fft2(self.rho * z_k[:, :, c] + mu_k[:, :, c])
                        denominator = 2.0 * torch.abs(otfH[:, :, c]) ** 2 + self.rho
                        x_k_new[:, :, c] = torch.real(
                            torch.fft.ifft2(numerator / denominator)
                        )
                    x_k = x_k_new
                else:
                    numerator = 2.0 * torch.conj(otfH) * torch.fft.fft2(
                        f_k
                    ) + torch.fft.fft2(self.rho * z_k + mu_k)
                    denominator = 2.0 * torch.abs(otfH) ** 2 + self.rho
                    x_k = torch.real(torch.fft.ifft2(numerator / denominator))

                kernel_vars["x_k"] = x_k

                kernel_vars["mu_k"] = mu_k + self.gamma * self.rho * (z_k - x_k)

        if is_partial:
            patch_results = []
            for kernel_vars in per_kernel_vars:
                x_k = kernel_vars["x_k"]
                coords = kernel_vars["coords"]
                patch_size = kernel_vars["patch_size"]

                x_k_full = place_patch_in_full_image(x_k, coords, img_shape, patch_size)
                patch_results.append(x_k_full)

            x = average_overlapping_patches(patch_results)
        else:
            x_list = [kernel_vars["x_k"] for kernel_vars in per_kernel_vars]
            x = torch.stack(x_list).mean(dim=0)

        return x, aux_vars

    def _compute_otfs(
        self, kernels: List[torch.Tensor], img_shape: Tuple
    ) -> List[torch.Tensor]:
        otf_kernels = []
        for kernel in kernels:
            if len(img_shape) == 3:
                otf = psf2otf(kernel, img_shape[:2])
                otf = otf.unsqueeze(2).repeat(1, 1, img_shape[2])
            else:
                otf = psf2otf(kernel, img_shape)
            otf_kernels.append(otf)
        return otf_kernels

    def _compute_gradient_otfs(self, img_shape: Tuple) -> torch.Tensor:
        if len(img_shape) == 3:
            otfDx = psf2otf(
                torch.tensor([[1, -1]], dtype=torch.float32, device=self.device),
                img_shape[:2],
            )
            otfDy = psf2otf(
                torch.tensor([[1], [-1]], dtype=torch.float32, device=self.device),
                img_shape[:2],
            )
            DTD = torch.abs(otfDx) ** 2 + torch.abs(otfDy) ** 2
            DTD = DTD.unsqueeze(2).repeat(1, 1, img_shape[2])
        else:
            otfDx = psf2otf(
                torch.tensor([[1, -1]], dtype=torch.float32, device=self.device),
                img_shape,
            )
            otfDy = psf2otf(
                torch.tensor([[1], [-1]], dtype=torch.float32, device=self.device),
                img_shape,
            )
            DTD = torch.abs(otfDx) ** 2 + torch.abs(otfDy) ** 2
        return DTD
