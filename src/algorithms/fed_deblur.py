"""
FedDeblur (Federated Deblurring)
"""

import torch
from typing import Dict, Tuple, List
from .base_algorithm import BaseAlgorithm
from ..regularizers import TVRegularizer, PnPRegularizer
from ..utils.math_ops import Dx, Dy, Dxt, Dyt


class FedDeblur(BaseAlgorithm):
    """
    Federated Deblurring algorithm implementation.
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

        x0 = torch.stack(blurred_images).mean(dim=0).to(self.device)
        x = torch.rand_like(x0).to(self.device) * 255

        x_k = [img.clone().to(self.device) for img in blurred_images]

        if isinstance(self.regularizer, TVRegularizer):
            aux_vars = {
                "x_k": x_k,
                "z_x": torch.zeros_like(x),
                "z_y": torch.zeros_like(x),
                "lambda_k": [
                    torch.zeros_like(x).to(self.device) for _ in range(n_kernels)
                ],
                "mu_x": torch.zeros_like(x).to(self.device),
                "mu_y": torch.zeros_like(x).to(self.device),
            }
        else:
            aux_vars = {
                "x_k": x_k,
                "lambda_k": [
                    torch.zeros_like(x).to(self.device) for _ in range(n_kernels)
                ],
            }

        otf_kernels = self._compute_otfs(kernels, img_shape)
        DTD = self._compute_gradient_otfs(img_shape)

        otfs = {"otf_kernels": otf_kernels, "DTD": DTD, "n_kernels": n_kernels}

        return x, aux_vars, otfs

    def _initialize_partial(
        self, data: Dict, img_shape: Tuple, is_color: bool
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        kernels = data["kernels"]
        observations = data["observations"]
        masks = data["masks"]
        n_kernels = len(kernels)

        if isinstance(observations, list):
            x = torch.rand_like(observations[0]).to(self.device) * 255
            x_k = [obs.clone().to(self.device) for obs in observations]
            r_k = [torch.zeros_like(obs).to(self.device) for obs in observations]

        if isinstance(self.regularizer, TVRegularizer):
            aux_vars = {
                "x_k": x_k,
                "r_k": r_k,
                "z_x": torch.zeros_like(x),
                "z_y": torch.zeros_like(x),
                "lambda_k": [
                    torch.zeros_like(x).to(self.device) for _ in range(n_kernels)
                ],
                "beta_k": [
                    torch.zeros_like(x).to(self.device) for _ in range(n_kernels)
                ],
                "mu_x": torch.zeros_like(x).to(self.device),
                "mu_y": torch.zeros_like(x).to(self.device),
                "masks": masks,
            }
        else:
            aux_vars = {
                "x_k": x_k,
                "r_k": r_k,
                "lambda_k": [
                    torch.zeros_like(x).to(self.device) for _ in range(n_kernels)
                ],
                "beta_k": [
                    torch.zeros_like(x).to(self.device) for _ in range(n_kernels)
                ],
                "masks": masks,
            }

        otf_kernels = self._compute_otfs(kernels, img_shape)
        DTD = self._compute_gradient_otfs(img_shape)

        otfs = {"otf_kernels": otf_kernels, "DTD": DTD, "n_kernels": n_kernels}

        return x, aux_vars, otfs

    def _admm_iteration(
        self, x: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Perform one FedDeblur ADMM iteration.

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
        else:
            return self._pnp_admm_iteration(x, aux_vars, otfs, data)

    def _tv_admm_iteration(
        self, x: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        img_shape = self._get_image_shape(data)
        is_color = self._is_color_image(img_shape)
        is_partial = "observations" in data

        otf_kernels = otfs["otf_kernels"]
        DTD = otfs["DTD"]
        n_kernels = otfs["n_kernels"]

        x_k = aux_vars["x_k"]
        lambda_k = aux_vars["lambda_k"]
        mu_x = aux_vars["mu_x"]
        mu_y = aux_vars["mu_y"]

        if is_partial:
            observations = data["observations"]
            masks = aux_vars["masks"]
            r_k = aux_vars["r_k"]
            beta_k = aux_vars["beta_k"]

            for k in range(n_kernels):
                if isinstance(observations, list):
                    y_k = observations[k].to(self.device)
                M_k = masks[k]
                otf_k = otf_kernels[k]

                if is_color:
                    k_k_conv_x_k = torch.zeros_like(x_k[k])
                    for c in range(img_shape[2]):
                        k_k_conv_x_k[:, :, c] = torch.real(
                            torch.fft.ifft2(
                                otf_k[:, :, c] * torch.fft.fft2(x_k[k][:, :, c])
                            )
                        )

                    for c in range(img_shape[2]):
                        numerator = (
                            self.rho * (k_k_conv_x_k[:, :, c] - y_k[:, :, c])
                            - beta_k[k][:, :, c]
                        )
                        denominator = (2.0 / n_kernels) * M_k + self.rho
                        r_k[k][:, :, c] = numerator / (denominator + 1e-10)
                else:
                    k_k_conv_x_k = torch.real(
                        torch.fft.ifft2(otf_k * torch.fft.fft2(x_k[k]))
                    )
                    numerator = self.rho * (k_k_conv_x_k - y_k) - beta_k[k]
                    denominator = (2.0 / n_kernels) * M_k + self.rho
                    r_k[k] = numerator / (denominator + 1e-10)

                if is_color:
                    x_k_new = torch.zeros_like(x)
                    for c in range(img_shape[2]):
                        numerator = torch.conj(otf_k[:, :, c]) * torch.fft.fft2(
                            r_k[k][:, :, c]
                            + y_k[:, :, c]
                            + beta_k[k][:, :, c] / self.rho
                        ) + torch.fft.fft2(x[:, :, c] - lambda_k[k][:, :, c] / self.rho)
                        denominator = torch.abs(otf_k[:, :, c]) ** 2 + 1
                        x_k_new[:, :, c] = torch.real(
                            torch.fft.ifft2(numerator / denominator)
                        )
                    x_k[k] = x_k_new
                else:
                    numerator = torch.conj(otf_k) * torch.fft.fft2(
                        r_k[k] + y_k + beta_k[k] / self.rho
                    ) + torch.fft.fft2(x - lambda_k[k] / self.rho)
                    denominator = torch.abs(otf_k) ** 2 + 1
                    x_k[k] = torch.real(torch.fft.ifft2(numerator / denominator))
        else:
            blurred_images = data["blurred_images"]

            for k in range(n_kernels):
                y_k = blurred_images[k].to(self.device)
                x_k[k] = self._solve_x_k_complete(
                    x_k[k],
                    x,
                    lambda_k[k],
                    otf_kernels[k],
                    y_k,
                    n_kernels,
                    is_color,
                    img_shape,
                )

        z_x, z_y = self.regularizer.step(x, mu_x, mu_y)
        aux_vars["z_x"] = z_x
        aux_vars["z_y"] = z_y

        x = self._solve_consensus_tv(
            x_k, lambda_k, z_x, z_y, mu_x, mu_y, DTD, n_kernels, is_color, img_shape
        )

        grad_x_new = Dx(x)
        grad_y_new = Dy(x)

        aux_vars["mu_x"] = mu_x + self.gamma * self.rho * (z_x - grad_x_new)
        aux_vars["mu_y"] = mu_y + self.gamma * self.rho * (z_y - grad_y_new)

        for k in range(n_kernels):
            aux_vars["lambda_k"][k] = lambda_k[k] + self.gamma * self.rho * (x_k[k] - x)

        if "beta_k" in aux_vars:
            beta_k = aux_vars["beta_k"]
            observations = data["observations"]
            masks = aux_vars["masks"]

            for k in range(n_kernels):
                if isinstance(observations, list):
                    y_k = observations[k].to(self.device)

                if is_color:
                    k_k_conv_x_k = torch.zeros_like(x_k[k])
                    for c in range(img_shape[2]):
                        k_k_conv_x_k[:, :, c] = torch.real(
                            torch.fft.ifft2(
                                otf_kernels[k][:, :, c]
                                * torch.fft.fft2(x_k[k][:, :, c])
                            )
                        )
                    aux_vars["beta_k"][k] = beta_k[k] + self.gamma * self.rho * (
                        aux_vars["r_k"][k] - k_k_conv_x_k + y_k
                    )
                else:
                    k_k_conv_x_k = torch.real(
                        torch.fft.ifft2(otf_kernels[k] * torch.fft.fft2(x_k[k]))
                    )
                    aux_vars["beta_k"][k] = beta_k[k] + self.gamma * self.rho * (
                        aux_vars["r_k"][k] - k_k_conv_x_k + y_k
                    )

        return x, aux_vars

    def _pnp_admm_iteration(
        self, x: torch.Tensor, aux_vars: Dict, otfs: Dict, data: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        img_shape = self._get_image_shape(data)
        is_color = self._is_color_image(img_shape)
        is_partial = "observations" in data

        otf_kernels = otfs["otf_kernels"]
        n_kernels = otfs["n_kernels"]

        x_k = aux_vars["x_k"]
        lambda_k = aux_vars["lambda_k"]

        if is_partial:
            observations = data["observations"]
            masks = aux_vars["masks"]
            r_k = aux_vars["r_k"]
            beta_k = aux_vars["beta_k"]

            for k in range(n_kernels):
                if isinstance(observations, list):
                    y_k = observations[k].to(self.device)
                M_k = masks[k]
                otf_k = otf_kernels[k]

                if is_color:
                    k_k_conv_x_k = torch.zeros_like(x_k[k])
                    for c in range(img_shape[2]):
                        k_k_conv_x_k[:, :, c] = torch.real(
                            torch.fft.ifft2(
                                otf_k[:, :, c] * torch.fft.fft2(x_k[k][:, :, c])
                            )
                        )

                    for c in range(img_shape[2]):
                        numerator = (
                            self.rho * (k_k_conv_x_k[:, :, c] - y_k[:, :, c])
                            - beta_k[k][:, :, c]
                        )
                        denominator = (2.0 / n_kernels) * M_k + self.rho
                        r_k[k][:, :, c] = numerator / (denominator + 1e-10)
                else:
                    k_k_conv_x_k = torch.real(
                        torch.fft.ifft2(otf_k * torch.fft.fft2(x_k[k]))
                    )
                    numerator = self.rho * (k_k_conv_x_k - y_k) - beta_k[k]
                    denominator = (2.0 / n_kernels) * M_k + self.rho
                    r_k[k] = numerator / (denominator + 1e-10)

                if is_color:
                    x_k_new = torch.zeros_like(x)
                    for c in range(img_shape[2]):
                        numerator = torch.conj(otf_k[:, :, c]) * torch.fft.fft2(
                            r_k[k][:, :, c]
                            + y_k[:, :, c]
                            + beta_k[k][:, :, c] / self.rho
                        ) + torch.fft.fft2(x[:, :, c] - lambda_k[k][:, :, c] / self.rho)
                        denominator = torch.abs(otf_k[:, :, c]) ** 2 + 1
                        x_k_new[:, :, c] = torch.real(
                            torch.fft.ifft2(numerator / denominator)
                        )
                    x_k[k] = x_k_new
                else:
                    numerator = torch.conj(otf_k) * torch.fft.fft2(
                        r_k[k] + y_k + beta_k[k] / self.rho
                    ) + torch.fft.fft2(x - lambda_k[k] / self.rho)
                    denominator = torch.abs(otf_k) ** 2 + 1
                    x_k[k] = torch.real(torch.fft.ifft2(numerator / denominator))

            mask_weights = torch.zeros_like(x).to(self.device)
            weighted_consensus = torch.zeros_like(x).to(self.device)
            for k in range(n_kernels):
                mask_k = masks[k].to(self.device)
                if is_color and mask_k.dim() == 2:
                    mask_k = mask_k.unsqueeze(2).repeat(1, 1, img_shape[2])
                mask_weights += mask_k
                weighted_consensus += mask_k * (x_k[k] + lambda_k[k] / self.rho)
            consensus = weighted_consensus / (mask_weights + 1e-10)
        else:
            blurred_images = data["blurred_images"]

            for k in range(n_kernels):
                y_k = blurred_images[k].to(self.device)
                x_k[k] = self._solve_x_k_complete(
                    x_k[k],
                    x,
                    lambda_k[k],
                    otf_kernels[k],
                    y_k,
                    n_kernels,
                    is_color,
                    img_shape,
                )

            consensus = torch.zeros_like(x).to(self.device)
            for k in range(n_kernels):
                consensus += x_k[k] + lambda_k[k] / self.rho
            consensus = consensus / n_kernels

        x = self.regularizer.step(consensus)
        if is_partial:
            self.regularizer.sigma = 0.995 * self.regularizer.sigma

        if "beta_k" in aux_vars:
            beta_k = aux_vars["beta_k"]
            observations = data["observations"]
            masks = aux_vars["masks"]

            for k in range(n_kernels):
                if isinstance(observations, list):
                    y_k = observations[k].to(self.device)

                if is_color:
                    k_k_conv_x_k = torch.zeros_like(x_k[k])
                    for c in range(img_shape[2]):
                        k_k_conv_x_k[:, :, c] = torch.real(
                            torch.fft.ifft2(
                                otf_kernels[k][:, :, c]
                                * torch.fft.fft2(x_k[k][:, :, c])
                            )
                        )
                    aux_vars["beta_k"][k] = beta_k[k] + self.gamma * self.rho * (
                        aux_vars["r_k"][k] - k_k_conv_x_k + y_k
                    )
                else:
                    k_k_conv_x_k = torch.real(
                        torch.fft.ifft2(otf_kernels[k] * torch.fft.fft2(x_k[k]))
                    )
                    aux_vars["beta_k"][k] = beta_k[k] + self.gamma * self.rho * (
                        aux_vars["r_k"][k] - k_k_conv_x_k + y_k
                    )

                aux_vars["lambda_k"][k] = lambda_k[k] + self.gamma * self.rho * (
                    x_k[k] - x
                )
        else:
            for k in range(n_kernels):
                aux_vars["lambda_k"][k] = lambda_k[k] + self.gamma * self.rho * (
                    x_k[k] - x
                )

        return x, aux_vars

    def _solve_x_k_complete(
        self,
        x_k_curr: torch.Tensor,
        x: torch.Tensor,
        lambda_k: torch.Tensor,
        otf_k: torch.Tensor,
        y_k: torch.Tensor,
        n_kernels: int,
        is_color: bool,
        img_shape: Tuple,
    ) -> torch.Tensor:
        if is_color:
            x_k_new = torch.zeros_like(x)
            for c in range(img_shape[2]):
                numerator = (2.0 / n_kernels) * torch.conj(
                    otf_k[:, :, c]
                ) * torch.fft.fft2(y_k[:, :, c]) + torch.fft.fft2(
                    self.rho * x[:, :, c] - lambda_k[:, :, c]
                )
                denominator = (2.0 / n_kernels) * torch.abs(
                    otf_k[:, :, c]
                ) ** 2 + self.rho
                x_k_new[:, :, c] = torch.real(torch.fft.ifft2(numerator / denominator))
            return x_k_new
        else:
            numerator = (2.0 / n_kernels) * torch.conj(otf_k) * torch.fft.fft2(
                y_k
            ) + torch.fft.fft2(self.rho * x - lambda_k)
            denominator = (2.0 / n_kernels) * torch.abs(otf_k) ** 2 + self.rho
            return torch.real(torch.fft.ifft2(numerator / denominator))

    def _solve_consensus_tv(
        self,
        x_k: List[torch.Tensor],
        lambda_k: List[torch.Tensor],
        z_x: torch.Tensor,
        z_y: torch.Tensor,
        mu_x: torch.Tensor,
        mu_y: torch.Tensor,
        DTD: torch.Tensor,
        n_kernels: int,
        is_color: bool,
        img_shape: Tuple,
    ) -> torch.Tensor:
        sum_term = torch.zeros_like(x_k[0]).to(self.device)
        for k in range(n_kernels):
            sum_term += x_k[k] + lambda_k[k] / self.rho

        if is_color:
            x_new = torch.zeros_like(sum_term)
            for c in range(img_shape[2]):
                z_x_c = z_x[:, :, c] if z_x.dim() == 3 else z_x
                z_y_c = z_y[:, :, c] if z_y.dim() == 3 else z_y
                mu_x_c = mu_x[:, :, c] if mu_x.dim() == 3 else mu_x
                mu_y_c = mu_y[:, :, c] if mu_y.dim() == 3 else mu_y

                grad_term = Dxt(z_x_c + mu_x_c / self.rho) + Dyt(
                    z_y_c + mu_y_c / self.rho
                )

                numerator = torch.fft.fft2(sum_term[:, :, c] + grad_term)
                denominator = n_kernels + DTD[:, :, c]
                x_new[:, :, c] = torch.real(torch.fft.ifft2(numerator / denominator))
            return x_new
        else:
            grad_term = Dxt(z_x + mu_x / self.rho) + Dyt(z_y + mu_y / self.rho)

            numerator = torch.fft.fft2(sum_term + grad_term)
            denominator = n_kernels + DTD
            return torch.real(torch.fft.ifft2(numerator / denominator))
