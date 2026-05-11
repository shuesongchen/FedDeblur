import torch
import numpy as np
from pathlib import Path
from .network_unet import UNetRes as DRUNet


class DRUNetDenoiser:
    def __init__(self, model_type="gray", weights_dir="model_zoo", device=None):
        if model_type != "gray":
            raise ValueError("Regularization only supports grayscale images")

        self.device = device
        self.weights_dir = Path(weights_dir)
        self.model_type = model_type

        self.gray_net = None
        self._load_gray_net()

    def _load_gray_net(self):
        if self.gray_net is None:
            self.gray_net = DRUNet(
                in_nc=2,
                out_nc=1,
                nc=[64, 128, 256, 512],
                nb=4,
                act_mode="R",
                downsample_mode="strideconv",
                upsample_mode="convtranspose",
            )
            weight_path = self.weights_dir / "drunet_gray.pth"
            state = torch.load(str(weight_path), map_location="cpu")
            self.gray_net.load_state_dict(state, strict=True)
            self.gray_net.eval()
            for p in self.gray_net.parameters():
                p.requires_grad = False
            self.gray_net.to(self.device)

    @torch.no_grad()
    def denoise(self, image, sigma_255):
        if image.dim() == 3:
            raise ValueError("Regularization only supports grayscale images.")
        elif image.dim() != 2:
            raise ValueError(
                f"Expected 2D grayscale image tensor, got {image.dim()}D tensor with shape {image.shape}"
            )

        image = torch.clamp(image, 0, 255)
        img_01 = image.to(self.device).float() / 255.0

        img_01 = img_01.unsqueeze(0).unsqueeze(0)

        N, C, H, W = img_01.shape
        original_H, original_W = H, W

        modulo = 16
        pad_H = int(np.ceil(H / modulo) * modulo)
        pad_W = int(np.ceil(W / modulo) * modulo)

        if pad_H != H or pad_W != W:
            pad_bottom = pad_H - H
            pad_right = pad_W - W
            img_01 = torch.nn.functional.pad(
                img_01, (0, pad_right, 0, pad_bottom), mode="replicate"
            )
            H, W = pad_H, pad_W

        sigma_01 = sigma_255 / 255.0
        sigma_map = torch.full(
            (N, 1, H, W), sigma_01, dtype=img_01.dtype, device=self.device
        )

        inp = torch.cat([img_01, sigma_map], dim=1)

        out = self.gray_net(inp).clamp(0, 1)

        if pad_H != original_H or pad_W != original_W:
            out = out[..., :original_H, :original_W]

        out = out.squeeze(0).squeeze(0)

        out = (out * 255.0).to(image.dtype)

        return out
