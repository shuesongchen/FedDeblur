import os
import torch
from typing import Dict, Tuple, List
import torchvision.io as tvio
from pathlib import Path


def _generate_data_path(
    observation: str, blur_type: str, n_clients: int, noise_std: float
) -> str:
    if observation == "complete":
        if noise_std == 0.5:
            suffix = "0-5"
        elif noise_std == int(noise_std):
            suffix = str(int(noise_std))
        else:
            suffix = str(noise_std)
        return f"data_complete_n_{n_clients}_S_{suffix}"
    else:
        if noise_std == 0.5:
            suffix = "0-5"
        elif noise_std == int(noise_std):
            suffix = str(int(noise_std))
        else:
            suffix = str(noise_std)
        return f"data_partial_{blur_type}_S_{suffix}"


class DataLoader:
    def __init__(
        self, base_data_dir: str = "data", base_images_dir: str = "images", device=None
    ):
        self.base_data_dir = os.path.abspath(base_data_dir)
        self.base_images_dir = os.path.abspath(base_images_dir)
        self.device = device

    def _get_data_path(self, config: dict) -> str:
        observation = config["observation"]
        noise_std = config["noise_std"]

        if observation == "complete":
            n_clients = config["n_clients"]
            blur_type = "motion"
        else:
            blur_type = config["blur_type"]
            n_clients = 1

        dir_name = _generate_data_path(observation, blur_type, n_clients, noise_std)
        data_path = os.path.join(self.base_data_dir, dir_name)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        return data_path

    def _get_available_configs(self, observation: str, blur_type: str) -> List[Tuple]:
        configs = []

        if not os.path.exists(self.base_data_dir):
            return configs

        for item in os.listdir(self.base_data_dir):
            if not os.path.isdir(os.path.join(self.base_data_dir, item)):
                continue

            if observation == "complete":
                if item.startswith("data_complete_n_"):
                    try:
                        parts = item.split("_")
                        n_clients = int(parts[3])
                        noise_str = parts[5]
                        if noise_str == "0-5":
                            noise_std = 0.5
                        else:
                            noise_std = float(noise_str)
                        configs.append((n_clients, noise_std))
                    except (IndexError, ValueError):
                        continue
            else:  # partial observation
                if item.startswith(f"data_partial_{blur_type}_S_"):
                    try:
                        # Extract noise_std from name
                        noise_str = item.split("_S_")[1]
                        if noise_str == "0-5":
                            noise_std = 0.5
                        else:
                            noise_std = float(noise_str)
                        configs.append((1, noise_std))
                    except (IndexError, ValueError):
                        continue

        return sorted(configs)

    def load_single_image_data(
        self, image_id: str, image_type: str, config: dict
    ) -> Dict:
        data_path = self._get_data_path(config)
        image_dir = os.path.join(data_path, image_type, f"image_{image_id}")

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image data directory not found: {image_dir}")

        kernels_path = os.path.join(image_dir, "blur_kernels.pt")
        metadata_path = os.path.join(image_dir, "metadata.pt")

        if config["observation"] == "complete":
            blurred_path = os.path.join(image_dir, "blurred_images.pt")
            required_files = [kernels_path, blurred_path, metadata_path]
        else:
            masks_path = os.path.join(image_dir, "masks.pt")
            observations_path = os.path.join(image_dir, "masked_observations.pt")
            required_files = [
                kernels_path,
                masks_path,
                observations_path,
                metadata_path,
            ]

        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(
                f"Missing required data files in {image_dir}: {missing_files}"
            )

        data = {}

        kernels = torch.load(kernels_path, weights_only=True)
        data["metadata"] = torch.load(metadata_path, weights_only=True)

        if isinstance(kernels, list):
            data["kernels"] = [k.to(self.device) for k in kernels] if self.device else kernels

        if config["observation"] == "complete":
            blurred_images = torch.load(blurred_path, weights_only=True)
            if isinstance(blurred_images, list):
                data["blurred_images"] = [img.to(self.device) for img in blurred_images] if self.device else blurred_images
        else:
            masks = torch.load(masks_path, weights_only=True)
            observations = torch.load(observations_path, weights_only=True)

            if isinstance(masks, list):
                data["masks"] = [m.to(self.device) for m in masks] if self.device else masks

            if isinstance(observations, list):
                data["observations"] = [obs.to(self.device) for obs in observations] if self.device else observations

            if "patch_coords" in data["metadata"]:
                data["patch_coords"] = data["metadata"]["patch_coords"]
            if "patch_size" in data["metadata"]:
                data["patch_size"] = data["metadata"]["patch_size"]

        clean_image_path = os.path.join(
            self.base_images_dir, "clean_images", f"clean_{image_id}.png"
        )
        if os.path.exists(clean_image_path):
            try:
                clean_image = tvio.read_image(clean_image_path).float()

                if image_type == "grayscale":
                    if clean_image.shape[0] == 1:
                        clean_image = clean_image.squeeze(0)
                    elif clean_image.shape[0] == 3:
                        clean_image = clean_image.mean(dim=0)
                else:
                    if clean_image.shape[0] == 3:
                        clean_image = clean_image.permute(1, 2, 0)
                    elif clean_image.shape[0] == 1:
                        clean_image = clean_image.repeat(3, 1, 1).permute(1, 2, 0)

                data["clean_image"] = (
                    clean_image.to(self.device) if self.device else clean_image
                )
            except Exception as e:
                print(f"Warning: Failed to load clean image {clean_image_path}: {e}")

        return data

    def get_image_list(self, config: dict, image_type: str) -> List[str]:
        data_path = self._get_data_path(config)
        type_dir = os.path.join(data_path, image_type)

        if not os.path.exists(type_dir):
            raise FileNotFoundError(f"Image directory not found")

        image_dirs = []
        for item in os.listdir(type_dir):
            item_path = os.path.join(type_dir, item)
            if os.path.isdir(item_path) and item.startswith("image_"):
                image_id = item.split("_")[1]
                image_dirs.append(image_id)

        return sorted(image_dirs)

    def validate_data_availability(self, config: dict) -> Dict[str, List[str]]:
        data_path = self._get_data_path(config)

        availability = {"grayscale": [], "color": []}

        for image_type in ["grayscale", "color"]:
            try:
                availability[image_type] = self.get_image_list(config, image_type)
            except FileNotFoundError:
                availability[image_type] = []

        return availability

    def get_supported_configurations(self) -> List[Dict]:
        configs = []

        for item in os.listdir(self.base_data_dir):
            if not os.path.isdir(os.path.join(self.base_data_dir, item)):
                continue

            if item.startswith("data_complete_n_"):
                try:
                    parts = item.split("_")
                    n_clients = int(parts[3])
                    noise_str = parts[5]
                    noise_std = 0.5 if noise_str == "0-5" else float(noise_str)
                    configs.append(
                        {
                            "observation": "complete",
                            "blur_type": "motion",
                            "n_clients": n_clients,
                            "noise_std": noise_std,
                        }
                    )
                except (IndexError, ValueError):
                    continue
            elif item.startswith("data_partial_"):
                try:
                    if "_motion_S_" in item:
                        blur_type = "motion"
                        noise_str = item.split("_S_")[1]
                    elif "_synthesis_S_" in item:
                        blur_type = "synthesis"
                        noise_str = item.split("_S_")[1]
                    else:
                        continue

                    noise_std = 0.5 if noise_str == "0-5" else float(noise_str)
                    configs.append(
                        {
                            "observation": "partial",
                            "blur_type": blur_type,
                            "n_clients": 1,
                            "noise_std": noise_std,
                        }
                    )
                except (IndexError, ValueError):
                    continue

        return sorted(
            configs,
            key=lambda x: (
                x["observation"],
                x["blur_type"],
                x["n_clients"],
                x["noise_std"],
            ),
        )
