"""
Patch processing utilities for masked multi-kernel deblurring.

This module provides utilities for extracting patches from masked observations
and placing processed patches back into full images for averaging.
"""

import torch
from typing import List, Tuple


def extract_patch_from_masked_observation(masked_obs: torch.Tensor, mask: torch.Tensor,
                                        patch_coords: Tuple[int, int], patch_size: int) -> torch.Tensor:
    """
    Extract the actual patch region from a masked observation.

    Args:
        masked_obs: The masked observation (can be 2D or 3D)
        mask: The binary mask indicating the patch region
        patch_coords: (start_y, start_x) coordinates of the patch
        patch_size: Size of the patch

    Returns:
        Extracted patch tensor
    """
    start_y, start_x = patch_coords
    end_y = start_y + patch_size
    end_x = start_x + patch_size

    if masked_obs.dim() == 2:  # Grayscale
        return masked_obs[start_y:end_y, start_x:end_x].clone()
    else:  # Color
        return masked_obs[start_y:end_y, start_x:end_x, :].clone()


def place_patch_in_full_image(patch: torch.Tensor, patch_coords: Tuple[int, int],
                             img_shape: Tuple, patch_size: int) -> torch.Tensor:
    """
    Place a processed patch into a full-size image canvas filled with NaN.

    Args:
        patch: The processed patch
        patch_coords: (start_y, start_x) coordinates where to place the patch
        img_shape: Shape of the full image
        patch_size: Size of the patch

    Returns:
        Full-size image with patch placed at correct location, NaN elsewhere
    """
    start_y, start_x = patch_coords
    end_y = start_y + patch_size
    end_x = start_x + patch_size

    if len(img_shape) == 2:  # Grayscale
        full_img = torch.full(img_shape, float('nan'), device=patch.device, dtype=patch.dtype)
        full_img[start_y:end_y, start_x:end_x] = patch
    else:  # Color
        full_img = torch.full(img_shape, float('nan'), device=patch.device, dtype=patch.dtype)
        full_img[start_y:end_y, start_x:end_x, :] = patch

    return full_img


def average_overlapping_patches(patch_images: List[torch.Tensor]) -> torch.Tensor:
    """
    Average multiple patch images using torch.nanmean.

    Args:
        patch_images: List of patch images placed on full-size canvas, with NaN elsewhere

    Returns:
        Averaged image where overlapping regions are averaged
    """
    # Stack images into tensor
    # Shape: (n_patches, H, W) or (n_patches, H, W, C)
    stacked = torch.stack(patch_images, dim=0)

    # Compute average along patch dimension (dim=0), torch.nanmean ignores NaN
    averaged = torch.nanmean(stacked, dim=0)

    # Replace remaining NaN with 0
    # If a pixel is NaN in all patches, nanmean result will also be NaN
    averaged = torch.nan_to_num(averaged, nan=0.0)

    return averaged


def average_overlapping_patches_with_masks(patch_images: List[torch.Tensor], masks: List[torch.Tensor]) -> torch.Tensor:
    """
    Average multiple patch images with explicit masks using torch.nanmean.

    This version is used by federated and multi-kernel algorithms where patch_images
    are the actual observation patches and masks define the valid regions.

    Args:
        patch_images: List of patch image tensors
        masks: List of corresponding mask tensors

    Returns:
        Averaged image where overlapping regions are averaged
    """
    # Create full-size patches with NaN outside mask regions
    full_size_patches = []

    # Get image shape
    if patch_images[0].dim() == 2:  # Grayscale
        img_shape = patch_images[0].shape
        is_color = False
    else:  # Color
        img_shape = patch_images[0].shape[:2]  # HW
        n_channels = patch_images[0].shape[2]
        is_color = True

    for patch, mask in zip(patch_images, masks):
        if is_color:
            # Color image: HWC format
            full_patch = torch.full(
                (*img_shape, n_channels), float("nan"), device=patch.device, dtype=patch.dtype
            )
            for c in range(n_channels):
                # Set mask=0 regions to NaN, keep mask=1 regions as original values
                full_patch[:, :, c] = torch.where(
                    mask > 0, patch[:, :, c], torch.tensor(float("nan"), device=patch.device)
                )
        else:
            # Grayscale image: HW format
            full_patch = torch.full(img_shape, float("nan"), device=patch.device, dtype=patch.dtype)
            full_patch = torch.where(
                mask > 0, patch, torch.tensor(float("nan"), device=patch.device)
            )

        full_size_patches.append(full_patch)

    # Stack images into tensor
    # Shape: (n_patches, H, W) or (n_patches, H, W, C)
    stacked = torch.stack(full_size_patches, dim=0)

    # Compute average along patch dimension (dim=0), torch.nanmean ignores NaN
    averaged = torch.nanmean(stacked, dim=0)

    # Replace remaining NaN with 0
    # If a pixel is NaN in all patches, nanmean result will also be NaN
    averaged = torch.nan_to_num(averaged, nan=0.0)

    return averaged