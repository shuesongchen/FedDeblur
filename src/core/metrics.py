import torch
from torchmetrics.functional import mean_squared_error
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from typing import Dict, Tuple


def calculate_image_metrics(
    restored: torch.Tensor, reference: torch.Tensor, data_range: float = 255.0
) -> Dict[str, float]:
    if restored.device != reference.device:
        restored = restored.to(reference.device)

    if restored.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: restored {restored.shape} vs reference {reference.shape}"
        )

    mse = mean_squared_error(restored, reference).item()

    rmse = torch.sqrt(torch.tensor(mse)).item()

    psnr = peak_signal_noise_ratio(restored, reference, data_range=data_range).item()

    if restored.dim() == 2:
        restored_ssim = restored.unsqueeze(0).unsqueeze(0)
        reference_ssim = reference.unsqueeze(0).unsqueeze(0)
    elif restored.dim() == 3:
        if restored.shape[2] == 3:
            restored_ssim = restored.permute(2, 0, 1).unsqueeze(0)
            reference_ssim = reference.permute(2, 0, 1).unsqueeze(0)
        else:
            restored_ssim = restored.unsqueeze(0)
            reference_ssim = reference.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported tensor dimensions: {restored.dim()}")

    ssim = structural_similarity_index_measure(
        restored_ssim, reference_ssim, data_range=data_range
    ).item()

    return {"mse": mse, "rmse": rmse, "psnr": psnr, "ssim": ssim}


def format_metrics_for_display(metrics: Dict[str, float], iteration: int = None) -> str:
    psnr = metrics.get("psnr", 0)
    ssim = metrics.get("ssim", 0)

    if iteration is not None:
        return f"PSNR={psnr:.2f}dB, SSIM={ssim:.4f} (converged in {iteration} iters)"
    else:
        return f"PSNR={psnr:.2f}dB, SSIM={ssim:.4f}"


def calculate_batch_metrics(
    restored_images: Dict[str, torch.Tensor],
    reference_images: Dict[str, torch.Tensor],
    data_range: float = 255.0,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    if set(restored_images.keys()) != set(reference_images.keys()):
        raise ValueError("Image IDs must match between restored and reference images")

    individual_metrics = {}
    all_metrics = {"mse": [], "rmse": [], "psnr": [], "ssim": []}

    for image_id in sorted(restored_images.keys()):
        restored = restored_images[image_id]
        reference = reference_images[image_id]

        metrics = calculate_image_metrics(restored, reference, data_range)
        individual_metrics[image_id] = metrics

        for metric_name, value in metrics.items():
            all_metrics[metric_name].append(value)

    average_metrics = {}
    for metric_name, values in all_metrics.items():
        average_metrics[metric_name] = sum(values) / len(values)

    return individual_metrics, average_metrics


def print_metrics_summary(
    individual_metrics: Dict[str, Dict[str, float]],
    average_metrics: Dict[str, float],
    iterations: Dict[str, int] = None,
    image_type: str = "images",
) -> None:
    print(f"Processing {image_type}...")

    for image_id in sorted(individual_metrics.keys()):
        metrics = individual_metrics[image_id]
        iter_count = iterations.get(image_id) if iterations else None
        metric_str = format_metrics_for_display(metrics, iter_count)
        print(f"  Image {image_id}: {metric_str}")

    avg_metric_str = format_metrics_for_display(average_metrics)
    print(f"  Average: {avg_metric_str}")
    print()
