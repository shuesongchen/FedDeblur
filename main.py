"""
Federated Image Deblurring (FedDeblur) Framework.

Usage:
    python main.py --algorithm CenDeblur --regularizer TV --observation complete --n_clients 5 --noise_std 1.0 --eta 0.03 --rho 0.001

    python main.py --algorithm FedDeblur --regularizer PnP --observation complete --n_clients 3 --noise_std 2.0 --sigma 30.0 --rho 0.01
"""

import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.core.parameter_manager import ParameterManager
from src.core.data_loader import DataLoader
from src.core.metrics import print_metrics_summary
from src.core.metrics import format_metrics_for_display
from src.regularizers import create_regularizer
from src.algorithms import create_algorithm
from src.utils.math_ops import setup_all


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FedDeblur Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --algorithm FedDeblur --regularizer PnP --observation complete --n_clients 3 --noise_std 2.0 --sigma 30.0 --rho 0.01
  
  python main.py --algorithm FedDeblur --regularizer PnP --observation partial --noise_std 1.0 --blur_type synthesis --sigma 25.0

Supported Algorithms: FedDeblur, CenDeblur, FedAvgDeblur, LocDeblur
Supported Regularizers: TV, PnP
Supported Observations: complete, partial
Supported Blur Types: motion, synthesis

Parameter Override:
  --eta: TV regularization parameter (for TV regularizer)
  --rho: ADMM penalty parameter (for both TV and PnP)
  --sigma: PnP denoising strength parameter (for PnP regularizer)
        """,
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="FedDeblur",
        choices=["FedDeblur", "CenDeblur", "FedAvgDeblur", "LocDeblur"],
        help="Deblurring algorithm to use",
    )

    parser.add_argument(
        "--regularizer",
        type=str,
        default="TV",
        choices=["TV", "PnP"],
        help="Regularization method",
    )

    parser.add_argument(
        "--observation",
        type=str,
        default="complete",
        choices=["complete", "partial"],
        help="Observation type",
    )

    parser.add_argument(
        "--noise_std",
        type=float,
        default=1.0,
        help="Noise standard deviation (0.0, 0.5, 1.0, 2.0, 3.0)",
    )

    parser.add_argument(
        "--n_clients",
        type=int,
        default=10,
        help="Number of clients (for complete observation, default: 10)",
    )

    parser.add_argument(
        "--blur_type",
        type=str,
        default="motion",
        choices=["motion", "synthesis"],
        help="Blur type ('motion', 'synthesis')",
    )

    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU ID to use (default: 0)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="TV regularization parameter",
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="ADMM penalty parameter",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="PnP denoising strength parameter",
    )

    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    # Validate parameter overrides
    if args.eta is not None and args.eta <= 0:
        print("Error: --eta must be positive")
        return False

    if args.sigma is not None and args.sigma <= 0:
        print("Error: --sigma must be positive")
        return False

    if args.rho is not None and args.rho <= 0:
        print("Error: --rho must be positive")
        return False

    # Validate noise_std
    if args.noise_std < 0:
        print("Error: --noise_std must be non-negative")
        return False

    return True


def create_config(args):
    """Create configuration dictionary from arguments."""
    config = {
        "algorithm": args.algorithm,
        "regularizer": args.regularizer,
        "observation": args.observation,
        "blur_type": args.blur_type,
        "n_clients": args.n_clients,
        "noise_std": args.noise_std,
    }

    # Add parameter overrides if provided
    parameter_overrides = {}
    if args.eta is not None:
        parameter_overrides["eta"] = args.eta
    if args.rho is not None:
        parameter_overrides["rho"] = args.rho
    if args.sigma is not None:
        parameter_overrides["sigma"] = args.sigma

    if parameter_overrides:
        config["parameter_overrides"] = parameter_overrides

    return config


def process_image_type(
    image_type: str,
    config: dict,
    param_manager: ParameterManager,
    data_loader: DataLoader,
    args,
) -> bool:
    is_color = image_type == "color"

    if args.regularizer == "PnP" and is_color:
        return True

    try:
        # Get parameters for this configuration
        params = param_manager.get_params(config, is_color)

        # Create regularizer and algorithm
        regularizer = create_regularizer(args.regularizer, params, device)
        algorithm = create_algorithm(args.algorithm, params, regularizer, device)

        # Get available images
        image_list = data_loader.get_image_list(config, image_type)
        if not image_list:
            print("No images found for this configuration")
            return True

        # Display parameters being used (including overrides)
        param_info = []
        if args.regularizer == "TV":
            param_info.append(f"eta={params.get('eta', 'N/A')}")
        elif args.regularizer == "PnP":
            param_info.append(f"sigma={params.get('sigma', 'N/A')}")
        param_info.append(f"rho={params.get('rho', 'N/A')}")
        param_str = ", ".join(param_info)

        print(
            f"\nProcessing {image_type} images ({len(image_list)} images) - {param_str}..."
        )

        # Process each image
        individual_metrics = {}
        iterations = {}

        for image_id in image_list:
            try:
                # Load image data
                data = data_loader.load_single_image_data(image_id, image_type, config)

                if (
                    args.algorithm == "FedDeblur"
                    and args.regularizer == "PnP"
                    and config["observation"] == "partial"
                ):
                    fresh_regularizer = create_regularizer(
                        args.regularizer, params, device
                    )
                    algorithm.regularizer = fresh_regularizer

                # Run algorithm
                _, iter_count, metrics = algorithm.run(data)

                # Store results
                if metrics:
                    individual_metrics[image_id] = metrics
                    iterations[image_id] = iter_count

                    metric_str = format_metrics_for_display(metrics, iter_count)
                    print(f"  Image {image_id}: {metric_str}")

            except Exception as e:
                print(f"  Image {image_id}: Error - {str(e)}")
                continue

        # Print summary if we have metrics
        if individual_metrics:
            # Calculate average metrics
            all_metrics = {"mse": [], "rmse": [], "psnr": [], "ssim": []}
            for metrics in individual_metrics.values():
                for metric_name, value in metrics.items():
                    all_metrics[metric_name].append(value)

            average_metrics = {}
            for metric_name, values in all_metrics.items():
                average_metrics[metric_name] = sum(values) / len(values)

            # Print average
            from src.core.metrics import format_metrics_for_display

            avg_metric_str = format_metrics_for_display(average_metrics)
            print(f"  Average: {avg_metric_str}")

        return True

    except Exception as e:
        print(f"Error processing {image_type} images: {str(e)}")
        return False


def main():
    """Main execution function."""
    # Parse and validate arguments
    args = parse_arguments()
    if not validate_arguments(args):
        sys.exit(1)

    # Setup device and reproducibility
    global device
    device = setup_all(seed=args.seed, gpu_id=args.gpu_id)

    # Create configuration
    config = create_config(args)

    # Initialize managers
    param_manager = ParameterManager()
    data_loader = DataLoader(device=device)

    # Print configuration
    config_str = f"{args.algorithm}-{args.regularizer}-{args.observation.title()}, noise_std={args.noise_std}"

    # Add parameter override information
    overrides = []
    if args.eta is not None:
        overrides.append(f"eta={args.eta}")
    if args.rho is not None:
        overrides.append(f"rho={args.rho}")
    if args.sigma is not None:
        overrides.append(f"sigma={args.sigma}")

    if overrides:
        config_str += f" (Overrides: {', '.join(overrides)})"

    print(config_str)
    print("=" * 60)

    # Validate data availability
    try:
        _ = data_loader.validate_data_availability(config)
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("\nPlease ensure the required data has been generated.")
        sys.exit(1)

    # Process image types based on arguments
    success = True

    # Always process grayscale images first
    success = (
        process_image_type("grayscale", config, param_manager, data_loader, args)
        and success
    )

    success = (
        process_image_type("color", config, param_manager, data_loader, args)
        and success
    )

    if success:
        print("\nProcessing completed successfully!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
