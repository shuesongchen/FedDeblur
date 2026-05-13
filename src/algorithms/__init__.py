"""
Algorithms module.

Available algorithms:
- FedDeblur: Federated deblurring with consensus constraints
- CenDeblur: Centralized deblurring
- FedAvgDeblur: Federated averaging approach
- LocDeblur: Localized deblurring
"""

from .base_algorithm import BaseAlgorithm
from .fed_deblur import FedDeblur
from .cen_deblur import CenDeblur
from .fedavg_deblur import FedAvgDeblur
from .loc_deblur import LocDeblur

__all__ = ["BaseAlgorithm", "FedDeblur", "CenDeblur", "FedAvgDeblur", "LocDeblur"]


def create_algorithm(algorithm_type: str, params: dict, regularizer, device):
    """
    Factory function to create algorithm instances.

    Args:
        algorithm_type: Type of algorithm ("FedDeblur", "CenDeblur", "FedAvgDeblur", "LocDeblur")
        params: Parameter dictionary for the algorithm
        regularizer: Regularizer instance (TV or PnP)
        device: PyTorch device

    Returns:
        Algorithm instance

    Raises:
        ValueError: For unsupported algorithm types
    """
    if algorithm_type == "FedDeblur":
        return FedDeblur(params, regularizer, device)
    elif algorithm_type == "CenDeblur":
        return CenDeblur(params, regularizer, device)
    elif algorithm_type == "FedAvgDeblur":
        return FedAvgDeblur(params, regularizer, device)
    elif algorithm_type == "LocDeblur":
        return LocDeblur(params, regularizer, device)
    else:
        raise ValueError(f"Unsupported algorithm type: {algorithm_type}.")
