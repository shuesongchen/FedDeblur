from abc import ABC, abstractmethod
import torch
from typing import Dict, Any


class BaseRegularizer(ABC):
    def __init__(self, params: Dict[str, Any], device):
        self.params = params
        self.device = device

        if "rho" not in params:
            raise ValueError("Required 'rho' non found")

    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    def validate_parameters(self):
        pass

    def get_parameter(self, name: str, default=None):
        return self.params.get(name, default)

    def update_parameter(self, name: str, value):
        self.params[name] = value
