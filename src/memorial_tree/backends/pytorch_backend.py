"""
PyTorch Backend module for Memorial Tree.

This module provides the PyTorchBackend class, which implements the BackendInterface
using PyTorch tensors for operations.
"""

from typing import Dict, List, Any, Optional, Union
import numpy as np

from .backend_manager import BackendInterface


class PyTorchBackend(BackendInterface):
    """
    PyTorch implementation of the BackendInterface.

    This class provides tensor operations using PyTorch tensors with support for
    automatic differentiation and GPU acceleration.
    """

    def __init__(self, use_gpu: bool = False):
        """
        Initialize a new PyTorchBackend.

        Args:
            use_gpu (bool): Whether to use GPU acceleration if available.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        try:
            import torch

            self.torch = torch
            self.device = torch.device(
                "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            )
        except ImportError:
            raise ImportError(
                "PyTorch is not installed. Please install it with 'pip install torch'."
            )

    def create_tensor(self, data: List[float]) -> Any:
        """
        Create a PyTorch tensor from the given data.

        Args:
            data (List[float]): The data to create a tensor from.

        Returns:
            torch.Tensor: A PyTorch tensor containing the data.
        """
        return self.torch.tensor(data, dtype=self.torch.float32, device=self.device)

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy array.

        Args:
            tensor (torch.Tensor): The tensor to convert.

        Returns:
            np.ndarray: The tensor as a NumPy array.
        """
        # Check if tensor is already a NumPy array
        if isinstance(tensor, np.ndarray):
            return tensor

        # Ensure tensor is on CPU and detached from computation graph
        return tensor.detach().cpu().numpy()

    def from_numpy(self, array: np.ndarray) -> Any:
        """
        Convert a NumPy array to a PyTorch tensor.

        Args:
            array (np.ndarray): The NumPy array to convert.

        Returns:
            torch.Tensor: The array as a PyTorch tensor.
        """
        tensor = self.torch.from_numpy(array.astype(np.float32))
        return tensor.to(self.device)

    def calculate_weights(
        self, tensors: List[Any], factors: Optional[List[float]] = None
    ) -> Any:
        """
        Calculate weighted values from a list of tensors.

        Args:
            tensors (List[torch.Tensor]): List of tensors to weight.
            factors (Optional[List[float]]): Optional weighting factors.

        Returns:
            torch.Tensor: The weighted result as a PyTorch tensor.
        """
        if not tensors:
            return self.torch.tensor([], dtype=self.torch.float32, device=self.device)

        if factors is None:
            factors = [1.0] * len(tensors)

        if len(tensors) != len(factors):
            raise ValueError("Number of tensors must match number of factors")

        # Convert factors to a tensor
        factors_tensor = self.torch.tensor(
            factors, dtype=self.torch.float32, device=self.device
        )

        # Ensure all tensors are PyTorch tensors on the correct device
        torch_tensors = []
        for tensor in tensors:
            if not isinstance(tensor, self.torch.Tensor):
                tensor = self.from_numpy(self.to_numpy(tensor))
            elif tensor.device != self.device:
                tensor = tensor.to(self.device)
            torch_tensors.append(tensor)

        # Apply weights and sum
        weighted_sum = self.torch.zeros_like(torch_tensors[0])
        for tensor, factor in zip(torch_tensors, factors_tensor):
            weighted_sum += tensor * factor

        return weighted_sum

    def apply_softmax(self, tensor: Any, temperature: float = 1.0) -> Any:
        """
        Apply softmax function to a tensor.

        Args:
            tensor (torch.Tensor): The input tensor.
            temperature (float): Temperature parameter for softmax.

        Returns:
            torch.Tensor: The result of applying softmax.
        """
        # Ensure tensor is a PyTorch tensor
        if not isinstance(tensor, self.torch.Tensor):
            tensor = self.from_numpy(self.to_numpy(tensor))

        # Apply temperature scaling
        scaled = tensor / max(temperature, 1e-8)  # Avoid division by zero

        # Use PyTorch's built-in softmax function
        return self.torch.nn.functional.softmax(scaled, dim=0)

    def get_backend_name(self) -> str:
        """
        Get the name of this backend.

        Returns:
            str: The backend name.
        """
        return "pytorch"

    def enable_grad(self) -> None:
        """
        Enable gradient computation for automatic differentiation.
        """
        self.torch.set_grad_enabled(True)

    def disable_grad(self) -> None:
        """
        Disable gradient computation to save memory and computation.
        """
        self.torch.set_grad_enabled(False)

    def use_gpu(self, enable: bool = True) -> bool:
        """
        Enable or disable GPU usage if available.

        Args:
            enable (bool): Whether to enable GPU usage.

        Returns:
            bool: True if GPU is now being used, False otherwise.
        """
        if enable and self.torch.cuda.is_available():
            self.device = self.torch.device("cuda")
            return True
        else:
            self.device = self.torch.device("cpu")
            return False
