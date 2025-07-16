"""
NumPy Backend module for Memorial Tree.

This module provides the NumpyBackend class, which implements the BackendInterface
using NumPy arrays for tensor operations.
"""

from typing import List, Any, Optional
import numpy as np

from .backend_manager import BackendInterface


class NumpyBackend(BackendInterface):
    """
    NumPy implementation of the BackendInterface.

    This class provides tensor operations using NumPy arrays.
    """

    def create_tensor(self, data: List[float]) -> np.ndarray:
        """
        Create a NumPy array from the given data.

        Args:
            data (List[float]): The data to create an array from.

        Returns:
            np.ndarray: A NumPy array containing the data.
        """
        return np.array(data, dtype=np.float32)

    def to_numpy(self, tensor: np.ndarray) -> np.ndarray:
        """
        Convert a tensor to a NumPy array (no-op for NumPy backend).

        Args:
            tensor (np.ndarray): The tensor to convert.

        Returns:
            np.ndarray: The tensor as a NumPy array.
        """
        return tensor  # Already a NumPy array

    def from_numpy(self, array: np.ndarray) -> np.ndarray:
        """
        Convert a NumPy array to a tensor (no-op for NumPy backend).

        Args:
            array (np.ndarray): The NumPy array to convert.

        Returns:
            np.ndarray: The array as a NumPy array.
        """
        return array  # Already a NumPy array

    def calculate_weights(
        self, tensors: List[np.ndarray], factors: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Calculate weighted values from a list of tensors.

        Args:
            tensors (List[np.ndarray]): List of tensors to weight.
            factors (Optional[List[float]]): Optional weighting factors.

        Returns:
            np.ndarray: The weighted result as a NumPy array.
        """
        if not tensors:
            return np.array([], dtype=np.float32)

        if factors is None:
            factors = [1.0] * len(tensors)

        if len(tensors) != len(factors):
            raise ValueError("Number of tensors must match number of factors")

        # Convert all tensors to numpy arrays if they aren't already
        numpy_tensors = [self.to_numpy(t) for t in tensors]

        # Apply weights and sum
        weighted_sum = np.zeros_like(numpy_tensors[0])
        for tensor, factor in zip(numpy_tensors, factors):
            weighted_sum += tensor * factor

        return weighted_sum

    def apply_softmax(self, tensor: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Apply softmax function to a tensor.

        This implementation is optimized for numerical stability and performance.

        Args:
            tensor (np.ndarray): The input tensor.
            temperature (float): Temperature parameter for softmax.

        Returns:
            np.ndarray: The result of applying softmax.
        """
        # Apply temperature scaling with vectorized operations
        # Use fmax for element-wise maximum to avoid division by zero
        scaled = tensor / np.fmax(temperature, 1e-8)

        # Subtract max for numerical stability (prevents overflow)
        # Use keepdims to ensure proper broadcasting
        max_val = np.max(scaled, keepdims=True)
        exp_values = np.exp(scaled - max_val)

        # Normalize with sum keepdims for proper broadcasting
        sum_val = np.sum(exp_values, keepdims=True)
        return exp_values / sum_val

    def get_backend_name(self) -> str:
        """
        Get the name of this backend.

        Returns:
            str: The backend name.
        """
        return "numpy"
