"""
Backend Manager module for Memorial Tree.

This module provides the BackendManager class, which manages different numerical
computation backends (NumPy, PyTorch, TensorFlow) for the Memorial Tree package.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np


class BackendInterface:
    """
    Interface defining common tensor operations for all backends.

    This abstract class defines the operations that must be implemented
    by all backend implementations.
    """

    def create_tensor(self, data: List[float]) -> Any:
        """
        Create a tensor from the given data.

        Args:
            data (List[float]): The data to create a tensor from.

        Returns:
            Any: A tensor object in the backend's format.
        """
        raise NotImplementedError("Backend must implement create_tensor")

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """
        Convert a tensor to a NumPy array.

        Args:
            tensor (Any): The tensor to convert.

        Returns:
            np.ndarray: The tensor as a NumPy array.
        """
        raise NotImplementedError("Backend must implement to_numpy")

    def from_numpy(self, array: np.ndarray) -> Any:
        """
        Convert a NumPy array to a tensor.

        Args:
            array (np.ndarray): The NumPy array to convert.

        Returns:
            Any: The array as a tensor in the backend's format.
        """
        raise NotImplementedError("Backend must implement from_numpy")

    def calculate_weights(
        self, tensors: List[Any], factors: Optional[List[float]] = None
    ) -> Any:
        """
        Calculate weighted values from a list of tensors.

        Args:
            tensors (List[Any]): List of tensors to weight.
            factors (Optional[List[float]]): Optional weighting factors.

        Returns:
            Any: The weighted result as a tensor.
        """
        raise NotImplementedError("Backend must implement calculate_weights")

    def apply_softmax(self, tensor: Any, temperature: float = 1.0) -> Any:
        """
        Apply softmax function to a tensor.

        Args:
            tensor (Any): The input tensor.
            temperature (float): Temperature parameter for softmax.

        Returns:
            Any: The result of applying softmax.
        """
        raise NotImplementedError("Backend must implement apply_softmax")

    def get_backend_name(self) -> str:
        """
        Get the name of this backend.

        Returns:
            str: The backend name.
        """
        raise NotImplementedError("Backend must implement get_backend_name")


class BackendManager:
    """
    Manager for different numerical computation backends.

    BackendManager provides a unified interface for working with different
    tensor computation libraries (NumPy, PyTorch, TensorFlow) and handles
    switching between them.

    Attributes:
        backend_type (str): The type of backend currently in use.
        backend (BackendInterface): The current backend implementation.
        _backend_cache (Dict[str, BackendInterface]): Cache of initialized backends.
    """

    def __init__(self, backend_type: str = "numpy"):
        """
        Initialize a new BackendManager.

        Args:
            backend_type (str): The type of backend to use.
                               Options: "numpy", "pytorch", "tensorflow".

        Raises:
            ValueError: If an unsupported backend type is specified.
        """
        self.backend_type = backend_type.lower()
        # Initialize backend cache for faster switching
        self._backend_cache = {}
        self.backend = self._initialize_backend()

    def _initialize_backend(self) -> BackendInterface:
        """
        Initialize the specified backend.

        Returns:
            BackendInterface: An instance of the specified backend.

        Raises:
            ValueError: If an unsupported backend type is specified.
        """
        # Check if backend is already in cache
        if self.backend_type in self._backend_cache:
            return self._backend_cache[self.backend_type]

        # Initialize new backend
        if self.backend_type == "numpy":
            # Import here to avoid circular imports
            from .numpy_backend import NumpyBackend

            backend = NumpyBackend()
        elif self.backend_type == "pytorch":
            # Import here to avoid circular imports
            from .pytorch_backend import PyTorchBackend

            backend = PyTorchBackend()
        elif self.backend_type == "tensorflow":
            # Import here to avoid circular imports
            from .tensorflow_backend import TensorFlowBackend

            backend = TensorFlowBackend()
        else:
            raise ValueError(f"Unsupported backend type: {self.backend_type}")

        # Cache the backend for future use
        self._backend_cache[self.backend_type] = backend
        return backend

    def switch_backend(self, new_backend_type: str) -> None:
        """
        Switch to a different backend type.

        Args:
            new_backend_type (str): The type of backend to switch to.
                                   Options: "numpy", "pytorch", "tensorflow".

        Raises:
            ValueError: If an unsupported backend type is specified.
        """
        new_backend_type = new_backend_type.lower()
        if new_backend_type == self.backend_type:
            return  # Already using this backend

        self.backend_type = new_backend_type
        self.backend = self._initialize_backend()

    def create_tensor(self, data: List[float]) -> Any:
        """
        Create a tensor from the given data using the current backend.

        Args:
            data (List[float]): The data to create a tensor from.

        Returns:
            Any: A tensor object in the current backend's format.
        """
        return self.backend.create_tensor(data)

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """
        Convert a tensor to a NumPy array using the current backend.

        Args:
            tensor (Any): The tensor to convert.

        Returns:
            np.ndarray: The tensor as a NumPy array.
        """
        return self.backend.to_numpy(tensor)

    def from_numpy(self, array: np.ndarray) -> Any:
        """
        Convert a NumPy array to a tensor using the current backend.

        Args:
            array (np.ndarray): The NumPy array to convert.

        Returns:
            Any: The array as a tensor in the current backend's format.
        """
        return self.backend.from_numpy(array)

    def calculate_weights(
        self, tensors: List[Any], factors: Optional[List[float]] = None
    ) -> Any:
        """
        Calculate weighted values from a list of tensors using the current backend.

        Args:
            tensors (List[Any]): List of tensors to weight.
            factors (Optional[List[float]]): Optional weighting factors.

        Returns:
            Any: The weighted result as a tensor.
        """
        return self.backend.calculate_weights(tensors, factors)

    def apply_softmax(self, tensor: Any, temperature: float = 1.0) -> Any:
        """
        Apply softmax function to a tensor using the current backend.

        Args:
            tensor (Any): The input tensor.
            temperature (float): Temperature parameter for softmax.

        Returns:
            Any: The result of applying softmax.
        """
        return self.backend.apply_softmax(tensor, temperature)

    def get_backend_name(self) -> str:
        """
        Get the name of the current backend.

        Returns:
            str: The current backend name.
        """
        return self.backend.get_backend_name()
