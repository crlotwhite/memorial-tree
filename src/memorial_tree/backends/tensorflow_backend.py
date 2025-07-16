"""
TensorFlow Backend module for Memorial Tree.

This module provides the TensorFlowBackend class, which implements the BackendInterface
using TensorFlow tensors for operations with Keras compatibility.
"""

from typing import Dict, List, Any, Optional, Union
import numpy as np

from .backend_manager import BackendInterface


class TensorFlowBackend(BackendInterface):
    """
    TensorFlow implementation of the BackendInterface.

    This class provides tensor operations using TensorFlow tensors with support for
    Keras compatibility and graph mode execution.
    """

    def __init__(self, use_gpu: bool = False, eager_mode: bool = True):
        """
        Initialize a new TensorFlowBackend.

        Args:
            use_gpu (bool): Whether to use GPU acceleration if available.
            eager_mode (bool): Whether to use eager execution (True) or graph mode (False).

        Raises:
            ImportError: If TensorFlow is not installed.
        """
        try:
            import tensorflow as tf

            self.tf = tf

            # Configure GPU usage
            if not use_gpu:
                # Limit TensorFlow to CPU only
                self.tf.config.set_visible_devices([], "GPU")

            # Configure execution mode
            self.tf.config.run_functions_eagerly(eager_mode)
            self.eager_mode = eager_mode

        except ImportError:
            raise ImportError(
                "TensorFlow is not installed. Please install it with 'pip install tensorflow'."
            )

    def create_tensor(self, data: List[float]) -> Any:
        """
        Create a TensorFlow tensor from the given data.

        Args:
            data (List[float]): The data to create a tensor from.

        Returns:
            tf.Tensor: A TensorFlow tensor containing the data.
        """
        return self.tf.convert_to_tensor(data, dtype=self.tf.float32)

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """
        Convert a TensorFlow tensor to a NumPy array.

        Args:
            tensor (tf.Tensor): The tensor to convert.

        Returns:
            np.ndarray: The tensor as a NumPy array.
        """
        # Check if tensor is already a NumPy array
        if isinstance(tensor, np.ndarray):
            return tensor

        # Convert TensorFlow tensor to NumPy
        return tensor.numpy()

    def from_numpy(self, array: np.ndarray) -> Any:
        """
        Convert a NumPy array to a TensorFlow tensor.

        Args:
            array (np.ndarray): The NumPy array to convert.

        Returns:
            tf.Tensor: The array as a TensorFlow tensor.
        """
        return self.tf.convert_to_tensor(array, dtype=self.tf.float32)

    def calculate_weights(
        self, tensors: List[Any], factors: Optional[List[float]] = None
    ) -> Any:
        """
        Calculate weighted values from a list of tensors.

        Args:
            tensors (List[tf.Tensor]): List of tensors to weight.
            factors (Optional[List[float]]): Optional weighting factors.

        Returns:
            tf.Tensor: The weighted result as a TensorFlow tensor.
        """
        if not tensors:
            return self.tf.constant([], dtype=self.tf.float32)

        if factors is None:
            factors = [1.0] * len(tensors)

        if len(tensors) != len(factors):
            raise ValueError("Number of tensors must match number of factors")

        # Convert factors to a tensor
        factors_tensor = self.tf.constant(factors, dtype=self.tf.float32)

        # Ensure all tensors are TensorFlow tensors
        tf_tensors = []
        for tensor in tensors:
            if not isinstance(tensor, self.tf.Tensor):
                tensor = self.from_numpy(self.to_numpy(tensor))
            tf_tensors.append(tensor)

        # Apply weights and sum
        weighted_sum = self.tf.zeros_like(tf_tensors[0])
        for tensor, factor in zip(tf_tensors, factors_tensor):
            weighted_sum += tensor * factor

        return weighted_sum

    def apply_softmax(self, tensor: Any, temperature: float = 1.0) -> Any:
        """
        Apply softmax function to a tensor.

        This implementation is optimized for performance using TensorFlow's XLA compilation
        when available.

        Args:
            tensor (tf.Tensor): The input tensor.
            temperature (float): Temperature parameter for softmax.

        Returns:
            tf.Tensor: The result of applying softmax.
        """
        # Ensure tensor is a TensorFlow tensor
        if not isinstance(tensor, self.tf.Tensor):
            tensor = self.from_numpy(self.to_numpy(tensor))

        # Use a function that can be XLA-compiled for better performance
        @self.tf_function
        def optimized_softmax(x, temp):
            # Apply temperature scaling with efficient broadcasting
            scaled = x / self.tf.maximum(
                self.tf.constant(temp, dtype=self.tf.float32),
                self.tf.constant(1e-8, dtype=self.tf.float32),
            )

            # Use TensorFlow's built-in softmax function which is already optimized
            return self.tf.nn.softmax(scaled)

        # Apply the optimized function
        return optimized_softmax(tensor, temperature)

    def get_backend_name(self) -> str:
        """
        Get the name of this backend.

        Returns:
            str: The backend name.
        """
        return "tensorflow"

    def set_eager_mode(self, enable: bool = True) -> None:
        """
        Enable or disable eager execution mode.

        Args:
            enable (bool): Whether to enable eager execution.
        """
        self.tf.config.run_functions_eagerly(enable)
        self.eager_mode = enable

    def is_eager_mode(self) -> bool:
        """
        Check if eager execution mode is enabled.

        Returns:
            bool: True if eager execution is enabled, False otherwise.
        """
        return self.eager_mode

    def use_gpu(self, enable: bool = True) -> bool:
        """
        Enable or disable GPU usage if available.

        Args:
            enable (bool): Whether to enable GPU usage.

        Returns:
            bool: True if GPU is now being used, False otherwise.
        """
        gpus = self.tf.config.list_physical_devices("GPU")

        if enable and gpus:
            # Enable GPU
            for gpu in gpus:
                self.tf.config.experimental.set_memory_growth(gpu, True)
            return True
        else:
            # Disable GPU
            self.tf.config.set_visible_devices([], "GPU")
            return False

    def get_keras_layer(self, layer_type: str, **kwargs) -> Any:
        """
        Create a Keras layer for use with the TensorFlow backend.

        Args:
            layer_type (str): Type of layer to create (e.g., 'Dense', 'Conv2D').
            **kwargs: Additional arguments to pass to the layer constructor.

        Returns:
            tf.keras.layers.Layer: A Keras layer instance.

        Raises:
            ValueError: If the layer type is not supported.
        """
        try:
            layer_class = getattr(self.tf.keras.layers, layer_type)
            return layer_class(**kwargs)
        except AttributeError:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    @staticmethod
    def tf_function(func):
        """
        Decorator to convert a Python function into a TensorFlow Function.

        This enables graph execution for better performance when not in eager mode.

        Args:
            func: The function to convert.

        Returns:
            A callable TensorFlow Function.
        """
        import tensorflow as tf

        return tf.function(func)
