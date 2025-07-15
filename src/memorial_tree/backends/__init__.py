"""
Backends module for Memorial Tree.

This module provides abstraction for different numerical computation libraries:
- BackendManager: Factory and manager for different backends
- NumpyBackend: NumPy implementation
- PyTorchBackend: PyTorch implementation
- TensorFlowBackend: TensorFlow/Keras implementation
"""

from .backend_manager import BackendManager, BackendInterface
from .numpy_backend import NumpyBackend
from .pytorch_backend import PyTorchBackend
from .tensorflow_backend import TensorFlowBackend

__all__ = [
    "BackendManager",
    "BackendInterface",
    "NumpyBackend",
    "PyTorchBackend",
    "TensorFlowBackend",
]
