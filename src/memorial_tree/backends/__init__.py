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

__all__ = ["BackendManager", "BackendInterface", "NumpyBackend"]
