"""
Memorial Tree - A Python package for modeling human thought processes and decision-making.

This package provides tools for computational psychiatry research by representing
both conscious choices and unconscious influences in human cognitive processes.
"""

__version__ = "0.1.0"
__author__ = "Memorial Tree Team"
__email__ = "info@memorialtree.org"

# Import main classes for easier access
from .core.thought_node import ThoughtNode
from .core.ghost_node import GhostNode
from .core.memorial_tree import MemorialTree

# Backend classes
from .backends.backend_manager import BackendManager, BackendInterface
from .backends.numpy_backend import NumpyBackend
from .backends.pytorch_backend import PyTorchBackend
from .backends.tensorflow_backend import TensorFlowBackend

# Model classes
from .models.adhd_model import ADHDModel
from .models.depression_model import DepressionModel
from .models.anxiety_model import AnxietyModel
from .models.model_comparison import ModelComparison

# Visualization tools
from .visualization.tree_visualizer import TreeVisualizer
from .visualization.path_analyzer import PathAnalyzer
from .visualization.statistical_analyzer import StatisticalAnalyzer
from .visualization.model_visualizer import ModelVisualizer

__all__ = [
    # Core
    "ThoughtNode",
    "GhostNode",
    "MemorialTree",
    # Backends
    "BackendManager",
    "BackendInterface",
    "NumpyBackend",
    "PyTorchBackend",
    "TensorFlowBackend",
    # Models
    "ADHDModel",
    "DepressionModel",
    "AnxietyModel",
    "ModelComparison",
    # Visualization
    "TreeVisualizer",
    "PathAnalyzer",
    "StatisticalAnalyzer",
    "ModelVisualizer",
]
