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
from .backends.backend_manager import BackendManager, BackendInterface

# These will be implemented in future tasks
# from .core.memorial_tree import MemorialTree

__all__ = ["ThoughtNode", "GhostNode", "BackendManager", "BackendInterface"]
