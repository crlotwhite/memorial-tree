"""
Core module for Memorial Tree.

This module contains the fundamental data structures and algorithms:
- ThoughtNode: Base class for representing thoughts and decisions
- GhostNode: Extension of ThoughtNode for unconscious influences
- MemorialTree: Main tree structure for modeling thought processes
"""

from .thought_node import ThoughtNode
from .ghost_node import GhostNode
from .memorial_tree import MemorialTree

__all__ = ["ThoughtNode", "GhostNode", "MemorialTree"]
