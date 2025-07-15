"""
MemorialTree module for Memorial Tree.

This module provides the MemorialTree class, which is the main class for modeling
thought processes and decision-making using a tree structure.
"""

from typing import List, Dict, Any, Optional, Union, Set
from datetime import datetime
import uuid

from .thought_node import ThoughtNode
from .ghost_node import GhostNode
from ..backends.backend_manager import BackendManager


class NodeNotFoundError(Exception):
    """Exception raised when a node cannot be found in the tree."""

    pass


class MemorialTree:
    """
    Main class for modeling thought processes and decision-making using a tree structure.

    MemorialTree manages the creation and manipulation of thought nodes, tracks the current
    state of the decision process, and provides methods for traversing and analyzing the tree.

    Attributes:
        root (ThoughtNode): The root node of the tree.
        backend_manager (BackendManager): Manager for numerical computation backend.
        ghost_nodes (List[GhostNode]): List of ghost nodes that can influence decisions.
        current_node (ThoughtNode): The current active node in the decision process.
        path_history (List[str]): History of node IDs in the current decision path.
        node_registry (Dict[str, ThoughtNode]): Registry of all nodes by ID for quick lookup.
    """

    def __init__(self, root_content: str = "Root", backend: str = "numpy"):
        """
        Initialize a new MemorialTree.

        Args:
            root_content (str): Content for the root node.
            backend (str): Backend type to use for numerical computations.
                          Options: "numpy", "pytorch", "tensorflow".
        """
        # Create root node
        self.root = ThoughtNode(root_content)

        # Initialize backend manager
        self.backend_manager = BackendManager(backend)

        # Initialize ghost nodes list
        self.ghost_nodes: List[GhostNode] = []

        # Set current node to root
        self.current_node = self.root

        # Initialize path history with root node ID
        self.path_history: List[str] = [self.root.node_id]

        # Initialize node registry with root node
        self.node_registry: Dict[str, ThoughtNode] = {self.root.node_id: self.root}

        # Initialize creation timestamp
        self.creation_time = datetime.now()

        # Initialize metadata
        self.metadata: Dict[str, Any] = {}

    def add_thought(
        self,
        parent_id: str,
        content: str,
        weight: float = 1.0,
        node_id: Optional[str] = None,
    ) -> ThoughtNode:
        """
        Add a new thought node to the tree.

        Args:
            parent_id (str): ID of the parent node.
            content (str): Content of the new thought.
            weight (float): Weight of the new thought (default: 1.0).
            node_id (Optional[str]): Optional ID for the new node.

        Returns:
            ThoughtNode: The newly created thought node.

        Raises:
            NodeNotFoundError: If the parent node cannot be found.
        """
        # Find parent node
        parent = self._get_node_by_id(parent_id)
        if not parent:
            raise NodeNotFoundError(f"Parent node with ID {parent_id} not found")

        # Create new thought node
        new_node = ThoughtNode(content, node_id=node_id, weight=weight)

        # Add as child to parent
        parent.add_child(new_node)

        # Add to registry
        self.node_registry[new_node.node_id] = new_node

        return new_node

    def add_ghost_node(
        self,
        content: str,
        influence: float = 0.3,
        visibility: float = 0.1,
        weight: float = 1.0,
        node_id: Optional[str] = None,
    ) -> GhostNode:
        """
        Add a new ghost node to the tree.

        Args:
            content (str): Content of the ghost node.
            influence (float): Influence strength (0-1).
            visibility (float): Visibility/consciousness level (0-1).
            weight (float): Base weight of the node.
            node_id (Optional[str]): Optional ID for the new node.

        Returns:
            GhostNode: The newly created ghost node.
        """
        # Create new ghost node
        ghost_node = GhostNode(
            content,
            node_id=node_id,
            influence=influence,
            visibility=visibility,
            weight=weight,
        )

        # Add to ghost nodes list
        self.ghost_nodes.append(ghost_node)

        # Add to registry
        self.node_registry[ghost_node.node_id] = ghost_node

        return ghost_node

    def make_choice(self, node_id: str) -> ThoughtNode:
        """
        Make a choice by selecting a node as the next step in the decision process.

        Args:
            node_id (str): ID of the node to select.

        Returns:
            ThoughtNode: The selected node.

        Raises:
            NodeNotFoundError: If the node cannot be found.
            ValueError: If the node is not a valid choice (not a child of current node).
        """
        # Find the node
        node = self._get_node_by_id(node_id)
        if not node:
            raise NodeNotFoundError(f"Node with ID {node_id} not found")

        # Check if node is a valid choice (child of current node)
        if node.parent != self.current_node:
            raise ValueError(f"Node {node_id} is not a valid choice from current node")

        # Update current node
        self.current_node = node

        # Activate the node
        node.activate()

        # Update path history
        self.path_history.append(node.node_id)

        return node

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of the decision process.

        Returns:
            Dict[str, Any]: Dictionary containing the current state.
        """
        return {
            "current_node": self.current_node,
            "current_node_id": self.current_node.node_id,
            "current_content": self.current_node.content,
            "path_history": self.path_history.copy(),
            "available_choices": [child for child in self.current_node.children],
            "ghost_nodes": self.ghost_nodes.copy(),
            "active_ghost_nodes": self._get_active_ghost_nodes(),
            "tree_depth": len(self.path_history) - 1,  # Root is at depth 0
            "timestamp": datetime.now(),
        }

    def _get_active_ghost_nodes(self) -> List[GhostNode]:
        """
        Get the currently active ghost nodes based on the current context.

        Returns:
            List[GhostNode]: List of active ghost nodes.
        """
        # Create current context
        context = {
            "current_node": self.current_node,
            "path_history": self.path_history,
            "timestamp": datetime.now(),
        }

        # Check each ghost node for activation
        active_nodes = []
        for ghost_node in self.ghost_nodes:
            if ghost_node.check_activation(context):
                ghost_node.activate()
                active_nodes.append(ghost_node)

        return active_nodes

    def get_available_choices(self) -> List[ThoughtNode]:
        """
        Get the available choices from the current node.

        Returns:
            List[ThoughtNode]: List of available child nodes.
        """
        return self.current_node.children.copy()

    def get_path_from_root(self) -> List[ThoughtNode]:
        """
        Get the path from root to the current node.

        Returns:
            List[ThoughtNode]: List of nodes from root to current node.
        """
        # Reverse the current node's path to root
        path = self.current_node.get_path_to_root()
        path.reverse()
        return path

    def reset_to_root(self) -> None:
        """
        Reset the current state to the root node.
        """
        self.current_node = self.root
        self.path_history = [self.root.node_id]

    def _get_node_by_id(self, node_id: str) -> Optional[ThoughtNode]:
        """
        Get a node by its ID.

        Args:
            node_id (str): ID of the node to find.

        Returns:
            Optional[ThoughtNode]: The found node, or None if not found.
        """
        return self.node_registry.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the tree.

        Args:
            node_id (str): ID of the node to remove.

        Returns:
            bool: True if the node was removed, False otherwise.

        Raises:
            ValueError: If attempting to remove the root node or current node.
        """
        # Find the node
        node = self._get_node_by_id(node_id)
        if not node:
            return False

        # Check if it's the root or current node
        if node == self.root:
            raise ValueError("Cannot remove the root node")
        if node == self.current_node:
            raise ValueError("Cannot remove the current node")

        # Get parent
        parent = node.parent
        if not parent:
            return False

        # Remove from parent's children
        parent.children.remove(node)

        # Remove from registry (including all descendants)
        self._remove_node_and_descendants_from_registry(node)

        # If it's a ghost node, remove from ghost nodes list
        if isinstance(node, GhostNode) and node in self.ghost_nodes:
            self.ghost_nodes.remove(node)

        return True

    def _remove_node_and_descendants_from_registry(self, node: ThoughtNode) -> None:
        """
        Remove a node and all its descendants from the registry.

        Args:
            node (ThoughtNode): The node to remove.
        """
        # Get all descendants
        descendants = node.get_descendants()

        # Remove node and all descendants from registry
        self.node_registry.pop(node.node_id, None)
        for descendant in descendants:
            self.node_registry.pop(descendant.node_id, None)

    def find_node(self, node_id: str) -> Optional[ThoughtNode]:
        """
        Find a node by its ID.

        Args:
            node_id (str): ID of the node to find.

        Returns:
            Optional[ThoughtNode]: The found node, or None if not found.
        """
        return self._get_node_by_id(node_id)

    def get_all_nodes(self) -> List[ThoughtNode]:
        """
        Get all nodes in the tree.

        Returns:
            List[ThoughtNode]: List of all nodes.
        """
        return list(self.node_registry.values())

    def get_tree_size(self) -> int:
        """
        Get the total number of nodes in the tree.

        Returns:
            int: Number of nodes.
        """
        return len(self.node_registry)

    def get_tree_depth(self) -> int:
        """
        Get the maximum depth of the tree.

        Returns:
            int: Maximum depth.
        """

        def calculate_depth(node: ThoughtNode, current_depth: int) -> int:
            if not node.children:
                return current_depth

            child_depths = [
                calculate_depth(child, current_depth + 1) for child in node.children
            ]
            return max(child_depths)

        return calculate_depth(self.root, 0)

    def __repr__(self) -> str:
        """
        Get a string representation of the tree.

        Returns:
            str: String representation.
        """
        return (
            f"MemorialTree(nodes={self.get_tree_size()}, "
            f"depth={self.get_tree_depth()}, "
            f"current_node='{self.current_node.content}')"
        )
