"""
ThoughtNode module for Memorial Tree.

This module provides the ThoughtNode class, which is the base class for representing
thoughts and decisions in the Memorial Tree structure.
"""

from typing import List, Optional, Any, Dict
from datetime import datetime
import uuid


class ThoughtNode:
    """
    Base class for representing thoughts and decisions in a Memorial Tree.

    ThoughtNode manages node ID, content, weight, and parent-child relationships.
    It provides methods for node creation, child addition, and path traversal.

    Attributes:
        content (str): The content or description of the thought.
        node_id (str): Unique identifier for the node.
        weight (float): The weight or importance of the node (default: 1.0).
        children (List['ThoughtNode']): List of child nodes.
        parent (Optional['ThoughtNode']): Reference to parent node.
        activation_count (int): Number of times this node has been activated.
        timestamp (datetime): When the node was created.
        metadata (Dict[str, Any]): Additional data associated with the node.
    """

    def __init__(
        self, content: str, node_id: Optional[str] = None, weight: float = 1.0
    ):
        """
        Initialize a new ThoughtNode.

        Args:
            content (str): The content or description of the thought.
            node_id (Optional[str]): Unique identifier for the node. If None, a UUID will be generated.
            weight (float): The weight or importance of the node (default: 1.0).
        """
        self.content = content
        self.node_id = node_id if node_id else str(uuid.uuid4())
        self.weight = weight
        self.children: List["ThoughtNode"] = []
        self.parent: Optional["ThoughtNode"] = None
        self.activation_count = 0
        self.timestamp = datetime.now()
        self.metadata: Dict[str, Any] = {}

    def add_child(self, child_node: "ThoughtNode") -> "ThoughtNode":
        """
        Add a child node to this node.

        Args:
            child_node (ThoughtNode): The node to add as a child.

        Returns:
            ThoughtNode: The added child node.

        Raises:
            ValueError: If adding the child would create a circular reference.
        """
        # Check for circular references
        if self._would_create_circular_reference(child_node):
            raise ValueError(
                f"Adding node {child_node.node_id} would create a circular reference"
            )

        # Set parent-child relationship
        child_node.parent = self
        self.children.append(child_node)
        return child_node

    def _would_create_circular_reference(self, node: "ThoughtNode") -> bool:
        """
        Check if adding the given node would create a circular reference.

        Args:
            node (ThoughtNode): The node to check.

        Returns:
            bool: True if adding the node would create a circular reference, False otherwise.
        """
        # If this node is the same as the node to add
        if self.node_id == node.node_id:
            return True

        # Check if the node to add is already an ancestor of this node
        current = self.parent
        while current:
            if current.node_id == node.node_id:
                return True
            current = current.parent

        return False

    def get_path_to_root(self) -> List["ThoughtNode"]:
        """
        Get the path from this node to the root node.

        Returns:
            List[ThoughtNode]: List of nodes from this node to the root (inclusive).
        """
        path = [self]
        current = self.parent

        while current:
            path.append(current)
            current = current.parent

        return path

    def calculate_influence(self) -> float:
        """
        Calculate the influence of this node based on its weight and activation history.

        Returns:
            float: The calculated influence value.
        """
        # Basic influence calculation based on weight and activation count
        base_influence = self.weight
        activation_factor = 1.0 + (
            0.1 * self.activation_count
        )  # Increases with more activations

        return base_influence * activation_factor

    def activate(self) -> None:
        """
        Activate this node, incrementing its activation count and updating timestamp.
        """
        self.activation_count += 1
        self.timestamp = datetime.now()

    def get_descendants(self) -> List["ThoughtNode"]:
        """
        Get all descendant nodes of this node.

        Returns:
            List[ThoughtNode]: List of all descendant nodes.
        """
        descendants = []

        def collect_descendants(node: "ThoughtNode") -> None:
            for child in node.children:
                descendants.append(child)
                collect_descendants(child)

        collect_descendants(self)
        return descendants

    def find_child_by_id(self, node_id: str) -> Optional["ThoughtNode"]:
        """
        Find a direct child node by its ID.

        Args:
            node_id (str): The ID of the node to find.

        Returns:
            Optional[ThoughtNode]: The found node, or None if not found.
        """
        for child in self.children:
            if child.node_id == node_id:
                return child
        return None

    def find_descendant_by_id(self, node_id: str) -> Optional["ThoughtNode"]:
        """
        Find a descendant node by its ID.

        Args:
            node_id (str): The ID of the node to find.

        Returns:
            Optional[ThoughtNode]: The found node, or None if not found.
        """
        # Check direct children first
        child = self.find_child_by_id(node_id)
        if child:
            return child

        # Then check descendants
        for child in self.children:
            descendant = child.find_descendant_by_id(node_id)
            if descendant:
                return descendant

        return None

    def __repr__(self) -> str:
        """
        Get a string representation of this node.

        Returns:
            str: String representation of the node.
        """
        return f"ThoughtNode(id={self.node_id}, content='{self.content}', weight={self.weight})"
