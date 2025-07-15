"""
GhostNode module for Memorial Tree.

This module provides the GhostNode class, which extends ThoughtNode to represent
unconscious influences or suppressed memories in the Memorial Tree structure.
"""

from typing import List, Optional, Dict, Any, Callable
from datetime import datetime

from .thought_node import ThoughtNode


class GhostNode(ThoughtNode):
    """
    Extension of ThoughtNode for representing unconscious influences in a Memorial Tree.

    GhostNode manages influence, visibility, and activation conditions for modeling
    unconscious factors that affect decision-making processes.

    Attributes:
        content (str): The content or description of the thought.
        node_id (str): Unique identifier for the node.
        weight (float): The weight or importance of the node (default: 1.0).
        influence (float): The strength of influence this ghost node has (0-1).
        visibility (float): How visible/conscious this node is to the decision-maker (0-1).
        trigger_conditions (List[Callable]): Conditions that can trigger this ghost node.
        activation_log (List[datetime]): History of when this node was activated.
    """

    def __init__(
        self,
        content: str,
        node_id: Optional[str] = None,
        influence: float = 0.3,
        visibility: float = 0.1,
        weight: float = 1.0,
    ):
        """
        Initialize a new GhostNode.

        Args:
            content (str): The content or description of the thought.
            node_id (Optional[str]): Unique identifier for the node. If None, a UUID will be generated.
            influence (float): The strength of influence this ghost node has (0-1).
            visibility (float): How visible/conscious this node is to the decision-maker (0-1).
            weight (float): The weight or importance of the node (default: 1.0).

        Raises:
            ValueError: If influence or visibility is not between 0 and 1.
        """
        super().__init__(content, node_id, weight)

        # Validate influence and visibility
        if not 0 <= influence <= 1:
            raise ValueError("Influence must be between 0 and 1")
        if not 0 <= visibility <= 1:
            raise ValueError("Visibility must be between 0 and 1")

        self.influence = influence
        self.visibility = visibility
        self.trigger_conditions: List[Callable] = []
        self.activation_log: List[datetime] = []

    def add_trigger_condition(self, condition: Callable) -> None:
        """
        Add a trigger condition that can activate this ghost node.

        Args:
            condition (Callable): A function that takes the current context and returns
                                 True if the ghost node should be activated, False otherwise.
        """
        self.trigger_conditions.append(condition)

    def check_activation(self, current_context: Dict[str, Any]) -> bool:
        """
        Check if this ghost node should be activated based on the current context.

        Args:
            current_context (Dict[str, Any]): The current context of the decision process.

        Returns:
            bool: True if the ghost node should be activated, False otherwise.
        """
        # If no trigger conditions, use a random chance based on visibility
        if not self.trigger_conditions:
            import random

            return random.random() < self.visibility

        # Check all trigger conditions
        for condition in self.trigger_conditions:
            if condition(current_context):
                return True

        return False

    def activate(self) -> None:
        """
        Activate this ghost node, incrementing its activation count and updating logs.
        """
        super().activate()
        self.activation_log.append(datetime.now())

    def apply_influence(self, decision_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply this ghost node's influence to a set of decision weights.

        Args:
            decision_weights (Dict[str, float]): Current weights for different decision options.

        Returns:
            Dict[str, float]: Modified decision weights after applying ghost node influence.
        """
        # Create a copy of the weights to modify
        modified_weights = decision_weights.copy()

        # Calculate the influence factor based on node properties
        influence_factor = self.calculate_influence()

        # Apply the ghost node's influence according to its properties
        # This is a simple implementation - real applications might use more complex logic
        for key in modified_weights:
            # Modify each weight based on the ghost node's influence
            # The exact modification would depend on the specific use case
            # Here we're just applying a simple scaling factor
            modified_weights[key] *= 1.0 + (self.influence - 0.5) * influence_factor

        return modified_weights

    def calculate_influence(self) -> float:
        """
        Calculate the influence of this ghost node based on its properties and history.

        Returns:
            float: The calculated influence value.
        """
        # Start with the base influence calculation from ThoughtNode
        base_influence = super().calculate_influence()

        # Adjust based on ghost-specific properties
        visibility_factor = 1.0 - (
            self.visibility * 0.5
        )  # Less visible nodes have more influence
        recency_factor = self._calculate_recency_factor()

        return base_influence * self.influence * visibility_factor * recency_factor

    def _calculate_recency_factor(self) -> float:
        """
        Calculate a factor based on how recently this ghost node was activated.

        Returns:
            float: A factor between 0.5 and 1.5 based on recency of activation.
        """
        if not self.activation_log:
            return 1.0

        # Calculate time since last activation
        now = datetime.now()
        last_activation = self.activation_log[-1]
        time_diff = (now - last_activation).total_seconds()

        # Recent activations have stronger influence, which decays over time
        # This is a simple decay function that ranges from 1.5 (recent) to 0.5 (old)
        decay_rate = 0.001  # Adjust this to control decay speed
        recency_factor = 1.0 + 0.5 * (2 * (1 / (1 + time_diff * decay_rate)) - 1)

        return recency_factor

    def __repr__(self) -> str:
        """
        Get a string representation of this ghost node.

        Returns:
            str: String representation of the ghost node.
        """
        return (
            f"GhostNode(id={self.node_id}, content='{self.content}', "
            f"influence={self.influence:.2f}, visibility={self.visibility:.2f})"
        )
