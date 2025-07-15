"""
Depression Model module for Memorial Tree.

This module provides the DepressionModel class, which models the cognitive patterns
associated with Depression in decision-making.
"""

from typing import Dict, Any, List, Optional
import random
from datetime import datetime, timedelta

from ..core.memorial_tree import MemorialTree
from ..core.thought_node import ThoughtNode


class DepressionModel:
    """
    Model for simulating Depression cognitive patterns in decision-making.

    DepressionModel modifies the decision-making process to reflect characteristics
    of Depression such as negative bias, decision delay, and low energy levels.

    Attributes:
        negative_bias (float): Tendency to focus on negative options (0-1).
        decision_delay (float): Factor representing slowed decision-making (1-5).
        energy_level (float): Available mental energy for decisions (0-1).
        rumination (float): Tendency to dwell on past negative choices (0-1).
        last_decision_time (datetime): Timestamp of the last decision.
    """

    def __init__(
        self,
        negative_bias: float = 0.7,
        decision_delay: float = 2.0,
        energy_level: float = 0.3,
        rumination: float = 0.6,
    ):
        """
        Initialize a new DepressionModel.

        Args:
            negative_bias (float): Tendency to focus on negative options (0-1, higher means more negative focus).
            decision_delay (float): Factor representing slowed decision-making (1-5, higher means slower decisions).
            energy_level (float): Available mental energy for decisions (0-1, lower means less energy).
            rumination (float): Tendency to dwell on past negative choices (0-1, higher means more rumination).

        Raises:
            ValueError: If parameters are outside their valid ranges.
        """
        # Validate parameters
        if not 0 <= negative_bias <= 1:
            raise ValueError("negative_bias must be between 0 and 1")
        if not 1 <= decision_delay <= 5:
            raise ValueError("decision_delay must be between 1 and 5")
        if not 0 <= energy_level <= 1:
            raise ValueError("energy_level must be between 0 and 1")
        if not 0 <= rumination <= 1:
            raise ValueError("rumination must be between 0 and 1")

        self.negative_bias = negative_bias
        self.decision_delay = decision_delay
        self.energy_level = energy_level
        self.rumination = rumination
        self.last_decision_time = datetime.now() - timedelta(
            minutes=10
        )  # Initialize in the past
        self.negative_words = [
            "bad",
            "wrong",
            "terrible",
            "awful",
            "horrible",
            "sad",
            "depressing",
            "failure",
            "mistake",
            "problem",
            "difficult",
            "impossible",
            "hopeless",
            "worthless",
            "useless",
            "meaningless",
            "pointless",
            "regret",
            "worry",
        ]

    def modify_decision_weights(
        self, decision_weights: Dict[str, float], context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Modify decision weights based on Depression characteristics.

        Args:
            decision_weights (Dict[str, float]): Original weights for decision options.
            context (Dict[str, Any]): Current decision context including tree state.

        Returns:
            Dict[str, float]: Modified decision weights reflecting Depression influence.
        """
        # Create a copy of weights to modify
        modified_weights = decision_weights.copy()

        # Apply negative bias effect - increase weight for negative options
        self._apply_negative_bias(modified_weights, context)

        # Apply energy depletion effect - reduce overall motivation
        self._apply_energy_depletion(modified_weights)

        # Apply rumination effect - focus on past negative choices
        self._apply_rumination(modified_weights, context)

        # Apply decision delay effect - simulate slowed decision-making
        self._apply_decision_delay()

        return modified_weights

    def _apply_negative_bias(
        self, weights: Dict[str, float], context: Dict[str, Any]
    ) -> None:
        """
        Apply negative bias effects to decision weights.

        This increases the weight of options that appear negative or problematic.

        Args:
            weights (Dict[str, float]): Decision weights to modify.
            context (Dict[str, Any]): Current decision context.
        """
        if not weights:
            return

        # Get current node content for context
        current_node = context.get("current_node")
        if not current_node:
            return

        # Analyze each option for negative content
        for node_id, weight in weights.items():
            node = context.get("available_choices", [])
            node = next((n for n in node if n.node_id == node_id), None)

            if node:
                # Check if content contains negative words
                content_lower = node.content.lower()
                negativity_score = sum(
                    1 for word in self.negative_words if word in content_lower
                )

                # Increase weight for negative options based on negative bias
                if negativity_score > 0:
                    negativity_factor = 1.0 + (
                        self.negative_bias * negativity_score * 0.2
                    )
                    weights[node_id] *= negativity_factor

    def _apply_energy_depletion(self, weights: Dict[str, float]) -> None:
        """
        Apply energy depletion effects to decision weights.

        This reduces the weights of all options proportionally to energy level,
        making all choices seem less appealing.

        Args:
            weights (Dict[str, float]): Decision weights to modify.
        """
        if not weights:
            return

        # Calculate energy depletion factor (inverse of energy level)
        energy_depletion = 1.0 - self.energy_level

        # Reduce all weights based on energy depletion
        for option_id in weights:
            # Reduce weight by up to 70% based on energy level
            reduction_factor = 1.0 - (energy_depletion * 0.7)
            weights[option_id] *= max(
                0.1, reduction_factor
            )  # Ensure weight doesn't go too low

    def _apply_rumination(
        self, weights: Dict[str, float], context: Dict[str, Any]
    ) -> None:
        """
        Apply rumination effects to decision weights.

        This causes a focus on past negative experiences, potentially
        avoiding similar choices.

        Args:
            weights (Dict[str, float]): Decision weights to modify.
            context (Dict[str, Any]): Current decision context.
        """
        if not weights or not self.rumination > 0:
            return

        # Get path history
        path_history = context.get("path_history", [])
        if len(path_history) <= 1:  # No previous decisions
            return

        # Simulate rumination on past choices
        # In a real implementation, this would analyze past choices for negative outcomes
        # Here we'll just randomly select a choice to ruminate on
        if random.random() < self.rumination:
            # Select a random option to be affected by rumination
            rumination_choice = random.choice(list(weights.keys()))

            # Decrease its weight to simulate avoidance due to rumination
            weights[rumination_choice] *= max(0.2, 1.0 - self.rumination)

    def _apply_decision_delay(self) -> None:
        """
        Apply decision delay effects.

        This simulates the slowed decision-making process in depression.
        In a real-time system, this would introduce actual delays.
        """
        # Update last decision time
        self.last_decision_time = datetime.now()

        # In a real-time system, we might add:
        # time.sleep(self.decision_delay * 0.1)  # Simulate delay
        # But for this implementation, we just track the time

    def modify_decision_process(
        self, tree: MemorialTree, current_node: ThoughtNode
    ) -> None:
        """
        Modify the decision process of a Memorial Tree to reflect Depression characteristics.

        This method applies Depression-specific modifications to the tree's decision process,
        affecting how choices are evaluated and selected.

        Args:
            tree (MemorialTree): The Memorial Tree to modify.
            current_node (ThoughtNode): The current node in the decision process.
        """
        # Get available choices
        choices = tree.get_available_choices()
        if not choices:
            return  # No choices available

        # Get current state for context
        state = tree.get_current_state()

        # Create initial decision weights based on node weights
        decision_weights = {node.node_id: node.weight for node in choices}

        # Apply Depression modifications
        modified_weights = self.modify_decision_weights(decision_weights, state)

        # Store modified weights in the tree's metadata for reference
        tree.metadata["depression_modified_weights"] = modified_weights

    def __repr__(self) -> str:
        """
        Get a string representation of this Depression model.

        Returns:
            str: String representation of the Depression model.
        """
        return (
            f"DepressionModel(negative_bias={self.negative_bias:.2f}, "
            f"decision_delay={self.decision_delay:.2f}, "
            f"energy_level={self.energy_level:.2f}, "
            f"rumination={self.rumination:.2f})"
        )
