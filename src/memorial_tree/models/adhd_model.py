"""
ADHD Model module for Memorial Tree.

This module provides the ADHDModel class, which models the cognitive patterns
associated with Attention Deficit Hyperactivity Disorder (ADHD) in decision-making.
"""

from typing import Dict, Any, List, Optional
import random
from datetime import datetime, timedelta

from ..core.memorial_tree import MemorialTree
from ..core.thought_node import ThoughtNode


class ADHDModel:
    """
    Model for simulating ADHD cognitive patterns in decision-making.

    ADHDModel modifies the decision-making process to reflect characteristics
    of ADHD such as attention deficits, impulsivity, and distractibility.

    Attributes:
        attention_span (float): Represents the ability to maintain focus (0-1).
        impulsivity (float): Represents tendency for impulsive decisions (0-1).
        distraction_rate (float): Probability of being distracted by non-relevant options (0-1).
        hyperactivity (float): Level of hyperactivity affecting decision stability (0-1).
        last_distraction (datetime): Timestamp of the last distraction event.
        distraction_cooldown (float): Cooldown period between distractions in seconds.
    """

    def __init__(
        self,
        attention_span: float = 0.3,
        impulsivity: float = 0.8,
        distraction_rate: float = 0.6,
        hyperactivity: float = 0.7,
    ):
        """
        Initialize a new ADHDModel.

        Args:
            attention_span (float): Ability to maintain focus (0-1, lower means less focus).
            impulsivity (float): Tendency for impulsive decisions (0-1, higher means more impulsive).
            distraction_rate (float): Probability of distraction (0-1, higher means more distractible).
            hyperactivity (float): Level of hyperactivity (0-1, higher means more hyperactive).

        Raises:
            ValueError: If any parameter is not between 0 and 1.
        """
        # Validate parameters
        for param_name, param_value in {
            "attention_span": attention_span,
            "impulsivity": impulsivity,
            "distraction_rate": distraction_rate,
            "hyperactivity": hyperactivity,
        }.items():
            if not 0 <= param_value <= 1:
                raise ValueError(f"{param_name} must be between 0 and 1")

        self.attention_span = attention_span
        self.impulsivity = impulsivity
        self.distraction_rate = distraction_rate
        self.hyperactivity = hyperactivity
        self.last_distraction = datetime.now() - timedelta(
            minutes=10
        )  # Initialize in the past
        self.distraction_cooldown = 5.0  # 5 seconds between possible distractions

    def modify_decision_weights(
        self, decision_weights: Dict[str, float], context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Modify decision weights based on ADHD characteristics.

        Args:
            decision_weights (Dict[str, float]): Original weights for decision options.
            context (Dict[str, Any]): Current decision context including tree state.

        Returns:
            Dict[str, float]: Modified decision weights reflecting ADHD influence.
        """
        # Create a copy of weights to modify
        modified_weights = decision_weights.copy()

        # Apply attention deficit effect - reduce focus on optimal choices
        self._apply_attention_deficit(modified_weights)

        # Apply impulsivity effect - increase probability of quick, potentially suboptimal choices
        self._apply_impulsivity(modified_weights)

        # Apply distraction effect - potentially introduce focus on irrelevant options
        self._apply_distraction(modified_weights, context)

        # Apply hyperactivity effect - increase randomness in decision-making
        self._apply_hyperactivity(modified_weights)

        return modified_weights

    def _apply_attention_deficit(self, weights: Dict[str, float]) -> None:
        """
        Apply attention deficit effects to decision weights.

        This reduces the difference between high and low weighted options,
        making it harder to focus on the optimal choice.

        Args:
            weights (Dict[str, float]): Decision weights to modify.
        """
        if not weights:
            return

        # Calculate attention deficit factor (inverse of attention span)
        attention_deficit = 1.0 - self.attention_span

        # Find the average weight
        avg_weight = sum(weights.values()) / len(weights)

        # Move all weights closer to the average based on attention deficit
        for option_id in weights:
            # Calculate how much to shift toward average
            shift = (avg_weight - weights[option_id]) * attention_deficit
            weights[option_id] += shift

    def _apply_impulsivity(self, weights: Dict[str, float]) -> None:
        """
        Apply impulsivity effects to decision weights.

        This increases the weight of options that would be chosen quickly
        without much deliberation.

        Args:
            weights (Dict[str, float]): Decision weights to modify.
        """
        if not weights:
            return

        # Select a random option to be the "impulsive choice"
        impulsive_choice = random.choice(list(weights.keys()))

        # Increase its weight based on impulsivity level
        weights[impulsive_choice] *= 1.0 + self.impulsivity

    def _apply_distraction(
        self, weights: Dict[str, float], context: Dict[str, Any]
    ) -> None:
        """
        Apply distraction effects to decision weights.

        This may temporarily shift focus to less relevant options based on
        environmental factors or internal distractions.

        Args:
            weights (Dict[str, float]): Decision weights to modify.
            context (Dict[str, Any]): Current decision context.
        """
        if not weights:
            return

        # Check if enough time has passed since last distraction
        now = datetime.now()
        time_since_last = (now - self.last_distraction).total_seconds()

        if time_since_last < self.distraction_cooldown:
            return  # Still in cooldown period

        # Check if distraction occurs based on distraction rate
        if random.random() < self.distraction_rate:
            # Select a random option to be the "distraction"
            distraction_choice = random.choice(list(weights.keys()))

            # Significantly increase its weight temporarily
            weights[distraction_choice] *= 2.0

            # Update last distraction time
            self.last_distraction = now

    def _apply_hyperactivity(self, weights: Dict[str, float]) -> None:
        """
        Apply hyperactivity effects to decision weights.

        This introduces randomness and variability in the decision process.

        Args:
            weights (Dict[str, float]): Decision weights to modify.
        """
        if not weights:
            return

        # Add random noise to all weights based on hyperactivity level
        for option_id in weights:
            # Generate random noise factor between -0.5 and 0.5
            noise = (random.random() - 0.5) * self.hyperactivity

            # Apply noise to weight (ensure weight stays positive)
            weights[option_id] = max(0.01, weights[option_id] * (1.0 + noise))

    def modify_decision_process(
        self, tree: MemorialTree, current_node: ThoughtNode
    ) -> None:
        """
        Modify the decision process of a Memorial Tree to reflect ADHD characteristics.

        This method applies ADHD-specific modifications to the tree's decision process,
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

        # Apply ADHD modifications
        modified_weights = self.modify_decision_weights(decision_weights, state)

        # Store modified weights in the tree's metadata for reference
        # This doesn't change the actual decision but provides visibility into the model's effect
        tree.metadata["adhd_modified_weights"] = modified_weights

    def __repr__(self) -> str:
        """
        Get a string representation of this ADHD model.

        Returns:
            str: String representation of the ADHD model.
        """
        return (
            f"ADHDModel(attention_span={self.attention_span:.2f}, "
            f"impulsivity={self.impulsivity:.2f}, "
            f"distraction_rate={self.distraction_rate:.2f}, "
            f"hyperactivity={self.hyperactivity:.2f})"
        )
