"""
Anxiety Model module for Memorial Tree.

This module provides the AnxietyModel class, which models the cognitive patterns
associated with Anxiety Disorder in decision-making.
"""

from typing import Dict, Any
import random
from datetime import datetime, timedelta

from ..core.memorial_tree import MemorialTree
from ..core.thought_node import ThoughtNode


class AnxietyModel:
    """
    Model for simulating Anxiety Disorder cognitive patterns in decision-making.

    AnxietyModel modifies the decision-making process to reflect characteristics
    of Anxiety Disorder such as excessive worry, risk aversion, and rumination.

    Attributes:
        worry_amplification (float): Tendency to amplify potential risks (0-1).
        risk_aversion (float): Tendency to avoid perceived risky options (0-1).
        rumination_cycles (int): Number of cycles spent overthinking decisions (1-5).
        uncertainty_intolerance (float): Difficulty handling uncertain situations (0-1).
        last_worry_time (datetime): Timestamp of the last worry event.
        worry_keywords (list): List of words that trigger anxiety responses.
    """

    def __init__(
        self,
        worry_amplification: float = 0.8,
        risk_aversion: float = 0.9,
        rumination_cycles: int = 3,
        uncertainty_intolerance: float = 0.7,
    ):
        """
        Initialize a new AnxietyModel.

        Args:
            worry_amplification (float): Tendency to amplify potential risks (0-1).
            risk_aversion (float): Tendency to avoid perceived risky options (0-1).
            rumination_cycles (int): Number of cycles spent overthinking decisions (1-5).
            uncertainty_intolerance (float): Difficulty handling uncertain situations (0-1).

        Raises:
            ValueError: If parameters are outside their valid ranges.
        """
        # Validate parameters
        if not 0 <= worry_amplification <= 1:
            raise ValueError("worry_amplification must be between 0 and 1")
        if not 0 <= risk_aversion <= 1:
            raise ValueError("risk_aversion must be between 0 and 1")
        if not 1 <= rumination_cycles <= 5:
            raise ValueError("rumination_cycles must be between 1 and 5")
        if not 0 <= uncertainty_intolerance <= 1:
            raise ValueError("uncertainty_intolerance must be between 0 and 1")

        self.worry_amplification = worry_amplification
        self.risk_aversion = risk_aversion
        self.rumination_cycles = rumination_cycles
        self.uncertainty_intolerance = uncertainty_intolerance
        self.last_worry_time = datetime.now() - timedelta(minutes=10)

        # Words that trigger anxiety or indicate risk
        self.worry_keywords = [
            "risk",
            "danger",
            "uncertain",
            "maybe",
            "possibly",
            "might",
            "could",
            "perhaps",
            "worry",
            "anxious",
            "fear",
            "scary",
            "afraid",
            "unknown",
            "doubt",
            "unsure",
            "problem",
            "difficult",
            "challenge",
            "threat",
            "harmful",
            "unsafe",
            "concern",
        ]

    def modify_decision_weights(
        self, decision_weights: Dict[str, float], context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Modify decision weights based on Anxiety Disorder characteristics.

        Args:
            decision_weights (Dict[str, float]): Original weights for decision options.
            context (Dict[str, Any]): Current decision context including tree state.

        Returns:
            Dict[str, float]: Modified decision weights reflecting Anxiety influence.
        """
        # Create a copy of weights to modify
        modified_weights = decision_weights.copy()

        # Apply worry amplification effect - increase focus on perceived risks
        self._apply_worry_amplification(modified_weights, context)

        # Apply risk aversion effect - reduce weight of risky options
        self._apply_risk_aversion(modified_weights, context)

        # Apply rumination effect - simulate overthinking by cycling through options
        self._apply_rumination(modified_weights)

        # Apply uncertainty intolerance - reduce weight of ambiguous options
        self._apply_uncertainty_intolerance(modified_weights, context)

        return modified_weights

    def _apply_worry_amplification(
        self, weights: Dict[str, float], context: Dict[str, Any]
    ) -> None:
        """
        Apply worry amplification effects to decision weights.

        This increases focus on options that contain worry-triggering keywords,
        as anxious individuals tend to fixate on potential threats.

        Args:
            weights (Dict[str, float]): Decision weights to modify.
            context (Dict[str, Any]): Current decision context.
        """
        if not weights:
            return

        # Get available choices from context
        available_choices = context.get("available_choices", [])
        if not available_choices:
            return

        # Check each option for worry-triggering content
        for node in available_choices:
            if node.node_id not in weights:
                continue

            # Count worry keywords in the content
            content_lower = node.content.lower()
            worry_score = sum(
                1 for word in self.worry_keywords if word in content_lower
            )

            if worry_score > 0:
                # Amplify the weight of worry-inducing options to reflect fixation
                # This makes anxious individuals paradoxically focus more on what worries them
                amplification_factor = 1.0 + (
                    self.worry_amplification * worry_score * 0.2
                )
                weights[node.node_id] *= amplification_factor

    def _apply_risk_aversion(
        self, weights: Dict[str, float], context: Dict[str, Any]
    ) -> None:
        """
        Apply risk aversion effects to decision weights.

        This reduces the weight of options perceived as risky and increases
        the weight of options perceived as safe.

        Args:
            weights (Dict[str, float]): Decision weights to modify.
            context (Dict[str, Any]): Current decision context.
        """
        if not weights:
            return

        # Get available choices from context
        available_choices = context.get("available_choices", [])
        if not available_choices:
            return

        # Identify risky and safe options
        risky_options = []
        safe_options = []

        for node in available_choices:
            if node.node_id not in weights:
                continue

            content_lower = node.content.lower()

            # Check for risk indicators
            risk_score = sum(
                1
                for word in ["risk", "danger", "uncertain", "might", "could", "maybe"]
                if word in content_lower
            )

            # Check for safety indicators
            safety_score = sum(
                1
                for word in [
                    "safe",
                    "secure",
                    "certain",
                    "sure",
                    "definitely",
                    "always",
                ]
                if word in content_lower
            )

            if risk_score > safety_score:
                risky_options.append(node.node_id)
            elif safety_score > risk_score:
                safe_options.append(node.node_id)

        # Apply risk aversion - reduce weight of risky options
        for option_id in risky_options:
            weights[option_id] *= 1.0 - self.risk_aversion * 0.5

        # Increase weight of safe options
        for option_id in safe_options:
            weights[option_id] *= 1.0 + self.risk_aversion * 0.3

    def _apply_rumination(self, weights: Dict[str, float]) -> None:
        """
        Apply rumination effects to decision weights.

        This simulates overthinking by cycling through options multiple times,
        potentially changing their weights in each cycle to reflect the
        back-and-forth nature of anxious rumination.

        Args:
            weights (Dict[str, float]): Decision weights to modify.
        """
        if not weights:
            return

        # Simulate rumination cycles
        for _ in range(self.rumination_cycles):
            # Select a random option to focus on in this cycle
            focus_option = random.choice(list(weights.keys()))

            # In each cycle, we might either increase or decrease focus on an option
            # This simulates the back-and-forth nature of rumination
            if random.random() < 0.5:
                # Increase focus on this option
                weights[focus_option] *= 1.1
            else:
                # Decrease focus on this option
                weights[focus_option] *= 0.9

    def _apply_uncertainty_intolerance(
        self, weights: Dict[str, float], context: Dict[str, Any]
    ) -> None:
        """
        Apply uncertainty intolerance effects to decision weights.

        This reduces the weight of options with ambiguous or uncertain outcomes,
        as anxious individuals tend to avoid uncertainty.

        Args:
            weights (Dict[str, float]): Decision weights to modify.
            context (Dict[str, Any]): Current decision context.
        """
        if not weights:
            return

        # Get available choices from context
        available_choices = context.get("available_choices", [])
        if not available_choices:
            return

        # Check each option for uncertainty indicators
        uncertainty_keywords = [
            "maybe",
            "perhaps",
            "might",
            "could",
            "possibly",
            "uncertain",
        ]

        for node in available_choices:
            if node.node_id not in weights:
                continue

            content_lower = node.content.lower()

            # Count uncertainty indicators
            uncertainty_score = sum(
                1 for word in uncertainty_keywords if word in content_lower
            )

            if uncertainty_score > 0:
                # Reduce weight of uncertain options based on intolerance level
                reduction_factor = 1.0 - (
                    self.uncertainty_intolerance * uncertainty_score * 0.15
                )
                weights[node.node_id] *= max(
                    0.2, reduction_factor
                )  # Ensure weight doesn't go too low

    def modify_decision_process(
        self, tree: MemorialTree, current_node: ThoughtNode
    ) -> None:
        """
        Modify the decision process of a Memorial Tree to reflect Anxiety Disorder characteristics.

        This method applies Anxiety-specific modifications to the tree's decision process,
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

        # Apply Anxiety modifications
        modified_weights = self.modify_decision_weights(decision_weights, state)

        # Store modified weights in the tree's metadata for reference
        tree.metadata["anxiety_modified_weights"] = modified_weights

    def __repr__(self) -> str:
        """
        Get a string representation of this Anxiety model.

        Returns:
            str: String representation of the Anxiety model.
        """
        return (
            f"AnxietyModel(worry_amplification={self.worry_amplification:.2f}, "
            f"risk_aversion={self.risk_aversion:.2f}, "
            f"rumination_cycles={self.rumination_cycles}, "
            f"uncertainty_intolerance={self.uncertainty_intolerance:.2f})"
        )
