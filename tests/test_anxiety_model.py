"""
Tests for the Anxiety model module of Memorial Tree.
"""

import unittest
from datetime import datetime, timedelta

from src.memorial_tree.models import AnxietyModel
from src.memorial_tree.core import MemorialTree, ThoughtNode


class TestAnxietyModel(unittest.TestCase):
    """Test cases for the AnxietyModel class."""

    def test_init(self):
        """Test initialization of AnxietyModel."""
        # Test with default values
        model = AnxietyModel()
        self.assertEqual(model.worry_amplification, 0.8)
        self.assertEqual(model.risk_aversion, 0.9)
        self.assertEqual(model.rumination_cycles, 3)
        self.assertEqual(model.uncertainty_intolerance, 0.7)

        # Test with custom values
        model = AnxietyModel(
            worry_amplification=0.5,
            risk_aversion=0.6,
            rumination_cycles=2,
            uncertainty_intolerance=0.4,
        )
        self.assertEqual(model.worry_amplification, 0.5)
        self.assertEqual(model.risk_aversion, 0.6)
        self.assertEqual(model.rumination_cycles, 2)
        self.assertEqual(model.uncertainty_intolerance, 0.4)

        # Test with invalid values
        with self.assertRaises(ValueError):
            AnxietyModel(worry_amplification=1.5)
        with self.assertRaises(ValueError):
            AnxietyModel(risk_aversion=-0.1)
        with self.assertRaises(ValueError):
            AnxietyModel(rumination_cycles=0)
        with self.assertRaises(ValueError):
            AnxietyModel(rumination_cycles=6)
        with self.assertRaises(ValueError):
            AnxietyModel(uncertainty_intolerance=1.2)

    def test_modify_decision_weights(self):
        """Test modifying decision weights based on Anxiety characteristics."""
        model = AnxietyModel()

        # Create sample decision weights
        decision_weights = {
            "option1": 1.0,
            "option2": 2.0,
            "option3": 3.0,
        }

        # Create mock nodes for context
        node1 = ThoughtNode("A safe and certain option", node_id="option1")
        node2 = ThoughtNode(
            "A risky and uncertain option that might be dangerous", node_id="option2"
        )
        node3 = ThoughtNode("A neutral option", node_id="option3")

        # Create sample context
        context = {
            "current_node": node1,  # Set a current node
            "path_history": ["root", "option1"],
            "timestamp": datetime.now(),
            "available_choices": [node1, node2, node3],
        }

        # Modify weights
        modified_weights = model.modify_decision_weights(decision_weights, context)

        # Check that the original weights were not modified
        self.assertEqual(decision_weights["option1"], 1.0)
        self.assertEqual(decision_weights["option2"], 2.0)
        self.assertEqual(decision_weights["option3"], 3.0)

        # Check that the modified weights are different
        self.assertNotEqual(modified_weights, decision_weights)

        # Check that all options are still present
        self.assertEqual(set(modified_weights.keys()), set(decision_weights.keys()))

    def test_worry_amplification_effect(self):
        """Test the worry amplification effect on decision weights."""
        # Create a model with high worry amplification
        model = AnxietyModel(worry_amplification=1.0)

        # Create sample decision weights
        decision_weights = {
            "option1": 1.0,  # neutral option
            "option2": 1.0,  # worry-inducing option
        }

        # Create mock nodes for context
        node1 = ThoughtNode("A normal option", node_id="option1")
        node2 = ThoughtNode("A worrying option with risk and danger", node_id="option2")

        # Create sample context with a current node
        context = {"current_node": node1, "available_choices": [node1, node2]}

        # Apply worry amplification effect
        model._apply_worry_amplification(decision_weights, context)

        # The worry-inducing option should have a higher weight due to fixation
        self.assertGreater(decision_weights["option2"], decision_weights["option1"])

    def test_risk_aversion_effect(self):
        """Test the risk aversion effect on decision weights."""
        # Create a model with high risk aversion
        model = AnxietyModel(risk_aversion=1.0)

        # Create sample decision weights
        decision_weights = {
            "option1": 1.0,  # safe option
            "option2": 1.0,  # risky option
            "option3": 1.0,  # neutral option
        }

        # Create mock nodes for context
        node1 = ThoughtNode("A safe and secure option", node_id="option1")
        node2 = ThoughtNode("A risky and uncertain option", node_id="option2")
        node3 = ThoughtNode("A neutral option", node_id="option3")

        # Create sample context
        context = {"current_node": node3, "available_choices": [node1, node2, node3]}

        # Apply risk aversion effect
        model._apply_risk_aversion(decision_weights, context)

        # The safe option should have a higher weight than the risky option
        self.assertGreater(decision_weights["option1"], decision_weights["option2"])

        # The neutral option should remain unchanged
        self.assertEqual(decision_weights["option3"], 1.0)

    def test_rumination_effect(self):
        """Test the rumination effect on decision weights."""
        # Create a model with high rumination cycles
        model = AnxietyModel(rumination_cycles=5)

        # Create sample decision weights
        original_weights = {
            "option1": 1.0,
            "option2": 1.0,
        }
        decision_weights = original_weights.copy()

        # Apply rumination effect
        model._apply_rumination(decision_weights)

        # Weights should be modified after rumination
        self.assertNotEqual(decision_weights["option1"], original_weights["option1"])
        self.assertNotEqual(decision_weights["option2"], original_weights["option2"])

        # Test with no rumination
        model = AnxietyModel(rumination_cycles=1)
        decision_weights = original_weights.copy()

        # Set a seed to make the test deterministic
        import random

        random.seed(42)

        model._apply_rumination(decision_weights)

        # With just one cycle, there should still be some change
        self.assertNotEqual(decision_weights, original_weights)

    def test_uncertainty_intolerance_effect(self):
        """Test the uncertainty intolerance effect on decision weights."""
        # Create a model with high uncertainty intolerance
        model = AnxietyModel(uncertainty_intolerance=1.0)

        # Create sample decision weights
        decision_weights = {
            "option1": 1.0,  # certain option
            "option2": 1.0,  # uncertain option
        }

        # Create mock nodes for context
        node1 = ThoughtNode("A definite and certain option", node_id="option1")
        node2 = ThoughtNode(
            "An option that might possibly work maybe", node_id="option2"
        )

        # Create sample context
        context = {"current_node": node1, "available_choices": [node1, node2]}

        # Apply uncertainty intolerance effect
        model._apply_uncertainty_intolerance(decision_weights, context)

        # The uncertain option should have a lower weight
        self.assertLess(decision_weights["option2"], decision_weights["option1"])

    def test_modify_decision_process(self):
        """Test modifying the decision process of a Memorial Tree."""
        # Create a model
        model = AnxietyModel()

        # Create a simple tree
        tree = MemorialTree("Root")
        node1 = tree.add_thought(
            tree.root.node_id, "Safe and certain option", weight=1.0
        )
        node2 = tree.add_thought(
            tree.root.node_id, "Risky and uncertain option", weight=1.0
        )

        # Apply Anxiety model to decision process
        model.modify_decision_process(tree, tree.root)

        # Check that modified weights were stored in metadata
        self.assertIn("anxiety_modified_weights", tree.metadata)
        self.assertIn(node1.node_id, tree.metadata["anxiety_modified_weights"])
        self.assertIn(node2.node_id, tree.metadata["anxiety_modified_weights"])

        # Test with no available choices
        leaf_node = tree.add_thought(node1.node_id, "Leaf", weight=1.0)
        tree.current_node = leaf_node  # Move to leaf node

        # Clear metadata
        tree.metadata = {}

        # Apply Anxiety model to decision process with no choices
        model.modify_decision_process(tree, leaf_node)

        # No modified weights should be stored
        self.assertEqual(tree.metadata, {})


if __name__ == "__main__":
    unittest.main()
