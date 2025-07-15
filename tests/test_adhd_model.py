"""
Tests for the ADHD model module of Memorial Tree.
"""

import unittest
from datetime import datetime, timedelta
import time

from src.memorial_tree.models import ADHDModel
from src.memorial_tree.core import MemorialTree, ThoughtNode


class TestADHDModel(unittest.TestCase):
    """Test cases for the ADHDModel class."""

    def test_init(self):
        """Test initialization of ADHDModel."""
        # Test with default values
        model = ADHDModel()
        self.assertEqual(model.attention_span, 0.3)
        self.assertEqual(model.impulsivity, 0.8)
        self.assertEqual(model.distraction_rate, 0.6)
        self.assertEqual(model.hyperactivity, 0.7)

        # Test with custom values
        model = ADHDModel(
            attention_span=0.5, impulsivity=0.6, distraction_rate=0.4, hyperactivity=0.3
        )
        self.assertEqual(model.attention_span, 0.5)
        self.assertEqual(model.impulsivity, 0.6)
        self.assertEqual(model.distraction_rate, 0.4)
        self.assertEqual(model.hyperactivity, 0.3)

        # Test with invalid values
        with self.assertRaises(ValueError):
            ADHDModel(attention_span=1.5)
        with self.assertRaises(ValueError):
            ADHDModel(impulsivity=-0.1)
        with self.assertRaises(ValueError):
            ADHDModel(distraction_rate=1.2)
        with self.assertRaises(ValueError):
            ADHDModel(hyperactivity=-0.5)

    def test_modify_decision_weights(self):
        """Test modifying decision weights based on ADHD characteristics."""
        model = ADHDModel()

        # Create sample decision weights
        decision_weights = {
            "option1": 1.0,
            "option2": 2.0,
            "option3": 3.0,
        }

        # Create sample context
        context = {
            "current_node": None,
            "path_history": [],
            "timestamp": datetime.now(),
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

    def test_attention_deficit_effect(self):
        """Test the attention deficit effect on decision weights."""
        # Create a model with extreme attention deficit
        model = ADHDModel(attention_span=0.0)  # No attention span

        # Create sample decision weights with large differences
        decision_weights = {
            "option1": 1.0,
            "option2": 10.0,
        }

        # Apply attention deficit effect
        model._apply_attention_deficit(decision_weights)

        # With no attention span, weights should be equal (average)
        self.assertAlmostEqual(decision_weights["option1"], decision_weights["option2"])
        self.assertAlmostEqual(decision_weights["option1"], 5.5)

        # Test with partial attention deficit
        model = ADHDModel(attention_span=0.5)  # Moderate attention span
        decision_weights = {
            "option1": 1.0,
            "option2": 10.0,
        }
        model._apply_attention_deficit(decision_weights)

        # With moderate attention span, weights should move toward average but not be equal
        self.assertLess(decision_weights["option1"], decision_weights["option2"])
        self.assertGreater(decision_weights["option1"], 1.0)
        self.assertLess(decision_weights["option2"], 10.0)

    def test_impulsivity_effect(self):
        """Test the impulsivity effect on decision weights."""
        # Create a model with high impulsivity
        model = ADHDModel(impulsivity=1.0)

        # Create sample decision weights
        decision_weights = {
            "option1": 1.0,
            "option2": 1.0,
        }

        # Apply impulsivity effect
        # Since the choice is random, we need to check that one weight was increased
        model._apply_impulsivity(decision_weights)

        # One weight should be doubled (1.0 + 1.0 impulsivity)
        self.assertEqual(sum(decision_weights.values()), 3.0)
        self.assertTrue(
            decision_weights["option1"] == 2.0 or decision_weights["option2"] == 2.0
        )

    def test_distraction_effect(self):
        """Test the distraction effect on decision weights."""
        # Create a model with guaranteed distraction
        model = ADHDModel(distraction_rate=1.0)
        model.last_distraction = datetime.now() - timedelta(
            seconds=10
        )  # Reset cooldown

        # Create sample decision weights
        decision_weights = {
            "option1": 1.0,
            "option2": 1.0,
        }

        # Create sample context
        context = {
            "current_node": None,
            "path_history": [],
            "timestamp": datetime.now(),
        }

        # Apply distraction effect
        model._apply_distraction(decision_weights, context)

        # One weight should be doubled
        self.assertEqual(sum(decision_weights.values()), 3.0)
        self.assertTrue(
            decision_weights["option1"] == 2.0 or decision_weights["option2"] == 2.0
        )

        # Test cooldown period
        # Reset weights
        decision_weights = {
            "option1": 1.0,
            "option2": 1.0,
        }

        # Apply distraction again immediately - should not change due to cooldown
        model._apply_distraction(decision_weights, context)
        self.assertEqual(decision_weights["option1"], 1.0)
        self.assertEqual(decision_weights["option2"], 1.0)

    def test_hyperactivity_effect(self):
        """Test the hyperactivity effect on decision weights."""
        # Create a model with maximum hyperactivity
        model = ADHDModel(hyperactivity=1.0)

        # Create sample decision weights
        original_weights = {
            "option1": 1.0,
            "option2": 1.0,
        }
        decision_weights = original_weights.copy()

        # Apply hyperactivity effect
        model._apply_hyperactivity(decision_weights)

        # Weights should be different from original due to random noise
        self.assertNotEqual(decision_weights["option1"], original_weights["option1"])
        self.assertNotEqual(decision_weights["option2"], original_weights["option2"])

        # Weights should remain positive
        self.assertGreater(decision_weights["option1"], 0)
        self.assertGreater(decision_weights["option2"], 0)

    def test_modify_decision_process(self):
        """Test modifying the decision process of a Memorial Tree."""
        # Create a model
        model = ADHDModel()

        # Create a simple tree
        tree = MemorialTree("Root")
        node1 = tree.add_thought(tree.root.node_id, "Option 1", weight=1.0)
        node2 = tree.add_thought(tree.root.node_id, "Option 2", weight=2.0)

        # Apply ADHD model to decision process
        model.modify_decision_process(tree, tree.root)

        # Check that modified weights were stored in metadata
        self.assertIn("adhd_modified_weights", tree.metadata)
        self.assertIn(node1.node_id, tree.metadata["adhd_modified_weights"])
        self.assertIn(node2.node_id, tree.metadata["adhd_modified_weights"])

        # Test with no available choices
        leaf_node = tree.add_thought(node1.node_id, "Leaf", weight=1.0)
        tree.current_node = leaf_node  # Move to leaf node

        # Clear metadata
        tree.metadata = {}

        # Apply ADHD model to decision process with no choices
        model.modify_decision_process(tree, leaf_node)

        # No modified weights should be stored
        self.assertEqual(tree.metadata, {})


if __name__ == "__main__":
    unittest.main()
