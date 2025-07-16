"""
Tests for the Depression Model example.
"""

import unittest
import sys
import os
from io import StringIO
from unittest.mock import patch

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the example module
import examples.depression_model_example as depression_example


class TestDepressionModelExample(unittest.TestCase):
    """Test cases for the Depression Model example."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_example_runs_without_errors(self, mock_stdout):
        """Test that the example runs without raising exceptions."""
        try:
            depression_example.main()
            output = mock_stdout.getvalue()

            # Check that the example produced output
            self.assertGreater(len(output), 0)

            # Check for key sections in the output
            self.assertIn("Depression Model Example", output)
            self.assertIn("Original decision weights", output)
            self.assertIn("Depression-modified decision weights", output)
            self.assertIn("Analysis of Depression Model Effects", output)

        except Exception as e:
            self.fail(f"Depression model example raised an exception: {e}")

    def test_example_creates_valid_tree(self):
        """Test that the example creates a valid tree structure."""
        # Create a tree as in the example
        tree = depression_example.MemorialTree("What should I do today?")

        # Add options
        option1 = tree.add_thought(
            tree.root.node_id, "Go out and meet friends", weight=1.0
        )
        option2 = tree.add_thought(
            tree.root.node_id, "Work on a personal project", weight=1.0
        )
        option3 = tree.add_thought(
            tree.root.node_id,
            "Stay in bed, it's pointless to try anything today",
            weight=1.0,
        )
        option4 = tree.add_thought(
            tree.root.node_id, "Think about past failures and mistakes", weight=1.0
        )

        # Check tree structure
        self.assertEqual(tree.root.content, "What should I do today?")
        self.assertEqual(len(tree.root.children), 4)
        self.assertEqual(tree.get_tree_size(), 5)  # Root + 4 options

        # Check that all options are children of the root
        for option in [option1, option2, option3, option4]:
            self.assertIn(option, tree.root.children)

    def test_depression_model_modifies_weights(self):
        """Test that the depression model modifies decision weights as expected."""
        # Create a tree as in the example
        tree = depression_example.MemorialTree("What should I do today?")

        # Add options with varying emotional content
        option1 = tree.add_thought(
            tree.root.node_id, "Go out and meet friends", weight=1.0
        )
        option2 = tree.add_thought(
            tree.root.node_id, "Work on a personal project", weight=1.0
        )
        option3 = tree.add_thought(
            tree.root.node_id,
            "Stay in bed, it's pointless to try anything today",
            weight=1.0,
        )
        option4 = tree.add_thought(
            tree.root.node_id, "Think about past failures and mistakes", weight=1.0
        )

        # Get original weights
        original_weights = {
            node.node_id: node.weight for node in tree.get_available_choices()
        }

        # Create and apply depression model
        depression_model = depression_example.DepressionModel(
            negative_bias=0.7,
            decision_delay=2.0,
            energy_level=0.3,
            rumination=0.6,
        )
        depression_model.modify_decision_process(tree, tree.root)

        # Get modified weights
        modified_weights = tree.metadata.get("depression_modified_weights", {})

        # Check that weights were modified
        self.assertNotEqual(original_weights, modified_weights)

        # Check that all options have entries in modified weights
        for node_id in original_weights:
            self.assertIn(node_id, modified_weights)

        # Check that negative options (3 and 4) have higher relative weights due to negative bias
        # We compare the ratio of modified to original weights
        option3_ratio = (
            modified_weights[option3.node_id] / original_weights[option3.node_id]
        )
        option4_ratio = (
            modified_weights[option4.node_id] / original_weights[option4.node_id]
        )
        option1_ratio = (
            modified_weights[option1.node_id] / original_weights[option1.node_id]
        )

        # At least one of the negative options should have a higher ratio than the positive option
        self.assertTrue(
            option3_ratio > option1_ratio or option4_ratio > option1_ratio,
            "Depression model should increase weight of negative options relative to positive ones",
        )


if __name__ == "__main__":
    unittest.main()
