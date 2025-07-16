"""
Tests for the basic usage example.
"""

import unittest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import importlib.util

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the example module
spec = importlib.util.spec_from_file_location(
    "basic_usage",
    os.path.join(Path(__file__).parent.parent, "examples", "basic_usage.py"),
)
basic_usage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(basic_usage)


class TestBasicUsage(unittest.TestCase):
    """Test cases for the basic usage example."""

    @patch(
        "src.memorial_tree.visualization.tree_visualizer.TreeVisualizer.visualize_tree"
    )
    @patch(
        "src.memorial_tree.visualization.tree_visualizer.TreeVisualizer.visualize_path"
    )
    @patch(
        "src.memorial_tree.visualization.tree_visualizer.TreeVisualizer.visualize_ghost_influence"
    )
    @patch("matplotlib.pyplot.savefig")
    def test_main_execution(self, mock_savefig, mock_ghost, mock_path, mock_tree):
        """Test that the main function executes without errors."""
        # Configure mocks to return a mock figure
        mock_fig = MagicMock()
        mock_tree.return_value = mock_fig
        mock_path.return_value = mock_fig
        mock_ghost.return_value = mock_fig

        # Execute the main function
        basic_usage.main()

        # Check that visualization methods were called
        self.assertEqual(mock_tree.call_count, 1)
        self.assertEqual(mock_path.call_count, 1)
        self.assertEqual(mock_ghost.call_count, 1)

    def test_tree_creation(self):
        """Test the tree creation part of the example."""
        # Create a tree as in the example
        tree = basic_usage.MemorialTree(root_content="Should I go for a walk today?")

        # Add thoughts to the root node
        yes_node = tree.add_thought(
            parent_id=tree.root.node_id, content="Yes, I'll go for a walk", weight=0.7
        )

        no_node = tree.add_thought(
            parent_id=tree.root.node_id, content="No, I'll stay home", weight=0.3
        )

        # Verify the tree structure
        self.assertEqual(tree.root.content, "Should I go for a walk today?")
        self.assertEqual(len(tree.root.children), 2)
        self.assertEqual(yes_node.content, "Yes, I'll go for a walk")
        self.assertEqual(yes_node.weight, 0.7)
        self.assertEqual(no_node.content, "No, I'll stay home")
        self.assertEqual(no_node.weight, 0.3)

    def test_decision_making(self):
        """Test the decision-making part of the example."""
        # Create a tree as in the example
        tree = basic_usage.MemorialTree(root_content="Should I go for a walk today?")

        # Add thoughts to the root node
        yes_node = tree.add_thought(
            parent_id=tree.root.node_id, content="Yes, I'll go for a walk", weight=0.7
        )

        tree.add_thought(
            parent_id=tree.root.node_id, content="No, I'll stay home", weight=0.3
        )

        # Add second-level thoughts
        morning_node = tree.add_thought(
            parent_id=yes_node.node_id, content="Go in the morning", weight=0.6
        )

        # Make decisions
        tree.make_choice(yes_node.node_id)
        tree.make_choice(morning_node.node_id)

        # Verify the decision path
        path = tree.get_path_from_root()
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0].content, "Should I go for a walk today?")
        self.assertEqual(path[1].content, "Yes, I'll go for a walk")
        self.assertEqual(path[2].content, "Go in the morning")

        # Verify current state
        self.assertEqual(tree.current_node, morning_node)
        self.assertEqual(len(tree.path_history), 3)

    def test_ghost_node_addition(self):
        """Test adding ghost nodes as in the example."""
        # Create a tree as in the example
        tree = basic_usage.MemorialTree(root_content="Should I go for a walk today?")

        # Add a ghost node
        ghost_node = tree.add_ghost_node(
            content="Walking makes me anxious", influence=0.4, visibility=0.2
        )

        # Verify the ghost node
        self.assertEqual(len(tree.ghost_nodes), 1)
        self.assertEqual(ghost_node.content, "Walking makes me anxious")
        self.assertEqual(ghost_node.influence, 0.4)
        self.assertEqual(ghost_node.visibility, 0.2)


if __name__ == "__main__":
    unittest.main()
