"""
Tests for the MemorialTree class.
"""

import unittest
from datetime import datetime

from src.memorial_tree.core import MemorialTree, ThoughtNode, GhostNode
from src.memorial_tree.core.memorial_tree import NodeNotFoundError


class TestMemorialTree(unittest.TestCase):
    """Test cases for the MemorialTree class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tree = MemorialTree("Root Thought")

    def test_init(self):
        """Test initialization of MemorialTree."""
        # Check default initialization
        self.assertEqual(self.tree.root.content, "Root Thought")
        self.assertEqual(self.tree.current_node, self.tree.root)
        self.assertEqual(len(self.tree.path_history), 1)
        self.assertEqual(self.tree.path_history[0], self.tree.root.node_id)
        self.assertEqual(len(self.tree.ghost_nodes), 0)
        self.assertEqual(len(self.tree.node_registry), 1)
        self.assertEqual(self.tree.backend_manager.get_backend_name(), "numpy")

        # Check initialization with different backend
        tree_pytorch = MemorialTree("Root", backend="pytorch")
        self.assertEqual(tree_pytorch.backend_manager.get_backend_name(), "pytorch")

    def test_add_thought(self):
        """Test adding a thought node."""
        # Add a thought to the root
        child = self.tree.add_thought(
            self.tree.root.node_id, "Child Thought", weight=1.5
        )

        # Check node properties
        self.assertEqual(child.content, "Child Thought")
        self.assertEqual(child.weight, 1.5)
        self.assertEqual(child.parent, self.tree.root)

        # Check tree state
        self.assertIn(child, self.tree.root.children)
        self.assertIn(child.node_id, self.tree.node_registry)
        self.assertEqual(self.tree.node_registry[child.node_id], child)

        # Add another level
        grandchild = self.tree.add_thought(child.node_id, "Grandchild Thought")
        self.assertEqual(grandchild.parent, child)
        self.assertIn(grandchild, child.children)

        # Test adding with invalid parent ID
        with self.assertRaises(NodeNotFoundError):
            self.tree.add_thought("invalid-id", "Invalid Child")

    def test_add_ghost_node(self):
        """Test adding a ghost node."""
        # Add a ghost node
        ghost = self.tree.add_ghost_node(
            "Ghost Thought", influence=0.7, visibility=0.2, weight=1.2
        )

        # Check node properties
        self.assertEqual(ghost.content, "Ghost Thought")
        self.assertEqual(ghost.influence, 0.7)
        self.assertEqual(ghost.visibility, 0.2)
        self.assertEqual(ghost.weight, 1.2)

        # Check tree state
        self.assertIn(ghost, self.tree.ghost_nodes)
        self.assertIn(ghost.node_id, self.tree.node_registry)
        self.assertEqual(self.tree.node_registry[ghost.node_id], ghost)

    def test_make_choice(self):
        """Test making a choice in the decision process."""
        # Add some choices
        child1 = self.tree.add_thought(self.tree.root.node_id, "Choice 1")
        child2 = self.tree.add_thought(self.tree.root.node_id, "Choice 2")
        grandchild = self.tree.add_thought(child1.node_id, "Grandchild")

        # Make a choice
        selected = self.tree.make_choice(child1.node_id)

        # Check tree state
        self.assertEqual(selected, child1)
        self.assertEqual(self.tree.current_node, child1)
        self.assertEqual(len(self.tree.path_history), 2)
        self.assertEqual(self.tree.path_history[-1], child1.node_id)
        self.assertEqual(child1.activation_count, 1)

        # Try to make an invalid choice (not a child of current node)
        with self.assertRaises(ValueError):
            self.tree.make_choice(child2.node_id)

        # Make another valid choice
        self.tree.make_choice(grandchild.node_id)
        self.assertEqual(self.tree.current_node, grandchild)
        self.assertEqual(len(self.tree.path_history), 3)

        # Try with non-existent node ID
        with self.assertRaises(NodeNotFoundError):
            self.tree.make_choice("non-existent-id")

    def test_get_current_state(self):
        """Test getting the current state of the decision process."""
        # Add some nodes
        child = self.tree.add_thought(self.tree.root.node_id, "Child")
        self.tree.add_ghost_node("Ghost")

        # Get state at root
        state = self.tree.get_current_state()
        self.assertEqual(state["current_node"], self.tree.root)
        self.assertEqual(state["current_node_id"], self.tree.root.node_id)
        self.assertEqual(state["current_content"], "Root Thought")
        self.assertEqual(len(state["path_history"]), 1)
        self.assertEqual(len(state["available_choices"]), 1)
        self.assertEqual(len(state["ghost_nodes"]), 1)
        self.assertEqual(state["tree_depth"], 0)

        # Make a choice and check state again
        self.tree.make_choice(child.node_id)
        state = self.tree.get_current_state()
        self.assertEqual(state["current_node"], child)
        self.assertEqual(state["tree_depth"], 1)

    def test_get_available_choices(self):
        """Test getting available choices from current node."""
        # Add some choices
        child1 = self.tree.add_thought(self.tree.root.node_id, "Choice 1")
        child2 = self.tree.add_thought(self.tree.root.node_id, "Choice 2")

        # Check available choices at root
        choices = self.tree.get_available_choices()
        self.assertEqual(len(choices), 2)
        self.assertIn(child1, choices)
        self.assertIn(child2, choices)

        # Make a choice and add grandchildren
        self.tree.make_choice(child1.node_id)
        grandchild1 = self.tree.add_thought(child1.node_id, "Grandchild 1")
        grandchild2 = self.tree.add_thought(child1.node_id, "Grandchild 2")

        # Check available choices at child1
        choices = self.tree.get_available_choices()
        self.assertEqual(len(choices), 2)
        self.assertIn(grandchild1, choices)
        self.assertIn(grandchild2, choices)

    def test_get_path_from_root(self):
        """Test getting the path from root to current node."""
        # Create a path
        child = self.tree.add_thought(self.tree.root.node_id, "Child")
        grandchild = self.tree.add_thought(child.node_id, "Grandchild")
        self.tree.make_choice(child.node_id)
        self.tree.make_choice(grandchild.node_id)

        # Get path
        path = self.tree.get_path_from_root()
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0], self.tree.root)
        self.assertEqual(path[1], child)
        self.assertEqual(path[2], grandchild)

    def test_reset_to_root(self):
        """Test resetting to the root node."""
        # Create a path and navigate to it
        child = self.tree.add_thought(self.tree.root.node_id, "Child")
        grandchild = self.tree.add_thought(child.node_id, "Grandchild")
        self.tree.make_choice(child.node_id)
        self.tree.make_choice(grandchild.node_id)

        # Reset to root
        self.tree.reset_to_root()
        self.assertEqual(self.tree.current_node, self.tree.root)
        self.assertEqual(len(self.tree.path_history), 1)
        self.assertEqual(self.tree.path_history[0], self.tree.root.node_id)

    def test_remove_node(self):
        """Test removing a node from the tree."""
        # Add some nodes
        child1 = self.tree.add_thought(self.tree.root.node_id, "Child 1")
        child2 = self.tree.add_thought(self.tree.root.node_id, "Child 2")
        grandchild = self.tree.add_thought(child1.node_id, "Grandchild")

        # Remove a leaf node
        result = self.tree.remove_node(grandchild.node_id)
        self.assertTrue(result)
        self.assertEqual(len(child1.children), 0)
        self.assertNotIn(grandchild.node_id, self.tree.node_registry)

        # Remove a node with children
        result = self.tree.remove_node(child1.node_id)
        self.assertTrue(result)
        self.assertNotIn(child1, self.tree.root.children)
        self.assertNotIn(child1.node_id, self.tree.node_registry)

        # Try to remove root
        with self.assertRaises(ValueError):
            self.tree.remove_node(self.tree.root.node_id)

        # Try to remove current node
        self.tree.make_choice(child2.node_id)
        with self.assertRaises(ValueError):
            self.tree.remove_node(child2.node_id)

        # Try to remove non-existent node
        result = self.tree.remove_node("non-existent-id")
        self.assertFalse(result)

    def test_find_node(self):
        """Test finding a node by ID."""
        # Add some nodes
        child = self.tree.add_thought(self.tree.root.node_id, "Child")

        # Find existing node
        found = self.tree.find_node(child.node_id)
        self.assertEqual(found, child)

        # Try to find non-existent node
        found = self.tree.find_node("non-existent-id")
        self.assertIsNone(found)

    def test_get_all_nodes(self):
        """Test getting all nodes in the tree."""
        # Add some nodes
        child1 = self.tree.add_thought(self.tree.root.node_id, "Child 1")
        child2 = self.tree.add_thought(self.tree.root.node_id, "Child 2")
        grandchild = self.tree.add_thought(child1.node_id, "Grandchild")
        ghost = self.tree.add_ghost_node("Ghost")

        # Get all nodes
        all_nodes = self.tree.get_all_nodes()
        self.assertEqual(len(all_nodes), 5)  # root + 3 thoughts + 1 ghost
        self.assertIn(self.tree.root, all_nodes)
        self.assertIn(child1, all_nodes)
        self.assertIn(child2, all_nodes)
        self.assertIn(grandchild, all_nodes)
        self.assertIn(ghost, all_nodes)

    def test_get_tree_size(self):
        """Test getting the tree size."""
        # Initial size
        self.assertEqual(self.tree.get_tree_size(), 1)  # Just root

        # Add some nodes
        self.tree.add_thought(self.tree.root.node_id, "Child 1")
        self.tree.add_thought(self.tree.root.node_id, "Child 2")
        child3 = self.tree.add_thought(self.tree.root.node_id, "Child 3")
        self.tree.add_thought(child3.node_id, "Grandchild")
        self.tree.add_ghost_node("Ghost")

        # Check size
        self.assertEqual(self.tree.get_tree_size(), 6)  # root + 4 thoughts + 1 ghost

    def test_get_tree_depth(self):
        """Test getting the tree depth."""
        # Initial depth
        self.assertEqual(self.tree.get_tree_depth(), 0)  # Just root

        # Add some nodes
        child1 = self.tree.add_thought(self.tree.root.node_id, "Child 1")
        child2 = self.tree.add_thought(self.tree.root.node_id, "Child 2")
        grandchild = self.tree.add_thought(child1.node_id, "Grandchild")
        great_grandchild = self.tree.add_thought(grandchild.node_id, "Great Grandchild")

        # Check depth
        self.assertEqual(
            self.tree.get_tree_depth(), 3
        )  # root->child1->grandchild->great_grandchild

    def test_active_ghost_nodes(self):
        """Test getting active ghost nodes."""
        # Create a ghost node with a simple activation condition
        ghost = self.tree.add_ghost_node(
            "Ghost", visibility=1.0
        )  # Always visible for testing

        # Add a trigger condition that always activates
        ghost.add_trigger_condition(lambda context: True)

        # Get active ghost nodes
        active_nodes = self.tree._get_active_ghost_nodes()
        self.assertEqual(len(active_nodes), 1)
        self.assertEqual(active_nodes[0], ghost)
        self.assertEqual(ghost.activation_count, 1)


if __name__ == "__main__":
    unittest.main()
