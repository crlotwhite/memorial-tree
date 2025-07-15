"""
Tests for the core module of Memorial Tree.
"""

import unittest
from datetime import datetime

from src.memorial_tree.core import ThoughtNode


class TestThoughtNode(unittest.TestCase):
    """Test cases for the ThoughtNode class."""

    def test_init(self):
        """Test initialization of ThoughtNode."""
        # Test with default values
        node = ThoughtNode("Test thought")
        self.assertEqual(node.content, "Test thought")
        self.assertIsNotNone(node.node_id)
        self.assertEqual(node.weight, 1.0)
        self.assertEqual(node.children, [])
        self.assertIsNone(node.parent)
        self.assertEqual(node.activation_count, 0)
        self.assertIsInstance(node.timestamp, datetime)
        self.assertEqual(node.metadata, {})

        # Test with custom values
        node = ThoughtNode("Custom thought", node_id="custom-id", weight=2.5)
        self.assertEqual(node.content, "Custom thought")
        self.assertEqual(node.node_id, "custom-id")
        self.assertEqual(node.weight, 2.5)

    def test_add_child(self):
        """Test adding a child node."""
        parent = ThoughtNode("Parent")
        child = ThoughtNode("Child")

        # Add child and check relationships
        added_child = parent.add_child(child)
        self.assertEqual(added_child, child)
        self.assertEqual(len(parent.children), 1)
        self.assertEqual(parent.children[0], child)
        self.assertEqual(child.parent, parent)

        # Add another child
        another_child = ThoughtNode("Another child")
        parent.add_child(another_child)
        self.assertEqual(len(parent.children), 2)
        self.assertEqual(parent.children[1], another_child)
        self.assertEqual(another_child.parent, parent)

    def test_circular_reference_prevention(self):
        """Test prevention of circular references."""
        node1 = ThoughtNode("Node 1")
        node2 = ThoughtNode("Node 2")
        node3 = ThoughtNode("Node 3")

        # Create a chain: node1 -> node2 -> node3
        node1.add_child(node2)
        node2.add_child(node3)

        # Attempt to create a circular reference
        with self.assertRaises(ValueError):
            node3.add_child(node1)

        # Also test direct self-reference
        with self.assertRaises(ValueError):
            node1.add_child(node1)

    def test_get_path_to_root(self):
        """Test getting the path from a node to the root."""
        root = ThoughtNode("Root")
        child1 = ThoughtNode("Child 1")
        child2 = ThoughtNode("Child 2")
        grandchild = ThoughtNode("Grandchild")

        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)

        # Test path from grandchild to root
        path = grandchild.get_path_to_root()
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0], grandchild)
        self.assertEqual(path[1], child1)
        self.assertEqual(path[2], root)

        # Test path from child to root
        path = child1.get_path_to_root()
        self.assertEqual(len(path), 2)
        self.assertEqual(path[0], child1)
        self.assertEqual(path[1], root)

        # Test path from root (should be just the root)
        path = root.get_path_to_root()
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0], root)

    def test_calculate_influence(self):
        """Test calculating node influence."""
        node = ThoughtNode("Test", weight=2.0)

        # Initial influence should be equal to weight
        self.assertEqual(node.calculate_influence(), 2.0)

        # Activate the node and check influence increase
        node.activate()
        self.assertEqual(node.activation_count, 1)
        self.assertEqual(
            node.calculate_influence(), 2.0 * 1.1
        )  # weight * (1 + 0.1 * activation_count)

        # Activate again
        node.activate()
        self.assertEqual(node.activation_count, 2)
        self.assertEqual(
            node.calculate_influence(), 2.0 * 1.2
        )  # weight * (1 + 0.1 * activation_count)

    def test_activate(self):
        """Test node activation."""
        node = ThoughtNode("Test")
        original_timestamp = node.timestamp

        # Wait a moment to ensure timestamp changes
        import time

        time.sleep(0.001)

        # Activate and check changes
        node.activate()
        self.assertEqual(node.activation_count, 1)
        self.assertNotEqual(node.timestamp, original_timestamp)

    def test_get_descendants(self):
        """Test getting all descendants of a node."""
        root = ThoughtNode("Root")
        child1 = ThoughtNode("Child 1")
        child2 = ThoughtNode("Child 2")
        grandchild1 = ThoughtNode("Grandchild 1")
        grandchild2 = ThoughtNode("Grandchild 2")

        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild1)
        child2.add_child(grandchild2)

        # Get all descendants of root
        descendants = root.get_descendants()
        self.assertEqual(len(descendants), 4)
        self.assertIn(child1, descendants)
        self.assertIn(child2, descendants)
        self.assertIn(grandchild1, descendants)
        self.assertIn(grandchild2, descendants)

        # Get descendants of child1
        descendants = child1.get_descendants()
        self.assertEqual(len(descendants), 1)
        self.assertIn(grandchild1, descendants)

    def test_find_child_by_id(self):
        """Test finding a child node by ID."""
        root = ThoughtNode("Root")
        child1 = ThoughtNode("Child 1", node_id="child1")
        child2 = ThoughtNode("Child 2", node_id="child2")

        root.add_child(child1)
        root.add_child(child2)

        # Find existing child
        found = root.find_child_by_id("child1")
        self.assertEqual(found, child1)

        # Try to find non-existent child
        found = root.find_child_by_id("non-existent")
        self.assertIsNone(found)

    def test_find_descendant_by_id(self):
        """Test finding a descendant node by ID."""
        root = ThoughtNode("Root")
        child1 = ThoughtNode("Child 1", node_id="child1")
        child2 = ThoughtNode("Child 2", node_id="child2")
        grandchild = ThoughtNode("Grandchild", node_id="grandchild")

        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)

        # Find direct child
        found = root.find_descendant_by_id("child1")
        self.assertEqual(found, child1)

        # Find grandchild
        found = root.find_descendant_by_id("grandchild")
        self.assertEqual(found, grandchild)

        # Try to find non-existent descendant
        found = root.find_descendant_by_id("non-existent")
        self.assertIsNone(found)


if __name__ == "__main__":
    unittest.main()
