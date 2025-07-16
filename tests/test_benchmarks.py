"""
Tests for the benchmarking module.

This module contains tests for the benchmarking functionality of Memorial Tree.
"""

import unittest
import os
import tempfile
import shutil
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import gc

from src.memorial_tree.benchmarks.performance_benchmark import (
    PerformanceBenchmark,
    MemoryOptimizer,
)
from src.memorial_tree.benchmarks.backend_comparison import BackendComparison
from src.memorial_tree.core import MemorialTree


class TestPerformanceBenchmark(unittest.TestCase):
    """Test the PerformanceBenchmark class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.benchmark = PerformanceBenchmark(self.test_dir)

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)
        plt.close("all")
        gc.collect()

    def test_benchmark_tree_creation(self):
        """Test benchmarking tree creation."""
        # Use small sizes for quick testing
        sizes = [5, 10]
        results = self.benchmark.benchmark_tree_creation(sizes)

        # Check results structure
        self.assertIn("sizes", results)
        self.assertIn("creation_times", results)
        self.assertIn("nodes_per_second", results)
        self.assertIn("scaling_factor", results)

        # Check that output files were created
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "tree_creation_performance.png"))
        )

    def test_benchmark_tree_traversal(self):
        """Test benchmarking tree traversal."""
        # Use small sizes for quick testing
        sizes = [5, 10]
        results = self.benchmark.benchmark_tree_traversal(sizes)

        # Check results structure
        self.assertIn("sizes", results)
        self.assertIn("traversal_times", results)
        self.assertIn("nodes_per_second", results)
        self.assertIn("scaling_factor", results)

        # Check that output files were created
        self.assertTrue(
            os.path.exists(
                os.path.join(self.test_dir, "tree_traversal_performance.png")
            )
        )

    def test_benchmark_backend_operations(self):
        """Test benchmarking backend operations."""
        # Use small sizes for quick testing
        sizes = [5, 10]
        results = self.benchmark.benchmark_backend_operations(sizes)

        # Check that numpy backend is always present
        self.assertIn("numpy", results)

        # Check that output files were created
        self.assertTrue(
            os.path.exists(
                os.path.join(self.test_dir, "backend_creation_comparison.png")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.test_dir, "backend_softmax_comparison.png")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.test_dir, "backend_weights_comparison.png")
            )
        )

    def test_create_tree_with_size(self):
        """Test creating a tree with a specific size."""
        size = 10
        tree = self.benchmark._create_tree_with_size(size)

        # Check tree size (including root and ghost nodes)
        expected_size = size + 1 + (size // 10)  # nodes + root + ghost nodes
        self.assertEqual(tree.get_tree_size(), expected_size)


class TestMemoryOptimizer(unittest.TestCase):
    """Test the MemoryOptimizer class."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def tearDown(self):
        """Clean up after tests."""
        gc.collect()

    def test_analyze_memory_usage(self):
        """Test analyzing memory usage of a tree."""
        # Create a small tree
        tree = MemorialTree("Memory Test")
        for i in range(5):
            tree.add_thought(tree.root.node_id, f"Child-{i}")

        # Analyze memory usage
        analysis = MemoryOptimizer.analyze_memory_usage(tree)

        # Check analysis structure
        self.assertIn("node_count", analysis)
        self.assertIn("ghost_node_count", analysis)
        self.assertIn("avg_node_size_bytes", analysis)
        self.assertIn("estimated_total_node_size_bytes", analysis)
        self.assertIn("registry_size_bytes", analysis)
        self.assertIn("path_history_size_bytes", analysis)
        self.assertIn("total_estimated_size_bytes", analysis)
        self.assertIn("total_estimated_size_mb", analysis)

        # Check values
        self.assertEqual(analysis["node_count"], 6)  # root + 5 children
        self.assertEqual(analysis["ghost_node_count"], 0)

    def test_optimize_tree(self):
        """Test optimizing memory usage of a tree."""
        # Create a tree with some unused nodes
        tree = MemorialTree("Optimization Test")

        # Add first level
        node1 = tree.add_thought(tree.root.node_id, "Node1")
        node2 = tree.add_thought(tree.root.node_id, "Node2")

        # Add second level to node1
        node1_1 = tree.add_thought(node1.node_id, "Node1-1")
        node1_2 = tree.add_thought(node1.node_id, "Node1-2")

        # Add second level to node2 (will be inactive after we choose node1)
        node2_1 = tree.add_thought(node2.node_id, "Node2-1")
        node2_2 = tree.add_thought(node2.node_id, "Node2-2")

        # Make a choice to make some nodes inactive
        tree.make_choice(node1.node_id)

        # Optimize the tree
        result = MemoryOptimizer.optimize_tree(tree)

        # Check result structure
        self.assertIn("before", result)
        self.assertIn("after", result)
        self.assertIn("nodes_removed", result)
        self.assertIn("memory_saved_bytes", result)
        self.assertIn("memory_saved_mb", result)

        # Check that nodes were removed
        self.assertGreaterEqual(
            result["nodes_removed"], 2
        )  # At least node2_1 and node2_2


class TestBackendComparison(unittest.TestCase):
    """Test the BackendComparison class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.comparison = BackendComparison(self.test_dir)

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)
        plt.close("all")
        gc.collect()

    def test_compare_tensor_creation(self):
        """Test comparing tensor creation performance."""
        # Use small sizes for quick testing
        sizes = [5, 10]
        results = self.comparison.compare_tensor_creation(sizes)

        # Check that numpy backend is always present
        self.assertIn("numpy", results)

        # Check that output files were created
        self.assertTrue(
            os.path.exists(
                os.path.join(self.test_dir, "tensor_creation_comparison.png")
            )
        )

    def test_compare_softmax_operation(self):
        """Test comparing softmax operation performance."""
        # Use small sizes for quick testing
        sizes = [5, 10]
        results = self.comparison.compare_softmax_operation(sizes)

        # Check that numpy backend is always present
        self.assertIn("numpy", results)

        # Check that output files were created
        self.assertTrue(
            os.path.exists(
                os.path.join(self.test_dir, "softmax_operation_comparison.png")
            )
        )

    def test_compare_weight_calculation(self):
        """Test comparing weight calculation performance."""
        # Use small sizes for quick testing
        sizes = [5, 10]
        results = self.comparison.compare_weight_calculation(sizes)

        # Check that numpy backend is always present
        self.assertIn("numpy", results)

        # Check that output files were created
        self.assertTrue(
            os.path.exists(
                os.path.join(self.test_dir, "weight_calculation_comparison.png")
            )
        )


if __name__ == "__main__":
    unittest.main()
