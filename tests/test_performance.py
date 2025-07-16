"""
Performance tests for the Memorial Tree system.

These tests focus on identifying performance bottlenecks and memory leaks
in the Memorial Tree system under various load conditions.
"""

import unittest
import os
import tempfile
import shutil
import time
import gc
import sys
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import memory_usage

from src.memorial_tree.core import MemorialTree
from src.memorial_tree.models import ADHDModel, DepressionModel, AnxietyModel
from src.memorial_tree.visualization import TreeVisualizer


class TestPerformanceBottlenecks(unittest.TestCase):
    """Test performance bottlenecks in the Memorial Tree system."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)
        plt.close("all")
        gc.collect()

    def test_tree_creation_scaling(self):
        """Test how tree creation performance scales with tree size."""
        sizes = [10, 50, 100, 200, 500]
        creation_times = []

        for size in sizes:
            start_time = time.time()
            tree = self._create_tree_with_size(size)
            end_time = time.time()
            creation_times.append(end_time - start_time)

            # Verify tree size
            self.assertEqual(tree.get_tree_size(), size + 1)  # +1 for root

            # Clean up to avoid memory buildup
            del tree
            gc.collect()

        # Print performance results
        print("\nTree Creation Performance:")
        for size, t in zip(sizes, creation_times):
            print(f"  {size} nodes: {t:.4f} seconds ({size/t:.1f} nodes/sec)")

        # Check for non-linear scaling (which would indicate a bottleneck)
        # We expect roughly linear scaling for tree creation
        if len(sizes) > 2:
            # Calculate scaling factor between smallest and largest test
            scaling_factor = (creation_times[-1] / creation_times[0]) / (
                sizes[-1] / sizes[0]
            )
            print(
                f"  Scaling factor: {scaling_factor:.2f} (1.0 is perfect linear scaling)"
            )

            # Allow some overhead, but scaling should be roughly linear
            self.assertLess(
                scaling_factor,
                2.0,
                "Tree creation shows significant non-linear scaling",
            )

    def test_tree_traversal_scaling(self):
        """Test how tree traversal performance scales with tree size."""
        sizes = [10, 50, 100, 200, 500]
        traversal_times = []

        for size in sizes:
            tree = self._create_tree_with_size(size)

            # Measure traversal time
            start_time = time.time()
            all_nodes = tree.get_all_nodes()
            end_time = time.time()
            traversal_times.append(end_time - start_time)

            # Verify traversal
            self.assertEqual(len(all_nodes), size + 1)  # +1 for root

            # Clean up
            del tree
            gc.collect()

        # Print performance results
        print("\nTree Traversal Performance:")
        for size, t in zip(sizes, traversal_times):
            print(f"  {size} nodes: {t:.4f} seconds ({size/t:.1f} nodes/sec)")

        # Check for non-linear scaling
        if len(sizes) > 2:
            scaling_factor = (traversal_times[-1] / traversal_times[0]) / (
                sizes[-1] / sizes[0]
            )
            print(
                f"  Scaling factor: {scaling_factor:.2f} (1.0 is perfect linear scaling)"
            )

            self.assertLess(
                scaling_factor,
                2.0,
                "Tree traversal shows significant non-linear scaling",
            )

    def test_visualization_scaling(self):
        """Test how visualization performance scales with tree size."""
        sizes = [
            10,
            50,
            100,
            200,
        ]  # Smaller sizes for visualization which is more intensive
        visualization_times = []

        visualizer = TreeVisualizer(output_dir=self.test_dir)

        for size in sizes:
            tree = self._create_tree_with_size(size)

            # Measure visualization time
            start_time = time.time()
            fig = visualizer.visualize_tree(
                tree=tree, save_path=os.path.join(self.test_dir, f"tree_{size}.png")
            )
            end_time = time.time()
            visualization_times.append(end_time - start_time)

            # Verify visualization was created
            self.assertTrue(
                os.path.exists(os.path.join(self.test_dir, f"tree_{size}.png"))
            )

            # Clean up
            plt.close(fig)
            del tree
            gc.collect()

        # Print performance results
        print("\nVisualization Performance:")
        for size, t in zip(sizes, visualization_times):
            print(f"  {size} nodes: {t:.4f} seconds ({size/t:.1f} nodes/sec)")

        # Check for non-linear scaling
        if len(sizes) > 2:
            scaling_factor = (visualization_times[-1] / visualization_times[0]) / (
                sizes[-1] / sizes[0]
            )
            print(f"  Scaling factor: {scaling_factor:.2f}")

            # Visualization is expected to scale worse than linearly due to layout algorithms
            # But we still want to identify extreme cases
            self.assertLess(
                scaling_factor, 5.0, "Visualization shows extremely poor scaling"
            )

    def test_model_application_scaling(self):
        """Test how mental health model application scales with tree size."""
        sizes = [10, 50, 100, 200, 500]
        model_times = []

        for size in sizes:
            tree = self._create_tree_with_size(size)
            model = ADHDModel()

            # Measure model application time
            start_time = time.time()
            model.modify_decision_process(tree, tree.current_node)
            end_time = time.time()
            model_times.append(end_time - start_time)

            # Verify model was applied
            self.assertIn("adhd_modified_weights", tree.metadata)

            # Clean up
            del tree
            gc.collect()

        # Print performance results
        print("\nModel Application Performance:")
        for size, t in zip(sizes, model_times):
            print(f"  {size} nodes: {t:.4f} seconds ({size/t:.1f} nodes/sec)")

        # Check for non-linear scaling
        if len(sizes) > 2:
            scaling_factor = (model_times[-1] / model_times[0]) / (sizes[-1] / sizes[0])
            print(
                f"  Scaling factor: {scaling_factor:.2f} (1.0 is perfect linear scaling)"
            )

            self.assertLess(
                scaling_factor,
                2.0,
                "Model application shows significant non-linear scaling",
            )

    def test_backend_switching_performance(self):
        """Test performance of backend switching with different tree sizes."""
        # Skip test if PyTorch or TensorFlow is not available
        try:
            import torch
            import tensorflow as tf
        except ImportError:
            self.skipTest("PyTorch or TensorFlow not available")

        sizes = [10, 50, 100, 200]
        switch_times = {
            "numpy_to_pytorch": [],
            "pytorch_to_tensorflow": [],
            "tensorflow_to_numpy": [],
        }

        for size in sizes:
            # Create trees with different backends
            tree_numpy = self._create_tree_with_size(size, backend="numpy")
            tree_pytorch = self._create_tree_with_size(size, backend="pytorch")
            tree_tensorflow = self._create_tree_with_size(size, backend="tensorflow")

            # Measure switching time: numpy -> pytorch
            start_time = time.time()
            tree_numpy.switch_backend("pytorch")
            end_time = time.time()
            switch_times["numpy_to_pytorch"].append(end_time - start_time)

            # Measure switching time: pytorch -> tensorflow
            start_time = time.time()
            tree_pytorch.switch_backend("tensorflow")
            end_time = time.time()
            switch_times["pytorch_to_tensorflow"].append(end_time - start_time)

            # Measure switching time: tensorflow -> numpy
            start_time = time.time()
            tree_tensorflow.switch_backend("numpy")
            end_time = time.time()
            switch_times["tensorflow_to_numpy"].append(end_time - start_time)

            # Clean up
            del tree_numpy, tree_pytorch, tree_tensorflow
            gc.collect()

        # Print performance results
        print("\nBackend Switching Performance:")
        for switch_type, times in switch_times.items():
            print(f"  {switch_type}:")
            for size, t in zip(sizes, times):
                print(f"    {size} nodes: {t:.4f} seconds")

        # Check for non-linear scaling
        if len(sizes) > 2:
            for switch_type, times in switch_times.items():
                scaling_factor = (times[-1] / times[0]) / (sizes[-1] / sizes[0])
                print(f"  {switch_type} scaling factor: {scaling_factor:.2f}")

                self.assertLess(
                    scaling_factor, 3.0, f"{switch_type} shows poor scaling"
                )

    def _create_tree_with_size(self, size, backend="numpy"):
        """Helper method to create a tree with the specified number of nodes."""
        tree = MemorialTree("Performance Test Root", backend=backend)

        # Keep track of available parent nodes
        parents = [tree.root]
        nodes_created = 0

        # Create nodes until we reach the desired size
        while nodes_created < size:
            # Select a random parent from available parents
            parent = parents[nodes_created % len(parents)]

            # Create a new node
            new_node = tree.add_thought(
                parent_id=parent.node_id, content=f"Node-{nodes_created}", weight=1.0
            )

            # Add the new node to available parents
            parents.append(new_node)

            nodes_created += 1

            # Add some ghost nodes (approximately 10% of total)
            if nodes_created % 10 == 0:
                tree.add_ghost_node(f"Ghost-{nodes_created//10}", influence=0.3)

        return tree


class TestMemoryLeaks(unittest.TestCase):
    """Test for memory leaks in the Memorial Tree system."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)
        plt.close("all")
        gc.collect()

    def test_repeated_tree_creation(self):
        """Test for memory leaks during repeated tree creation and deletion."""

        def create_and_delete_trees():
            for _ in range(50):
                tree = MemorialTree("Memory Test")

                # Add some nodes
                for i in range(20):
                    tree.add_thought(tree.root.node_id, f"Child-{i}")

                # Add some ghost nodes
                for i in range(5):
                    tree.add_ghost_node(f"Ghost-{i}")

                # Force deletion
                del tree
                gc.collect()

        # Measure memory usage
        mem_usage = memory_usage(create_and_delete_trees)

        # Calculate memory stability
        baseline = mem_usage[0]
        peak = max(mem_usage)
        final = mem_usage[-1]

        print(f"\nMemory usage during repeated tree creation:")
        print(f"  Baseline: {baseline:.2f} MiB")
        print(f"  Peak: {peak:.2f} MiB")
        print(f"  Final: {final:.2f} MiB")
        print(f"  Difference (final - baseline): {final - baseline:.2f} MiB")

        # Check for memory leaks - final should be close to baseline
        # Allow for some overhead due to Python's memory management
        self.assertLess(
            final - baseline, 5.0, "Memory leak detected in tree creation/deletion"
        )

    def test_repeated_visualization(self):
        """Test for memory leaks during repeated visualization operations."""

        def create_and_visualize():
            tree = MemorialTree("Visualization Test")

            # Add some nodes
            for i in range(20):
                tree.add_thought(tree.root.node_id, f"Child-{i}")

            visualizer = TreeVisualizer(output_dir=self.test_dir)

            # Perform multiple visualizations
            for i in range(10):
                fig = visualizer.visualize_tree(
                    tree=tree, save_path=os.path.join(self.test_dir, f"viz_{i}.png")
                )
                plt.close(fig)

            # Force cleanup
            del visualizer
            del tree
            gc.collect()

        # Measure memory usage
        mem_usage = memory_usage(create_and_visualize)

        # Calculate memory stability
        baseline = mem_usage[0]
        peak = max(mem_usage)
        final = mem_usage[-1]

        print(f"\nMemory usage during repeated visualization:")
        print(f"  Baseline: {baseline:.2f} MiB")
        print(f"  Peak: {peak:.2f} MiB")
        print(f"  Final: {final:.2f} MiB")
        print(f"  Difference (final - baseline): {final - baseline:.2f} MiB")

        # Check for memory leaks - final should be close to baseline
        # Visualization can use more memory due to matplotlib
        self.assertLess(
            final - baseline, 10.0, "Memory leak detected in visualization operations"
        )

    def test_repeated_model_application(self):
        """Test for memory leaks during repeated model application."""

        def apply_models_repeatedly():
            tree = MemorialTree("Model Test")

            # Add some nodes
            for i in range(20):
                tree.add_thought(tree.root.node_id, f"Child-{i}")

            # Create models
            adhd_model = ADHDModel()
            depression_model = DepressionModel()
            anxiety_model = AnxietyModel()

            # Apply models repeatedly
            for _ in range(10):
                # Reset tree metadata
                tree.metadata = {}

                # Apply each model
                adhd_model.modify_decision_process(tree, tree.current_node)
                depression_model.modify_decision_process(tree, tree.current_node)
                anxiety_model.modify_decision_process(tree, tree.current_node)

            # Force cleanup
            del adhd_model, depression_model, anxiety_model
            del tree
            gc.collect()

        # Measure memory usage
        mem_usage = memory_usage(apply_models_repeatedly)

        # Calculate memory stability
        baseline = mem_usage[0]
        peak = max(mem_usage)
        final = mem_usage[-1]

        print(f"\nMemory usage during repeated model application:")
        print(f"  Baseline: {baseline:.2f} MiB")
        print(f"  Peak: {peak:.2f} MiB")
        print(f"  Final: {final:.2f} MiB")
        print(f"  Difference (final - baseline): {final - baseline:.2f} MiB")

        # Check for memory leaks
        self.assertLess(
            final - baseline, 5.0, "Memory leak detected in model application"
        )

    def test_backend_switching_memory(self):
        """Test for memory leaks during backend switching."""
        # Skip test if PyTorch or TensorFlow is not available
        try:
            import torch
            import tensorflow as tf
        except ImportError:
            self.skipTest("PyTorch or TensorFlow not available")

        def switch_backends_repeatedly():
            tree = MemorialTree("Backend Test", backend="numpy")

            # Add some nodes
            for i in range(20):
                tree.add_thought(tree.root.node_id, f"Child-{i}")

            # Switch backends repeatedly
            backends = ["numpy", "pytorch", "tensorflow"]
            for _ in range(10):
                for backend in backends:
                    tree.switch_backend(backend)

            # Force cleanup
            del tree
            gc.collect()

        # Measure memory usage
        mem_usage = memory_usage(switch_backends_repeatedly)

        # Calculate memory stability
        baseline = mem_usage[0]
        peak = max(mem_usage)
        final = mem_usage[-1]

        print(f"\nMemory usage during backend switching:")
        print(f"  Baseline: {baseline:.2f} MiB")
        print(f"  Peak: {peak:.2f} MiB")
        print(f"  Final: {final:.2f} MiB")
        print(f"  Difference (final - baseline): {final - baseline:.2f} MiB")

        # Check for memory leaks
        # Backend switching can use more memory due to loading libraries
        self.assertLess(
            final - baseline, 15.0, "Memory leak detected in backend switching"
        )


if __name__ == "__main__":
    unittest.main()
