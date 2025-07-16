"""
Performance Benchmarking module for Memorial Tree.

This module provides tools for benchmarking the performance of different components
and configurations of the Memorial Tree system, including backend comparisons,
memory usage analysis, and scaling tests.
"""

import time
import gc
import os
import json
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage

from ..core import MemorialTree, ThoughtNode
from ..models import ADHDModel, DepressionModel, AnxietyModel
from ..visualization import TreeVisualizer


class PerformanceBenchmark:
    """
    Benchmark class for measuring performance of Memorial Tree components.

    This class provides methods for benchmarking tree operations, backend performance,
    memory usage, and scaling characteristics.

    Attributes:
        results (Dict[str, Any]): Dictionary storing benchmark results.
        output_dir (str): Directory for saving benchmark results and plots.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize a new PerformanceBenchmark.

        Args:
            output_dir (Optional[str]): Directory for saving benchmark results and plots.
                                       If None, a temporary directory will be used.
        """
        self.results: Dict[str, Any] = {}

        if output_dir is None:
            self.output_dir = tempfile.mkdtemp(prefix="memorial_tree_benchmark_")
        else:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)

    def benchmark_tree_creation(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark tree creation performance for different tree sizes.

        Args:
            sizes (List[int]): List of tree sizes to benchmark.
                              If None, default sizes will be used.

        Returns:
            Dict[str, Any]: Dictionary containing benchmark results.
        """
        if sizes is None:
            sizes = [10, 50, 100, 200, 500, 1000]

        print("\nBenchmarking tree creation...")
        creation_times = []
        nodes_per_second = []

        for size in sizes:
            # Force garbage collection before each test
            gc.collect()

            # Measure creation time
            start_time = time.time()
            tree = self._create_tree_with_size(size)
            end_time = time.time()

            elapsed_time = end_time - start_time
            creation_times.append(elapsed_time)
            nodes_per_second.append(size / elapsed_time)

            print(
                f"  {size} nodes: {elapsed_time:.4f} seconds ({size/elapsed_time:.1f} nodes/sec)"
            )

            # Clean up
            del tree
            gc.collect()

        # Calculate scaling factor
        if len(sizes) > 1:
            scaling_factor = (creation_times[-1] / creation_times[0]) / (
                sizes[-1] / sizes[0]
            )
            print(
                f"  Scaling factor: {scaling_factor:.2f} (1.0 is perfect linear scaling)"
            )
        else:
            scaling_factor = 1.0

        # Store results
        results = {
            "sizes": sizes,
            "creation_times": creation_times,
            "nodes_per_second": nodes_per_second,
            "scaling_factor": scaling_factor,
        }

        self.results["tree_creation"] = results

        # Generate plot
        self._plot_benchmark_results(
            sizes,
            nodes_per_second,
            "Tree Creation Performance",
            "Tree Size (nodes)",
            "Performance (nodes/sec)",
            "tree_creation_performance.png",
        )

        return results

    def benchmark_tree_traversal(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark tree traversal performance for different tree sizes.

        Args:
            sizes (List[int]): List of tree sizes to benchmark.
                              If None, default sizes will be used.

        Returns:
            Dict[str, Any]: Dictionary containing benchmark results.
        """
        if sizes is None:
            sizes = [10, 50, 100, 200, 500, 1000]

        print("\nBenchmarking tree traversal...")
        traversal_times = []
        nodes_per_second = []

        for size in sizes:
            # Create tree
            tree = self._create_tree_with_size(size)

            # Force garbage collection
            gc.collect()

            # Measure traversal time
            start_time = time.time()
            all_nodes = tree.get_all_nodes()
            end_time = time.time()

            elapsed_time = end_time - start_time
            traversal_times.append(elapsed_time)
            nodes_per_second.append(size / elapsed_time)

            print(
                f"  {size} nodes: {elapsed_time:.4f} seconds ({size/elapsed_time:.1f} nodes/sec)"
            )

            # Clean up
            del tree, all_nodes
            gc.collect()

        # Calculate scaling factor
        if len(sizes) > 1:
            scaling_factor = (traversal_times[-1] / traversal_times[0]) / (
                sizes[-1] / sizes[0]
            )
            print(
                f"  Scaling factor: {scaling_factor:.2f} (1.0 is perfect linear scaling)"
            )
        else:
            scaling_factor = 1.0

        # Store results
        results = {
            "sizes": sizes,
            "traversal_times": traversal_times,
            "nodes_per_second": nodes_per_second,
            "scaling_factor": scaling_factor,
        }

        self.results["tree_traversal"] = results

        # Generate plot
        self._plot_benchmark_results(
            sizes,
            nodes_per_second,
            "Tree Traversal Performance",
            "Tree Size (nodes)",
            "Performance (nodes/sec)",
            "tree_traversal_performance.png",
        )

        return results

    def benchmark_backend_operations(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark backend operations for different backends and data sizes.

        Args:
            sizes (List[int]): List of data sizes to benchmark.
                              If None, default sizes will be used.

        Returns:
            Dict[str, Any]: Dictionary containing benchmark results.
        """
        if sizes is None:
            sizes = [10, 100, 1000, 10000, 100000]

        backends = ["numpy"]

        # Check if PyTorch is available
        try:
            import torch

            backends.append("pytorch")
        except ImportError:
            print("PyTorch not available, skipping PyTorch backend benchmarks")

        # Check if TensorFlow is available
        try:
            import tensorflow as tf

            backends.append("tensorflow")
        except ImportError:
            print("TensorFlow not available, skipping TensorFlow backend benchmarks")

        print("\nBenchmarking backend operations...")

        results = {}

        for backend in backends:
            print(f"\n  {backend.upper()} Backend:")

            # Initialize tree with this backend
            tree = MemorialTree("Benchmark Root", backend=backend)

            creation_times = []
            softmax_times = []
            weight_calc_times = []

            for size in sizes:
                # Generate random data
                data = np.random.rand(size).tolist()

                # Force garbage collection
                gc.collect()

                # Benchmark tensor creation
                start_time = time.time()
                tensor = tree.backend_manager.create_tensor(data)
                end_time = time.time()
                creation_time = end_time - start_time
                creation_times.append(creation_time)

                # Benchmark softmax operation
                start_time = time.time()
                softmax_result = tree.backend_manager.apply_softmax(tensor)
                end_time = time.time()
                softmax_time = end_time - start_time
                softmax_times.append(softmax_time)

                # Benchmark weight calculation
                tensors = [
                    tree.backend_manager.create_tensor(np.random.rand(10).tolist())
                    for _ in range(size // 10 + 1)
                ]
                factors = np.random.rand(len(tensors)).tolist()

                start_time = time.time()
                weight_result = tree.backend_manager.calculate_weights(tensors, factors)
                end_time = time.time()
                weight_calc_time = end_time - start_time
                weight_calc_times.append(weight_calc_time)

                print(
                    f"    Size {size}: creation={creation_time:.6f}s, softmax={softmax_time:.6f}s, weights={weight_calc_time:.6f}s"
                )

                # Clean up
                del tensor, softmax_result, tensors, factors, weight_result
                gc.collect()

            # Store results for this backend
            results[backend] = {
                "sizes": sizes,
                "creation_times": creation_times,
                "softmax_times": softmax_times,
                "weight_calc_times": weight_calc_times,
            }

            # Clean up
            del tree
            gc.collect()

        self.results["backend_operations"] = results

        # Generate comparison plots
        self._plot_backend_comparison(
            sizes,
            results,
            "creation_times",
            "Tensor Creation Performance",
            "Data Size (elements)",
            "Time (seconds)",
            "backend_creation_comparison.png",
        )

        self._plot_backend_comparison(
            sizes,
            results,
            "softmax_times",
            "Softmax Operation Performance",
            "Data Size (elements)",
            "Time (seconds)",
            "backend_softmax_comparison.png",
        )

        self._plot_backend_comparison(
            sizes,
            results,
            "weight_calc_times",
            "Weight Calculation Performance",
            "Number of Tensors",
            "Time (seconds)",
            "backend_weights_comparison.png",
        )

        return results

    def benchmark_backend_switching(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark backend switching performance for different tree sizes.

        Args:
            sizes (List[int]): List of tree sizes to benchmark.
                              If None, default sizes will be used.

        Returns:
            Dict[str, Any]: Dictionary containing benchmark results.
        """
        if sizes is None:
            sizes = [10, 50, 100, 200, 500]

        # Check which backends are available
        available_backends = ["numpy"]

        try:
            import torch

            available_backends.append("pytorch")
            has_pytorch = True
        except ImportError:
            has_pytorch = False
            print("PyTorch not available, skipping PyTorch backend benchmarks")

        try:
            import tensorflow as tf

            available_backends.append("tensorflow")
            has_tensorflow = True
        except ImportError:
            has_tensorflow = False
            print("TensorFlow not available, skipping TensorFlow backend benchmarks")

        # Skip test if fewer than 2 backends are available
        if len(available_backends) < 2:
            print("At least 2 backends are required for switching benchmarks")
            return {"error": "Insufficient backends available"}

        print("\nBenchmarking backend switching...")

        switch_times = {}

        # Define switching pairs
        switch_pairs = []
        if has_pytorch:
            switch_pairs.append(("numpy", "pytorch"))
        if has_tensorflow:
            switch_pairs.append(("numpy", "tensorflow"))
        if has_pytorch and has_tensorflow:
            switch_pairs.append(("pytorch", "tensorflow"))

        for from_backend, to_backend in switch_pairs:
            pair_name = f"{from_backend}_to_{to_backend}"
            switch_times[pair_name] = []

            print(f"\n  {from_backend} -> {to_backend}:")

            for size in sizes:
                # Create tree with source backend
                tree = self._create_tree_with_size(size, backend=from_backend)

                # Force garbage collection
                gc.collect()

                # Measure switching time
                start_time = time.time()
                tree.switch_backend(to_backend)
                end_time = time.time()

                elapsed_time = end_time - start_time
                switch_times[pair_name].append(elapsed_time)

                print(f"    {size} nodes: {elapsed_time:.4f} seconds")

                # Clean up
                del tree
                gc.collect()

        # Store results
        results = {"sizes": sizes, "switch_times": switch_times}

        self.results["backend_switching"] = results

        # Generate plot
        self._plot_backend_switching_comparison(
            sizes,
            switch_times,
            "Backend Switching Performance",
            "Tree Size (nodes)",
            "Time (seconds)",
            "backend_switching_performance.png",
        )

        return results

    def benchmark_memory_usage(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark memory usage for different tree sizes and operations.

        Args:
            sizes (List[int]): List of tree sizes to benchmark.
                              If None, default sizes will be used.

        Returns:
            Dict[str, Any]: Dictionary containing benchmark results.
        """
        if sizes is None:
            sizes = [10, 50, 100, 200, 500, 1000]

        print("\nBenchmarking memory usage...")

        creation_memory = []
        traversal_memory = []
        visualization_memory = []

        for size in sizes:
            print(f"\n  Tree size: {size} nodes")

            # Measure memory for tree creation
            def create_tree_func():
                tree = self._create_tree_with_size(size)
                return tree

            mem_usage = memory_usage(create_tree_func, max_iterations=1)
            baseline = mem_usage[0]
            peak = max(mem_usage)
            memory_increase = peak - baseline
            creation_memory.append(memory_increase)

            print(f"    Creation memory increase: {memory_increase:.2f} MiB")

            # Get the created tree
            tree = create_tree_func()

            # Measure memory for tree traversal
            def traverse_tree_func():
                all_nodes = tree.get_all_nodes()
                tree.get_tree_depth()
                return all_nodes

            mem_usage = memory_usage(traverse_tree_func, max_iterations=1)
            baseline = mem_usage[0]
            peak = max(mem_usage)
            memory_increase = peak - baseline
            traversal_memory.append(memory_increase)

            print(f"    Traversal memory increase: {memory_increase:.2f} MiB")

            # Measure memory for visualization
            if size <= 500:  # Skip visualization for very large trees

                def visualize_tree_func():
                    visualizer = TreeVisualizer(output_dir=self.output_dir)
                    fig = visualizer.visualize_tree(tree=tree)
                    plt.close(fig)
                    return visualizer

                mem_usage = memory_usage(visualize_tree_func, max_iterations=1)
                baseline = mem_usage[0]
                peak = max(mem_usage)
                memory_increase = peak - baseline
                visualization_memory.append(memory_increase)

                print(f"    Visualization memory increase: {memory_increase:.2f} MiB")
            else:
                visualization_memory.append(None)
                print(f"    Visualization skipped for size {size} (too large)")

            # Clean up
            del tree
            gc.collect()

        # Store results
        results = {
            "sizes": sizes,
            "creation_memory": creation_memory,
            "traversal_memory": traversal_memory,
            "visualization_memory": visualization_memory,
        }

        self.results["memory_usage"] = results

        # Generate plot
        self._plot_memory_usage(
            sizes,
            results,
            "Memory Usage by Operation",
            "Tree Size (nodes)",
            "Memory Usage (MiB)",
            "memory_usage_comparison.png",
        )

        return results

    def benchmark_model_application(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark mental health model application performance for different tree sizes.

        Args:
            sizes (List[int]): List of tree sizes to benchmark.
                              If None, default sizes will be used.

        Returns:
            Dict[str, Any]: Dictionary containing benchmark results.
        """
        if sizes is None:
            sizes = [10, 50, 100, 200, 500, 1000]

        print("\nBenchmarking model application...")

        models = {
            "adhd": ADHDModel(),
            "depression": DepressionModel(),
            "anxiety": AnxietyModel(),
        }

        results = {}

        for model_name, model in models.items():
            print(f"\n  {model_name.upper()} Model:")

            application_times = []
            nodes_per_second = []

            for size in sizes:
                # Create tree
                tree = self._create_tree_with_size(size)

                # Force garbage collection
                gc.collect()

                # Measure model application time
                start_time = time.time()
                model.modify_decision_process(tree, tree.current_node)
                end_time = time.time()

                elapsed_time = end_time - start_time
                application_times.append(elapsed_time)
                nodes_per_second.append(size / elapsed_time)

                print(
                    f"    {size} nodes: {elapsed_time:.4f} seconds ({size/elapsed_time:.1f} nodes/sec)"
                )

                # Clean up
                del tree
                gc.collect()

            # Calculate scaling factor
            if len(sizes) > 1:
                scaling_factor = (application_times[-1] / application_times[0]) / (
                    sizes[-1] / sizes[0]
                )
                print(
                    f"    Scaling factor: {scaling_factor:.2f} (1.0 is perfect linear scaling)"
                )
            else:
                scaling_factor = 1.0

            # Store results for this model
            results[model_name] = {
                "application_times": application_times,
                "nodes_per_second": nodes_per_second,
                "scaling_factor": scaling_factor,
            }

        # Add sizes to results
        results["sizes"] = sizes

        self.results["model_application"] = results

        # Generate plot
        self._plot_model_comparison(
            sizes,
            results,
            "Model Application Performance",
            "Tree Size (nodes)",
            "Performance (nodes/sec)",
            "model_application_performance.png",
        )

        return results

    def benchmark_visualization(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark visualization performance for different tree sizes.

        Args:
            sizes (List[int]): List of tree sizes to benchmark.
                              If None, default sizes will be used.

        Returns:
            Dict[str, Any]: Dictionary containing benchmark results.
        """
        if sizes is None:
            sizes = [10, 50, 100, 200, 500]  # Limit sizes for visualization

        print("\nBenchmarking visualization...")

        visualization_times = []
        nodes_per_second = []

        visualizer = TreeVisualizer(output_dir=self.output_dir)

        for size in sizes:
            # Create tree
            tree = self._create_tree_with_size(size)

            # Force garbage collection
            gc.collect()

            # Measure visualization time
            start_time = time.time()
            fig = visualizer.visualize_tree(
                tree=tree,
                save_path=os.path.join(self.output_dir, f"viz_benchmark_{size}.png"),
            )
            end_time = time.time()

            elapsed_time = end_time - start_time
            visualization_times.append(elapsed_time)
            nodes_per_second.append(size / elapsed_time)

            print(
                f"  {size} nodes: {elapsed_time:.4f} seconds ({size/elapsed_time:.1f} nodes/sec)"
            )

            # Clean up
            plt.close(fig)
            del tree
            gc.collect()

        # Calculate scaling factor
        if len(sizes) > 1:
            scaling_factor = (visualization_times[-1] / visualization_times[0]) / (
                sizes[-1] / sizes[0]
            )
            print(
                f"  Scaling factor: {scaling_factor:.2f} (1.0 is perfect linear scaling)"
            )
        else:
            scaling_factor = 1.0

        # Store results
        results = {
            "sizes": sizes,
            "visualization_times": visualization_times,
            "nodes_per_second": nodes_per_second,
            "scaling_factor": scaling_factor,
        }

        self.results["visualization"] = results

        # Generate plot
        self._plot_benchmark_results(
            sizes,
            nodes_per_second,
            "Visualization Performance",
            "Tree Size (nodes)",
            "Performance (nodes/sec)",
            "visualization_performance.png",
        )

        return results

    def run_all_benchmarks(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Run all benchmarks and return combined results.

        Args:
            sizes (List[int]): List of tree sizes to benchmark.
                              If None, default sizes will be used.

        Returns:
            Dict[str, Any]: Dictionary containing all benchmark results.
        """
        print("Running all Memorial Tree benchmarks...")

        # Run individual benchmarks
        self.benchmark_tree_creation(sizes)
        self.benchmark_tree_traversal(sizes)
        self.benchmark_backend_operations()
        self.benchmark_backend_switching(sizes)
        self.benchmark_memory_usage(sizes)
        self.benchmark_model_application(sizes)
        self.benchmark_visualization(
            sizes[:5] if sizes else None
        )  # Limit sizes for visualization

        # Save results to JSON file
        results_path = os.path.join(self.output_dir, "benchmark_results.json")

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(self.results)

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nAll benchmarks completed. Results saved to {results_path}")
        print(f"Plots saved to {self.output_dir}")

        return self.results

    def _create_tree_with_size(self, size: int, backend: str = "numpy") -> MemorialTree:
        """
        Helper method to create a tree with the specified number of nodes.

        Args:
            size (int): Number of nodes to create (excluding root).
            backend (str): Backend to use for the tree.

        Returns:
            MemorialTree: A tree with the specified number of nodes.
        """
        tree = MemorialTree("Benchmark Root", backend=backend)

        # Keep track of available parent nodes
        parents = [tree.root]
        nodes_created = 0

        # Create nodes until we reach the desired size
        while nodes_created < size:
            # Select a parent from available parents
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

    def _plot_benchmark_results(
        self,
        x_values: List[int],
        y_values: List[float],
        title: str,
        x_label: str,
        y_label: str,
        filename: str,
    ) -> None:
        """
        Plot benchmark results.

        Args:
            x_values (List[int]): X-axis values.
            y_values (List[float]): Y-axis values.
            title (str): Plot title.
            x_label (str): X-axis label.
            y_label (str): Y-axis label.
            filename (str): Output filename.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, "o-", linewidth=2)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def _plot_backend_comparison(
        self,
        x_values: List[int],
        results: Dict[str, Dict[str, List[float]]],
        metric: str,
        title: str,
        x_label: str,
        y_label: str,
        filename: str,
    ) -> None:
        """
        Plot backend comparison results.

        Args:
            x_values (List[int]): X-axis values.
            results (Dict[str, Dict[str, List[float]]]): Results dictionary.
            metric (str): Metric to plot.
            title (str): Plot title.
            x_label (str): X-axis label.
            y_label (str): Y-axis label.
            filename (str): Output filename.
        """
        plt.figure(figsize=(10, 6))

        for backend, data in results.items():
            plt.plot(x_values, data[metric], "o-", linewidth=2, label=backend)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Use log scale for x-axis if range is large
        if max(x_values) / min(x_values) > 100:
            plt.xscale("log")

        # Save plot
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def _plot_backend_switching_comparison(
        self,
        x_values: List[int],
        switch_times: Dict[str, List[float]],
        title: str,
        x_label: str,
        y_label: str,
        filename: str,
    ) -> None:
        """
        Plot backend switching comparison results.

        Args:
            x_values (List[int]): X-axis values.
            switch_times (Dict[str, List[float]]): Switching times dictionary.
            title (str): Plot title.
            x_label (str): X-axis label.
            y_label (str): Y-axis label.
            filename (str): Output filename.
        """
        plt.figure(figsize=(10, 6))

        for pair_name, times in switch_times.items():
            # Make the label more readable
            label = pair_name.replace("_", " → ")
            plt.plot(x_values, times, "o-", linewidth=2, label=label)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def _plot_model_comparison(
        self,
        x_values: List[int],
        results: Dict[str, Any],
        title: str,
        x_label: str,
        y_label: str,
        filename: str,
    ) -> None:
        """
        Plot model comparison results.

        Args:
            x_values (List[int]): X-axis values.
            results (Dict[str, Any]): Results dictionary.
            title (str): Plot title.
            x_label (str): X-axis label.
            y_label (str): Y-axis label.
            filename (str): Output filename.
        """
        plt.figure(figsize=(10, 6))

        for model_name, data in results.items():
            if model_name != "sizes":  # Skip the sizes entry
                plt.plot(
                    x_values,
                    data["nodes_per_second"],
                    "o-",
                    linewidth=2,
                    label=model_name,
                )

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def _plot_memory_usage(
        self,
        x_values: List[int],
        results: Dict[str, Any],
        title: str,
        x_label: str,
        y_label: str,
        filename: str,
    ) -> None:
        """
        Plot memory usage results.

        Args:
            x_values (List[int]): X-axis values.
            results (Dict[str, Any]): Results dictionary.
            title (str): Plot title.
            x_label (str): X-axis label.
            y_label (str): Y-axis label.
            filename (str): Output filename.
        """
        plt.figure(figsize=(10, 6))

        # Filter out None values for visualization
        viz_sizes = []
        viz_memory = []

        for i, mem in enumerate(results["visualization_memory"]):
            if mem is not None:
                viz_sizes.append(x_values[i])
                viz_memory.append(mem)

        plt.plot(
            x_values, results["creation_memory"], "o-", linewidth=2, label="Creation"
        )
        plt.plot(
            x_values, results["traversal_memory"], "o-", linewidth=2, label="Traversal"
        )

        if viz_sizes:
            plt.plot(viz_sizes, viz_memory, "o-", linewidth=2, label="Visualization")

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert objects to JSON serializable format.

        Args:
            obj (Any): Object to convert.

        Returns:
            Any: JSON serializable object.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        else:
            return obj


class MemoryOptimizer:
    """
    Utility class for optimizing memory usage in Memorial Tree.

    This class provides methods for analyzing and optimizing memory usage
    in Memorial Tree operations.
    """

    @staticmethod
    def analyze_memory_usage(tree: MemorialTree) -> Dict[str, Any]:
        """
        Analyze memory usage of a Memorial Tree.

        Args:
            tree (MemorialTree): The tree to analyze.

        Returns:
            Dict[str, Any]: Dictionary containing memory usage analysis.
        """
        import sys

        # Get memory usage of different components
        node_count = tree.get_tree_size()
        ghost_node_count = len(tree.ghost_nodes)

        # Sample nodes for size estimation
        sample_size = min(10, node_count)
        sample_nodes = list(tree.node_registry.values())[:sample_size]

        # Estimate average node size
        total_size = 0
        for node in sample_nodes:
            total_size += sys.getsizeof(node)
            total_size += sys.getsizeof(node.content)
            total_size += sys.getsizeof(node.children)

        avg_node_size = total_size / sample_size if sample_size > 0 else 0
        estimated_total_node_size = avg_node_size * node_count

        # Estimate registry size
        registry_size = sys.getsizeof(tree.node_registry)

        # Estimate path history size
        path_history_size = sys.getsizeof(tree.path_history)

        # Total estimated size
        total_estimated_size = (
            estimated_total_node_size + registry_size + path_history_size
        )

        return {
            "node_count": node_count,
            "ghost_node_count": ghost_node_count,
            "avg_node_size_bytes": avg_node_size,
            "estimated_total_node_size_bytes": estimated_total_node_size,
            "registry_size_bytes": registry_size,
            "path_history_size_bytes": path_history_size,
            "total_estimated_size_bytes": total_estimated_size,
            "total_estimated_size_mb": total_estimated_size / (1024 * 1024),
        }

    @staticmethod
    def optimize_tree(tree: MemorialTree) -> Dict[str, Any]:
        """
        Optimize memory usage of a Memorial Tree.

        Args:
            tree (MemorialTree): The tree to optimize.

        Returns:
            Dict[str, Any]: Dictionary containing optimization results.
        """
        # Analyze before optimization
        before = MemoryOptimizer.analyze_memory_usage(tree)

        # Force garbage collection
        gc.collect()

        # Optimize node registry (remove unused references)
        active_nodes = set()

        # Add current path nodes
        for node_id in tree.path_history:
            active_nodes.add(node_id)

        # Add ghost nodes
        for ghost_node in tree.ghost_nodes:
            active_nodes.add(ghost_node.node_id)

        # Add nodes reachable from current node
        def add_reachable_nodes(node: ThoughtNode) -> None:
            for child in node.children:
                active_nodes.add(child.node_id)
                add_reachable_nodes(child)

        add_reachable_nodes(tree.current_node)

        # Remove inactive nodes from registry
        inactive_nodes = set(tree.node_registry.keys()) - active_nodes
        for node_id in inactive_nodes:
            tree.node_registry.pop(node_id, None)

        # Force garbage collection again
        gc.collect()

        # Analyze after optimization
        after = MemoryOptimizer.analyze_memory_usage(tree)

        return {
            "before": before,
            "after": after,
            "nodes_removed": len(inactive_nodes),
            "memory_saved_bytes": before["total_estimated_size_bytes"]
            - after["total_estimated_size_bytes"],
            "memory_saved_mb": (
                before["total_estimated_size_bytes"]
                - after["total_estimated_size_bytes"]
            )
            / (1024 * 1024),
        }


def run_benchmarks(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run all benchmarks and return results.

    Args:
        output_dir (Optional[str]): Directory for saving benchmark results and plots.
                                   If None, a temporary directory will be used.

    Returns:
        Dict[str, Any]: Dictionary containing all benchmark results.
    """
    benchmark = PerformanceBenchmark(output_dir)
    return benchmark.run_all_benchmarks()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Memorial Tree performance benchmarks"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory for saving benchmark results and plots",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="Tree sizes to benchmark (e.g., --sizes 10 50 100 500)",
    )

    args = parser.parse_args()

    benchmark = PerformanceBenchmark(args.output_dir)
    benchmark.run_all_benchmarks(args.sizes)
