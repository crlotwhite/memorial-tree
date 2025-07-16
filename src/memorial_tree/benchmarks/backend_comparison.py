#!/usr/bin/env python
"""
Backend comparison script for Memorial Tree.

This script provides utilities for comparing the performance of different
numerical computation backends (NumPy, PyTorch, TensorFlow) in Memorial Tree.
"""

import time
import gc
import argparse
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt

from ..core import MemorialTree


class BackendComparison:
    """
    Class for comparing the performance of different backends.

    This class provides methods for benchmarking and comparing the performance
    of different numerical computation backends in Memorial Tree.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize a new BackendComparison.

        Args:
            output_dir (Optional[str]): Directory for saving results and plots.
                                       If None, current directory will be used.
        """
        self.output_dir = output_dir or "."
        os.makedirs(self.output_dir, exist_ok=True)

        # Check available backends
        self.available_backends = ["numpy"]

        try:
            import torch

            self.available_backends.append("pytorch")
        except ImportError:
            print("PyTorch not available")

        try:
            import tensorflow as tf

            self.available_backends.append("tensorflow")
        except ImportError:
            print("TensorFlow not available")

    def compare_tensor_creation(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Compare tensor creation performance across backends.

        Args:
            sizes (List[int]): List of tensor sizes to benchmark.
                              If None, default sizes will be used.

        Returns:
            Dict[str, Any]: Dictionary containing benchmark results.
        """
        if sizes is None:
            sizes = [10, 100, 1000, 10000, 100000, 1000000]

        print("\nComparing tensor creation performance...")

        results = {}

        for backend in self.available_backends:
            print(f"  Testing {backend} backend...")

            # Initialize tree with this backend
            tree = MemorialTree("Benchmark Root", backend=backend)

            creation_times = []

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

                print(f"    Size {size}: {creation_time:.6f} seconds")

                # Clean up
                del tensor
                gc.collect()

            # Store results for this backend
            results[backend] = {"sizes": sizes, "creation_times": creation_times}

            # Clean up
            del tree
            gc.collect()

        # Generate comparison plot
        self._plot_comparison(
            sizes,
            results,
            "creation_times",
            "Tensor Creation Performance",
            "Tensor Size (elements)",
            "Time (seconds)",
            "tensor_creation_comparison.png",
        )

        return results

    def compare_softmax_operation(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Compare softmax operation performance across backends.

        Args:
            sizes (List[int]): List of tensor sizes to benchmark.
                              If None, default sizes will be used.

        Returns:
            Dict[str, Any]: Dictionary containing benchmark results.
        """
        if sizes is None:
            sizes = [10, 100, 1000, 10000, 100000]

        print("\nComparing softmax operation performance...")

        results = {}

        for backend in self.available_backends:
            print(f"  Testing {backend} backend...")

            # Initialize tree with this backend
            tree = MemorialTree("Benchmark Root", backend=backend)

            softmax_times = []

            for size in sizes:
                # Generate random data
                data = np.random.rand(size).tolist()
                tensor = tree.backend_manager.create_tensor(data)

                # Force garbage collection
                gc.collect()

                # Benchmark softmax operation
                start_time = time.time()
                softmax_result = tree.backend_manager.apply_softmax(tensor)
                end_time = time.time()

                softmax_time = end_time - start_time
                softmax_times.append(softmax_time)

                print(f"    Size {size}: {softmax_time:.6f} seconds")

                # Clean up
                del tensor, softmax_result
                gc.collect()

            # Store results for this backend
            results[backend] = {"sizes": sizes, "softmax_times": softmax_times}

            # Clean up
            del tree
            gc.collect()

        # Generate comparison plot
        self._plot_comparison(
            sizes,
            results,
            "softmax_times",
            "Softmax Operation Performance",
            "Tensor Size (elements)",
            "Time (seconds)",
            "softmax_operation_comparison.png",
        )

        return results

    def compare_weight_calculation(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Compare weight calculation performance across backends.

        Args:
            sizes (List[int]): List of tensor sizes to benchmark.
                              If None, default sizes will be used.

        Returns:
            Dict[str, Any]: Dictionary containing benchmark results.
        """
        if sizes is None:
            sizes = [10, 50, 100, 500, 1000]

        print("\nComparing weight calculation performance...")

        results = {}

        for backend in self.available_backends:
            print(f"  Testing {backend} backend...")

            # Initialize tree with this backend
            tree = MemorialTree("Benchmark Root", backend=backend)

            weight_calc_times = []

            for size in sizes:
                # Generate random tensors and factors
                tensors = [
                    tree.backend_manager.create_tensor(np.random.rand(10).tolist())
                    for _ in range(size)
                ]
                factors = np.random.rand(size).tolist()

                # Force garbage collection
                gc.collect()

                # Benchmark weight calculation
                start_time = time.time()
                weight_result = tree.backend_manager.calculate_weights(tensors, factors)
                end_time = time.time()

                weight_calc_time = end_time - start_time
                weight_calc_times.append(weight_calc_time)

                print(f"    Size {size}: {weight_calc_time:.6f} seconds")

                # Clean up
                del tensors, factors, weight_result
                gc.collect()

            # Store results for this backend
            results[backend] = {"sizes": sizes, "weight_calc_times": weight_calc_times}

            # Clean up
            del tree
            gc.collect()

        # Generate comparison plot
        self._plot_comparison(
            sizes,
            results,
            "weight_calc_times",
            "Weight Calculation Performance",
            "Number of Tensors",
            "Time (seconds)",
            "weight_calculation_comparison.png",
        )

        return results

    def compare_backend_switching(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Compare backend switching performance.

        Args:
            sizes (List[int]): List of tree sizes to benchmark.
                              If None, default sizes will be used.

        Returns:
            Dict[str, Any]: Dictionary containing benchmark results.
        """
        if sizes is None:
            sizes = [10, 50, 100, 200, 500]

        # Skip test if fewer than 2 backends are available
        if len(self.available_backends) < 2:
            print("At least 2 backends are required for switching benchmarks")
            return {"error": "Insufficient backends available"}

        print("\nComparing backend switching performance...")

        switch_times = {}

        # Define switching pairs
        switch_pairs = []
        for i, from_backend in enumerate(self.available_backends):
            for to_backend in self.available_backends[i + 1 :]:
                switch_pairs.append((from_backend, to_backend))
                switch_pairs.append((to_backend, from_backend))

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

        # Generate plot
        self._plot_backend_switching_comparison(
            sizes,
            switch_times,
            "Backend Switching Performance",
            "Tree Size (nodes)",
            "Time (seconds)",
            "backend_switching_comparison.png",
        )

        return results

    def run_all_comparisons(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Run all backend comparisons and return combined results.

        Args:
            sizes (List[int]): List of sizes to benchmark.
                              If None, default sizes will be used.

        Returns:
            Dict[str, Any]: Dictionary containing all benchmark results.
        """
        print(f"Running backend comparisons for: {', '.join(self.available_backends)}")

        # Run individual comparisons
        tensor_creation_results = self.compare_tensor_creation(sizes)
        softmax_results = self.compare_softmax_operation(sizes)
        weight_calc_results = self.compare_weight_calculation(sizes)

        # Run backend switching comparison if multiple backends are available
        if len(self.available_backends) >= 2:
            switching_results = self.compare_backend_switching(sizes)
        else:
            switching_results = {"error": "Insufficient backends available"}

        # Combine results
        results = {
            "tensor_creation": tensor_creation_results,
            "softmax_operation": softmax_results,
            "weight_calculation": weight_calc_results,
            "backend_switching": switching_results,
        }

        # Save results to JSON file
        results_path = os.path.join(self.output_dir, "backend_comparison_results.json")

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nAll comparisons completed. Results saved to {results_path}")
        print(f"Plots saved to {self.output_dir}")

        return results

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

        return tree

    def _plot_comparison(
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
        Plot comparison results.

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

        # Use log scale for y-axis if range is large
        y_values = [data[metric] for data in results.values()]
        if y_values:
            all_y = [y for sublist in y_values for y in sublist]
            if all_y and max(all_y) / min(all_y) > 100:
                plt.yscale("log")

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


def main():
    """Run the backend comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare performance of different backends in Memorial Tree"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="backend_comparison_results",
        help="Directory for saving comparison results and plots",
    )

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="Sizes to benchmark (e.g., --sizes 10 100 1000)",
    )

    parser.add_argument(
        "--comparison",
        type=str,
        choices=[
            "all",
            "tensor_creation",
            "softmax",
            "weight_calculation",
            "backend_switching",
        ],
        default="all",
        help="Specific comparison to run (default: all)",
    )

    args = parser.parse_args()

    # Create comparison instance
    comparison = BackendComparison(args.output_dir)

    # Run specified comparison(s)
    if args.comparison == "all":
        comparison.run_all_comparisons(args.sizes)
    elif args.comparison == "tensor_creation":
        comparison.compare_tensor_creation(args.sizes)
    elif args.comparison == "softmax":
        comparison.compare_softmax_operation(args.sizes)
    elif args.comparison == "weight_calculation":
        comparison.compare_weight_calculation(args.sizes)
    elif args.comparison == "backend_switching":
        comparison.compare_backend_switching(args.sizes)


if __name__ == "__main__":
    main()
