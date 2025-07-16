"""
Example script demonstrating the use of Memorial Tree benchmarking tools.

This script shows how to use the benchmarking and optimization tools
provided by the Memorial Tree package.
"""

import os
import time
import matplotlib.pyplot as plt
from datetime import datetime

from src.memorial_tree.benchmarks import (
    PerformanceBenchmark,
    MemoryOptimizer,
    BackendComparison,
)
from src.memorial_tree.core import MemorialTree


def run_performance_benchmarks():
    """Run performance benchmarks with smaller tree sizes for demonstration."""
    print("\n=== Running Performance Benchmarks ===\n")

    # Create output directory with timestamp
    output_dir = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Create benchmark instance with smaller tree sizes for quick demonstration
    benchmark = PerformanceBenchmark(output_dir)

    # Run tree creation benchmark
    print("\nBenchmarking tree creation...")
    benchmark.benchmark_tree_creation(sizes=[10, 50, 100])

    # Run tree traversal benchmark
    print("\nBenchmarking tree traversal...")
    benchmark.benchmark_tree_traversal(sizes=[10, 50, 100])

    # Run backend operations benchmark
    print("\nBenchmarking backend operations...")
    benchmark.benchmark_backend_operations(sizes=[10, 100, 1000])

    # Run memory usage benchmark
    print("\nBenchmarking memory usage...")
    benchmark.benchmark_memory_usage(sizes=[10, 50, 100])

    # Run visualization benchmark
    print("\nBenchmarking visualization...")
    benchmark.benchmark_visualization(sizes=[10, 50, 100])

    print(f"\nBenchmark results saved to {output_dir}")

    return output_dir


def demonstrate_memory_optimization():
    """Demonstrate memory optimization capabilities."""
    print("\n=== Demonstrating Memory Optimization ===\n")

    # Create a tree with many nodes
    print("Creating a large tree...")
    tree = MemorialTree("Memory Optimization Demo")

    # Add three levels with branching factor of 3
    level1_nodes = []
    for i in range(3):
        node = tree.add_thought(tree.root.node_id, f"Level1-{i}", weight=1.0)
        level1_nodes.append(node)

    level2_nodes = []
    for parent in level1_nodes:
        for i in range(3):
            node = tree.add_thought(parent.node_id, f"Level2-{i}", weight=1.0)
            level2_nodes.append(node)

    for parent in level2_nodes:
        for i in range(3):
            tree.add_thought(parent.node_id, f"Level3-{i}", weight=1.0)

    # Add some ghost nodes
    for i in range(5):
        tree.add_ghost_node(f"Ghost-{i}", influence=0.3)

    # Analyze memory usage before optimization
    print("\nAnalyzing memory usage before optimization...")
    before_analysis = MemoryOptimizer.analyze_memory_usage(tree)
    print(f"Node count: {before_analysis['node_count']}")
    print(f"Total memory: {before_analysis['total_estimated_size_mb']:.2f} MB")

    # Make some choices to create inactive nodes
    print("\nMaking choices to create inactive branches...")
    tree.make_choice(level1_nodes[0].node_id)
    tree.make_choice(level2_nodes[0].node_id)

    # Optimize memory usage
    print("\nOptimizing memory usage...")
    optimization_results = MemoryOptimizer.optimize_tree(tree)

    # Print optimization results
    print(f"Nodes removed: {optimization_results['nodes_removed']}")
    print(f"Memory saved: {optimization_results['memory_saved_mb']:.2f} MB")
    print(
        f"Memory after optimization: {optimization_results['after']['total_estimated_size_mb']:.2f} MB"
    )

    return optimization_results


def compare_backends():
    """Compare performance of different backends."""
    print("\n=== Comparing Backend Performance ===\n")

    # Create output directory with timestamp
    output_dir = f"backend_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Create comparison instance
    comparison = BackendComparison(output_dir)

    # Compare tensor creation with small sizes for demonstration
    print("\nComparing tensor creation...")
    comparison.compare_tensor_creation(sizes=[10, 100, 1000])

    # Compare softmax operation
    print("\nComparing softmax operation...")
    comparison.compare_softmax_operation(sizes=[10, 100, 1000])

    # Compare weight calculation
    print("\nComparing weight calculation...")
    comparison.compare_weight_calculation(sizes=[10, 50, 100])

    # Compare backend switching if multiple backends are available
    if len(comparison.available_backends) >= 2:
        print("\nComparing backend switching...")
        comparison.compare_backend_switching(sizes=[10, 50, 100])

    print(f"\nComparison results saved to {output_dir}")

    return output_dir


if __name__ == "__main__":
    # Run performance benchmarks
    benchmark_dir = run_performance_benchmarks()

    # Demonstrate memory optimization
    optimization_results = demonstrate_memory_optimization()

    # Compare backends
    backend_dir = compare_backends()

    print("\n=== All demonstrations completed ===")
    print(f"Performance benchmark results: {benchmark_dir}")
    print(f"Backend comparison results: {backend_dir}")
