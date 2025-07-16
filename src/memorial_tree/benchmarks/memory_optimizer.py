#!/usr/bin/env python
"""
Memory optimization script for Memorial Tree.

This script provides utilities for analyzing and optimizing memory usage
in Memorial Tree instances.
"""

import gc
import sys
import argparse
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from ..core import MemorialTree
from .performance_benchmark import MemoryOptimizer


def analyze_tree(tree: MemorialTree) -> Dict[str, Any]:
    """
    Analyze memory usage of a Memorial Tree.

    Args:
        tree (MemorialTree): The tree to analyze.

    Returns:
        Dict[str, Any]: Dictionary containing memory usage analysis.
    """
    return MemoryOptimizer.analyze_memory_usage(tree)


def optimize_tree(tree: MemorialTree) -> Dict[str, Any]:
    """
    Optimize memory usage of a Memorial Tree.

    Args:
        tree (MemorialTree): The tree to optimize.

    Returns:
        Dict[str, Any]: Dictionary containing optimization results.
    """
    return MemoryOptimizer.optimize_tree(tree)


def print_memory_analysis(analysis: Dict[str, Any]) -> None:
    """
    Print memory analysis results in a readable format.

    Args:
        analysis (Dict[str, Any]): Memory analysis results.
    """
    print("\nMemory Analysis:")
    print(f"  Node count: {analysis['node_count']}")
    print(f"  Ghost node count: {analysis['ghost_node_count']}")
    print(f"  Average node size: {analysis['avg_node_size_bytes']:.2f} bytes")
    print(
        f"  Estimated total node size: {analysis['estimated_total_node_size_bytes']:.2f} bytes "
        f"({analysis['estimated_total_node_size_mb']:.2f} MB)"
    )
    print(
        f"  Registry size: {analysis['registry_size_bytes']:.2f} bytes "
        f"({analysis['registry_size_bytes'] / (1024 * 1024):.2f} MB)"
    )
    print(f"  Path history size: {analysis['path_history_size_bytes']:.2f} bytes")
    print(
        f"  Total estimated size: {analysis['total_estimated_size_bytes']:.2f} bytes "
        f"({analysis['total_estimated_size_mb']:.2f} MB)"
    )


def print_optimization_results(results: Dict[str, Any]) -> None:
    """
    Print optimization results in a readable format.

    Args:
        results (Dict[str, Any]): Optimization results.
    """
    print("\nOptimization Results:")
    print(f"  Nodes removed: {results['nodes_removed']}")
    print(
        f"  Memory saved: {results['memory_saved_bytes']:.2f} bytes "
        f"({results['memory_saved_mb']:.2f} MB)"
    )

    print("\nBefore optimization:")
    print(f"  Node count: {results['before']['node_count']}")
    print(f"  Total size: {results['before']['total_estimated_size_mb']:.2f} MB")

    print("\nAfter optimization:")
    print(f"  Node count: {results['after']['node_count']}")
    print(f"  Total size: {results['after']['total_estimated_size_mb']:.2f} MB")


def main():
    """Run the memory optimization script."""
    parser = argparse.ArgumentParser(
        description="Analyze and optimize memory usage in Memorial Tree"
    )

    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze memory usage without optimizing",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file for saving analysis/optimization results as JSON",
    )

    parser.add_argument(
        "--tree-size",
        type=int,
        default=1000,
        help="Size of the test tree to create (default: 1000)",
    )

    args = parser.parse_args()

    # Create a test tree
    print(f"Creating test tree with {args.tree_size} nodes...")
    tree = create_test_tree(args.tree_size)

    # Force garbage collection
    gc.collect()

    # Analyze memory usage
    print("Analyzing memory usage...")
    analysis = analyze_tree(tree)
    print_memory_analysis(analysis)

    # Optimize if requested
    if not args.analyze_only:
        print("\nOptimizing tree...")
        results = optimize_tree(tree)
        print_optimization_results(results)

        # Save results if output file specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        # Save analysis if output file specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"\nAnalysis saved to {args.output}")


def create_test_tree(size: int) -> MemorialTree:
    """
    Create a test tree with the specified number of nodes.

    Args:
        size (int): Number of nodes to create.

    Returns:
        MemorialTree: A test tree.
    """
    tree = MemorialTree("Memory Test Root")

    # Create a balanced tree with branching factor of 3
    nodes_created = 0
    current_level = [tree.root]

    while nodes_created < size:
        next_level = []

        for parent in current_level:
            # Add up to 3 children per parent
            for i in range(3):
                if nodes_created >= size:
                    break

                child = tree.add_thought(
                    parent_id=parent.node_id,
                    content=f"Node-{nodes_created}",
                    weight=1.0,
                )
                next_level.append(child)
                nodes_created += 1

        current_level = next_level

        # Add some ghost nodes (approximately 5% of total)
        if nodes_created % 20 == 0:
            tree.add_ghost_node(f"Ghost-{nodes_created//20}", influence=0.3)

    return tree


if __name__ == "__main__":
    main()
