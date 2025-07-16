"""
Command-line interface for Memorial Tree.
"""

import argparse
import sys
from pathlib import Path

from .core.memorial_tree import MemorialTree
from .visualization.tree_visualizer import TreeVisualizer


def visualize_tree():
    """
    Command-line tool to visualize a Memorial Tree from a saved file.
    """
    parser = argparse.ArgumentParser(
        description="Visualize a Memorial Tree from a saved file."
    )
    parser.add_argument(
        "tree_file",
        type=str,
        help="Path to the saved Memorial Tree file (.pkl or .json)",
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output file path for the visualization"
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["png", "svg", "pdf", "html"],
        default="png",
        help="Output format for the visualization",
    )
    parser.add_argument(
        "--highlight-path",
        "-p",
        action="store_true",
        help="Highlight the current decision path",
    )
    parser.add_argument(
        "--show-ghost",
        "-g",
        action="store_true",
        help="Show ghost nodes in the visualization",
    )

    args = parser.parse_args()

    # Check if file exists
    tree_file = Path(args.tree_file)
    if not tree_file.exists():
        print(f"Error: File '{args.tree_file}' not found.", file=sys.stderr)
        sys.exit(1)

    # Load tree from file
    try:
        tree = MemorialTree.load(args.tree_file)
    except Exception as e:
        print(f"Error loading tree file: {e}", file=sys.stderr)
        sys.exit(1)

    # Create visualizer
    visualizer = TreeVisualizer(tree)

    # Generate visualization
    output_path = args.output or f"memorial_tree_visualization.{args.format}"
    try:
        visualizer.visualize(
            output_path=output_path,
            format=args.format,
            highlight_path=args.highlight_path,
            show_ghost_nodes=args.show_ghost,
        )
        print(f"Visualization saved to {output_path}")
    except Exception as e:
        print(f"Error generating visualization: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    visualize_tree()
