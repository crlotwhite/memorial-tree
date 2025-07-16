"""
Example demonstrating the tree visualization functionality of Memorial Tree.

This example creates a sample decision tree and shows different visualization options.
"""

import os
import sys
import matplotlib

# Use non-interactive backend if running in non-interactive mode
if not sys.stdout.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the Python path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memorial_tree.core.memorial_tree import MemorialTree
from src.memorial_tree.visualization.tree_visualizer import TreeVisualizer


def main():
    # Create output directory
    output_dir = os.path.join("examples", "outputs", "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    # Create a sample decision tree
    tree = create_sample_tree()

    # Create visualizer
    visualizer = TreeVisualizer(output_dir)

    # Example 1: Basic tree visualization
    print("Creating basic tree visualization...")
    fig1 = visualizer.visualize_tree(
        tree, save_path=os.path.join(output_dir, "basic_tree.png")
    )
    plt.close(fig1)

    # Example 2: Tree with highlighted path
    print("Creating tree with highlighted path...")
    # Make some choices to create a path
    tree.make_choice(tree.root.children[1].node_id)  # Choose Option B
    tree.make_choice(tree.current_node.children[0].node_id)  # Choose Option B.1

    # Visualize with the current path highlighted
    fig2 = visualizer.visualize_tree(
        tree,
        highlight_path=tree.path_history,
        save_path=os.path.join(output_dir, "highlighted_path_tree.png"),
    )
    plt.close(fig2)

    # Example 3: Path visualization
    print("Creating path visualization...")
    fig3 = visualizer.visualize_path(
        tree,
        tree.path_history,
        save_path=os.path.join(output_dir, "path_visualization.png"),
    )
    plt.close(fig3)

    # Example 4: Ghost node influence visualization
    print("Creating ghost node influence visualization...")
    fig4 = visualizer.visualize_ghost_influence(
        tree, save_path=os.path.join(output_dir, "ghost_influence.png")
    )
    plt.close(fig4)

    # Example 5: Different layout types
    print("Creating visualizations with different layouts...")
    for layout_type in ["spring", "kamada_kawai", "circular"]:
        fig = visualizer.visualize_tree(
            tree,
            layout_type=layout_type,
            save_path=os.path.join(output_dir, f"{layout_type}_layout.png"),
        )
        plt.close(fig)

    # Example 6: Interactive visualization (only works in interactive mode)
    print("Creating interactive visualization...")
    interactive_visualizer = TreeVisualizer(output_dir, interactive=True)
    fig6 = interactive_visualizer.visualize_tree(tree)
    print("Interactive visualization created. Close the figure window to continue.")
    plt.show()  # This will block until the figure is closed

    print(f"All visualizations saved to {output_dir}")


def create_sample_tree():
    """Create a sample decision tree for visualization."""
    tree = MemorialTree("Should I change my career?")

    # Add first level choices
    yes_choice = tree.add_thought(tree.root.node_id, "Yes, change career", weight=0.7)
    no_choice = tree.add_thought(
        tree.root.node_id, "No, stay in current job", weight=0.3
    )
    maybe_choice = tree.add_thought(
        tree.root.node_id, "Consider part-time transition", weight=0.5
    )

    # Add second level choices to "Yes"
    tech_choice = tree.add_thought(
        yes_choice.node_id, "Switch to tech industry", weight=0.8
    )
    arts_choice = tree.add_thought(yes_choice.node_id, "Pursue arts career", weight=0.4)
    edu_choice = tree.add_thought(yes_choice.node_id, "Go into education", weight=0.6)

    # Add third level choices to "Switch to tech"
    tree.add_thought(tech_choice.node_id, "Learn programming", weight=0.9)
    tree.add_thought(tech_choice.node_id, "Study data science", weight=0.7)
    tree.add_thought(tech_choice.node_id, "Explore UX design", weight=0.5)

    # Add second level choices to "No"
    tree.add_thought(no_choice.node_id, "Ask for promotion", weight=0.6)
    tree.add_thought(no_choice.node_id, "Develop new skills", weight=0.7)
    tree.add_thought(no_choice.node_id, "Find better work-life balance", weight=0.8)

    # Add second level choices to "Maybe"
    tree.add_thought(maybe_choice.node_id, "Start a side business", weight=0.7)
    tree.add_thought(maybe_choice.node_id, "Take evening classes", weight=0.6)
    tree.add_thought(maybe_choice.node_id, "Volunteer in new field", weight=0.5)

    # Add ghost nodes (unconscious influences)
    tree.add_ghost_node(
        "Fear of financial instability", influence=0.7, visibility=0.4, weight=0.8
    )
    tree.add_ghost_node(
        "Childhood dream of being an artist", influence=0.5, visibility=0.3, weight=0.6
    )
    tree.add_ghost_node(
        "Desire for social status", influence=0.6, visibility=0.2, weight=0.7
    )
    tree.add_ghost_node(
        "Burnout from current job", influence=0.8, visibility=0.6, weight=0.9
    )

    return tree


if __name__ == "__main__":
    main()
