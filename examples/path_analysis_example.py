"""
Example demonstrating the path analysis functionality of Memorial Tree.

This example creates a sample decision tree and shows different path analysis visualizations.
"""

import os
import sys
import matplotlib

# Use non-interactive backend if running in non-interactive mode
if not sys.stdout.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add the project root to the Python path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memorial_tree.core.memorial_tree import MemorialTree
from src.memorial_tree.visualization.path_analyzer import PathAnalyzer


def main():
    # Create output directory
    output_dir = os.path.join("examples", "outputs", "path_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Create a sample decision tree
    tree = create_sample_tree()

    # Create path analyzer
    analyzer = PathAnalyzer(output_dir)

    # Example 1: Highlight decision path
    print("Creating decision path highlighting visualization...")
    # Make some choices to create a path
    tree.make_choice(tree.root.children[1].node_id)  # Choose "No, stay in current job"
    tree.make_choice(
        tree.current_node.children[2].node_id
    )  # Choose "Find better work-life balance"

    # Visualize the decision path with highlighting
    fig1 = analyzer.highlight_decision_path(
        tree,
        tree.path_history,
        save_path=os.path.join(output_dir, "highlighted_path.png"),
    )
    plt.close(fig1)

    # Example 2: Ghost node influence on path
    print("Creating ghost node influence visualization...")
    fig2 = analyzer.visualize_ghost_influence_on_path(
        tree,
        tree.path_history,
        save_path=os.path.join(output_dir, "ghost_influence_on_path.png"),
    )
    plt.close(fig2)

    # Example 3: Path evolution animation
    print("Creating path evolution animation...")
    # Create a series of paths representing evolution over time
    path_history = [
        [tree.root.node_id],  # Start with just the root
        [
            tree.root.node_id,
            tree.root.children[1].node_id,
        ],  # Choose "No, stay in current job"
        tree.path_history,  # Final path including "Find better work-life balance"
    ]

    # Create the animation
    fig3 = analyzer.animate_path_evolution(
        tree,
        path_history,
        interval=1000,  # 1 second between frames
        save_path=os.path.join(output_dir, "path_evolution.gif"),
    )
    plt.close(fig3)

    # Example 4: Path with no ghost influence
    print("Creating path visualization without ghost influence...")
    fig4 = analyzer.highlight_decision_path(
        tree,
        tree.path_history,
        show_ghost_influence=False,
        save_path=os.path.join(output_dir, "path_no_ghosts.png"),
    )
    plt.close(fig4)

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
