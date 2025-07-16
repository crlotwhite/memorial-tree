"""
Basic usage example for Memorial Tree.

This example demonstrates the fundamental usage of the Memorial Tree package,
including creating a tree, adding nodes, making decisions, and visualizing the results.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path if running the example directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memorial_tree.core import MemorialTree
from src.memorial_tree.visualization import TreeVisualizer


def main():
    """
    Example of basic Memorial Tree usage.

    This demonstrates:
    - Creating a new Memorial Tree
    - Adding thought nodes
    - Adding ghost nodes
    - Making decisions
    - Visualizing the tree
    """
    print("Memorial Tree - Basic Usage Example")
    print("-----------------------------------")

    # Create a new thought tree with a root question
    tree = MemorialTree(root_content="Should I go for a walk today?")
    print(f"Created a new tree with root: '{tree.root.content}'")

    # Add thoughts (choices) to the root node
    yes_id = tree.add_thought(
        parent_id=tree.root.node_id, content="Yes, I'll go for a walk", weight=0.7
    ).node_id

    no_id = tree.add_thought(
        parent_id=tree.root.node_id, content="No, I'll stay home", weight=0.3
    ).node_id

    print(f"Added two choices to the root node")

    # Add second-level thoughts to the "Yes" branch
    morning_id = tree.add_thought(
        parent_id=yes_id, content="Go in the morning", weight=0.6
    ).node_id

    evening_id = tree.add_thought(
        parent_id=yes_id, content="Go in the evening", weight=0.4
    ).node_id

    print(f"Added two timing options to the 'Yes' choice")

    # Add second-level thoughts to the "No" branch
    read_id = tree.add_thought(
        parent_id=no_id, content="Read a book instead", weight=0.5
    ).node_id

    watch_id = tree.add_thought(
        parent_id=no_id, content="Watch a movie instead", weight=0.5
    ).node_id

    print(f"Added two alternative activities to the 'No' choice")

    # Add a ghost node (unconscious influence)
    tree.add_ghost_node(
        content="Walking makes me anxious", influence=0.4, visibility=0.2
    )
    print(f"Added a ghost node representing an unconscious influence")

    # Print the current state of the tree
    print("\nCurrent Tree State:")
    print(f"- Tree size: {tree.get_tree_size()} nodes")
    print(f"- Tree depth: {tree.get_tree_depth()} levels")
    print(f"- Current node: '{tree.current_node.content}'")

    # Show available choices
    print("\nAvailable choices:")
    for i, choice in enumerate(tree.get_available_choices()):
        print(f"  {i+1}. {choice.content} (weight: {choice.weight})")

    # Make a decision - choose to go for a walk
    print("\nMaking a decision to go for a walk...")
    tree.make_choice(yes_id)
    print(f"Selected: '{tree.current_node.content}'")

    # Show new available choices
    print("\nNew available choices:")
    for i, choice in enumerate(tree.get_available_choices()):
        print(f"  {i+1}. {choice.content} (weight: {choice.weight})")

    # Make another decision - choose to go in the morning
    print("\nDeciding to go in the morning...")
    tree.make_choice(morning_id)
    print(f"Selected: '{tree.current_node.content}'")

    # Show the path taken
    print("\nDecision path taken:")
    path = tree.get_path_from_root()
    for i, node in enumerate(path):
        print(f"  {i+1}. {node.content}")

    # Visualize the tree
    print("\nVisualizing the tree...")
    visualizer = TreeVisualizer(output_dir="examples/outputs")

    # Create the outputs directory if it doesn't exist
    os.makedirs("examples/outputs", exist_ok=True)

    # Visualize the full tree with the path highlighted
    path_ids = [node.node_id for node in path]
    fig = visualizer.visualize_tree(
        tree=tree,
        highlight_path=path_ids,
        show_ghost_nodes=True,
        save_path="examples/outputs/basic_usage_tree.png",
    )

    # Visualize just the path taken
    path_fig = visualizer.visualize_path(
        tree=tree, path=path_ids, save_path="examples/outputs/basic_usage_path.png"
    )

    # Visualize ghost node influence
    ghost_fig = visualizer.visualize_ghost_influence(
        tree=tree, save_path="examples/outputs/basic_usage_ghost_influence.png"
    )

    print("\nVisualization complete! Images saved to examples/outputs/")
    print("- Full tree: basic_usage_tree.png")
    print("- Decision path: basic_usage_path.png")
    print("- Ghost influence: basic_usage_ghost_influence.png")


if __name__ == "__main__":
    main()
