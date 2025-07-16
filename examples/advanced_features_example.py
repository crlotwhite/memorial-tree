"""
Advanced features example for Memorial Tree.

This example demonstrates advanced features of the Memorial Tree package,
including ghost nodes with custom trigger conditions, multi-backend switching,
and complex decision scenario modeling.
"""

import os
import sys
from pathlib import Path
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to the Python path if running the example directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memorial_tree.core import MemorialTree, GhostNode
from src.memorial_tree.backends import BackendManager
from src.memorial_tree.visualization import TreeVisualizer


def create_complex_decision_tree(backend="numpy"):
    """
    Create a complex decision tree for career planning with ghost nodes.

    Args:
        backend (str): The backend to use for the tree.

    Returns:
        MemorialTree: A complex decision tree with ghost nodes.
    """
    print(f"Creating complex decision tree with {backend} backend...")

    # Create a new tree with the specified backend
    tree = MemorialTree(
        root_content="What career path should I pursue?", backend=backend
    )

    # First level choices (main career paths)
    tech_id = tree.add_thought(
        parent_id=tree.root.node_id, content="Technology career", weight=0.8
    ).node_id

    art_id = tree.add_thought(
        parent_id=tree.root.node_id, content="Creative arts career", weight=0.6
    ).node_id

    business_id = tree.add_thought(
        parent_id=tree.root.node_id, content="Business career", weight=0.7
    ).node_id

    science_id = tree.add_thought(
        parent_id=tree.root.node_id, content="Scientific research career", weight=0.5
    ).node_id

    # Second level for technology path
    dev_id = tree.add_thought(
        parent_id=tech_id, content="Software development", weight=0.9
    ).node_id

    data_id = tree.add_thought(
        parent_id=tech_id, content="Data science", weight=0.8
    ).node_id

    cyber_id = tree.add_thought(
        parent_id=tech_id, content="Cybersecurity", weight=0.7
    ).node_id

    # Third level for software development
    tree.add_thought(parent_id=dev_id, content="Frontend development", weight=0.6)
    tree.add_thought(parent_id=dev_id, content="Backend development", weight=0.7)
    tree.add_thought(parent_id=dev_id, content="Full-stack development", weight=0.8)
    tree.add_thought(parent_id=dev_id, content="Mobile app development", weight=0.5)

    # Third level for data science
    tree.add_thought(parent_id=data_id, content="Machine learning engineer", weight=0.8)
    tree.add_thought(parent_id=data_id, content="Data analyst", weight=0.6)
    tree.add_thought(parent_id=data_id, content="AI researcher", weight=0.7)

    # Third level for cybersecurity
    tree.add_thought(parent_id=cyber_id, content="Security analyst", weight=0.7)
    tree.add_thought(parent_id=cyber_id, content="Penetration tester", weight=0.6)
    tree.add_thought(parent_id=cyber_id, content="Security architect", weight=0.8)

    # Second level for creative arts
    visual_id = tree.add_thought(
        parent_id=art_id, content="Visual arts", weight=0.7
    ).node_id

    perf_id = tree.add_thought(
        parent_id=art_id, content="Performing arts", weight=0.6
    ).node_id

    # Third level for visual arts
    tree.add_thought(parent_id=visual_id, content="Graphic design", weight=0.8)
    tree.add_thought(parent_id=visual_id, content="Photography", weight=0.6)
    tree.add_thought(parent_id=visual_id, content="Animation", weight=0.7)

    # Third level for performing arts
    tree.add_thought(parent_id=perf_id, content="Acting", weight=0.5)
    tree.add_thought(parent_id=perf_id, content="Music", weight=0.7)
    tree.add_thought(parent_id=perf_id, content="Dance", weight=0.6)

    # Second level for business
    mgmt_id = tree.add_thought(
        parent_id=business_id, content="Management", weight=0.8
    ).node_id

    finance_id = tree.add_thought(
        parent_id=business_id, content="Finance", weight=0.7
    ).node_id

    marketing_id = tree.add_thought(
        parent_id=business_id, content="Marketing", weight=0.6
    ).node_id

    # Third level for management
    tree.add_thought(parent_id=mgmt_id, content="Project management", weight=0.7)
    tree.add_thought(parent_id=mgmt_id, content="Human resources", weight=0.6)
    tree.add_thought(parent_id=mgmt_id, content="Operations management", weight=0.8)

    # Third level for finance
    tree.add_thought(parent_id=finance_id, content="Investment banking", weight=0.7)
    tree.add_thought(parent_id=finance_id, content="Financial analysis", weight=0.8)
    tree.add_thought(parent_id=finance_id, content="Accounting", weight=0.6)

    # Third level for marketing
    tree.add_thought(parent_id=marketing_id, content="Digital marketing", weight=0.8)
    tree.add_thought(parent_id=marketing_id, content="Brand management", weight=0.7)
    tree.add_thought(parent_id=marketing_id, content="Market research", weight=0.6)

    # Second level for scientific research
    bio_id = tree.add_thought(
        parent_id=science_id, content="Biological sciences", weight=0.7
    ).node_id

    phys_id = tree.add_thought(
        parent_id=science_id, content="Physical sciences", weight=0.6
    ).node_id

    # Third level for biological sciences
    tree.add_thought(parent_id=bio_id, content="Genetics", weight=0.7)
    tree.add_thought(parent_id=bio_id, content="Neuroscience", weight=0.8)
    tree.add_thought(parent_id=bio_id, content="Ecology", weight=0.6)

    # Third level for physical sciences
    tree.add_thought(parent_id=phys_id, content="Physics", weight=0.7)
    tree.add_thought(parent_id=phys_id, content="Chemistry", weight=0.6)
    tree.add_thought(parent_id=phys_id, content="Astronomy", weight=0.5)

    print(f"Created decision tree with {tree.get_tree_size()} nodes")
    print(f"Tree depth: {tree.get_tree_depth()} levels")

    return tree


def add_ghost_nodes_with_triggers(tree):
    """
    Add ghost nodes with custom trigger conditions to the tree.

    Args:
        tree (MemorialTree): The tree to add ghost nodes to.

    Returns:
        MemorialTree: The tree with ghost nodes added.
    """
    print("Adding ghost nodes with custom trigger conditions...")

    # Add ghost nodes with custom trigger conditions

    # Ghost node 1: Past failure in math
    math_ghost = tree.add_ghost_node(
        content="Past failure in mathematics courses", influence=0.7, visibility=0.3
    )

    # Custom trigger condition: Activate when considering data science or physical sciences
    def math_trigger(context):
        current_node = context.get("current_node")
        if current_node and current_node.content in [
            "Data science",
            "Physical sciences",
            "Machine learning engineer",
        ]:
            return True
        return False

    math_ghost.add_trigger_condition(math_trigger)

    # Ghost node 2: Childhood creativity encouragement
    creativity_ghost = tree.add_ghost_node(
        content="Childhood encouragement in creative activities",
        influence=0.6,
        visibility=0.4,
    )

    # Custom trigger condition: Activate when considering creative paths
    def creativity_trigger(context):
        current_node = context.get("current_node")
        if current_node and any(
            keyword in current_node.content.lower()
            for keyword in ["art", "creativ", "design", "music", "dance", "act"]
        ):
            return True
        return False

    creativity_ghost.add_trigger_condition(creativity_trigger)

    # Ghost node 3: Financial insecurity fear
    financial_ghost = tree.add_ghost_node(
        content="Fear of financial insecurity", influence=0.8, visibility=0.2
    )

    # Custom trigger condition: Activate when considering less financially stable careers
    def financial_trigger(context):
        current_node = context.get("current_node")
        if current_node and any(
            keyword in current_node.content.lower()
            for keyword in ["art", "music", "dance", "act", "ecology", "research"]
        ):
            return True
        return False

    financial_ghost.add_trigger_condition(financial_trigger)

    # Ghost node 4: Parental expectations
    parental_ghost = tree.add_ghost_node(
        content="Parental expectations for prestigious career",
        influence=0.7,
        visibility=0.3,
    )

    # Custom trigger condition: Activate when considering prestigious careers
    def parental_trigger(context):
        current_node = context.get("current_node")
        if current_node and any(
            keyword in current_node.content.lower()
            for keyword in ["doctor", "lawyer", "investment", "finance", "management"]
        ):
            return True
        return False

    parental_ghost.add_trigger_condition(parental_trigger)

    # Ghost node 5: Social media influence
    social_ghost = tree.add_ghost_node(
        content="Social media influence on career perception",
        influence=0.5,
        visibility=0.6,
    )

    # Custom trigger condition: Activate when considering trendy careers
    def social_trigger(context):
        current_node = context.get("current_node")
        if current_node and any(
            keyword in current_node.content.lower()
            for keyword in ["tech", "data", "ai", "machine learning", "digital"]
        ):
            return True
        return False

    social_ghost.add_trigger_condition(social_trigger)

    print(f"Added {len(tree.ghost_nodes)} ghost nodes with custom trigger conditions")

    return tree


def demonstrate_backend_switching(tree):
    """
    Demonstrate switching between different numerical computation backends.

    Args:
        tree (MemorialTree): The tree to use for demonstration.

    Returns:
        dict: Performance metrics for different backends.
    """
    print("\nDemonstrating backend switching...")

    backends = ["numpy", "pytorch", "tensorflow"]
    performance_metrics = {}

    for backend in backends:
        try:
            print(f"\nSwitching to {backend} backend...")
            start_time = time.time()

            # Switch backend
            tree.backend_manager.switch_backend(backend)

            # Verify backend
            current_backend = tree.backend_manager.get_backend_name()
            print(f"Current backend: {current_backend}")

            # Create some test tensors
            data = [0.2, 0.3, 0.5]
            tensor = tree.backend_manager.create_tensor(data)
            print(f"Created tensor with {backend} backend: {tensor}")

            # Apply softmax with different temperatures
            print("Applying softmax with different temperatures:")
            for temp in [0.5, 1.0, 2.0]:
                result = tree.backend_manager.apply_softmax(tensor, temperature=temp)
                print(f"  Temperature {temp}: {result}")

            # Measure performance
            end_time = time.time()
            elapsed = end_time - start_time
            performance_metrics[backend] = elapsed
            print(
                f"Operations with {backend} backend completed in {elapsed:.4f} seconds"
            )

        except Exception as e:
            print(f"Error with {backend} backend: {e}")
            performance_metrics[backend] = None

    return performance_metrics


def simulate_decision_process(tree):
    """
    Simulate a decision process with ghost node influences.

    Args:
        tree (MemorialTree): The tree to use for simulation.

    Returns:
        list: The path taken in the decision process.
    """
    print("\nSimulating decision process with ghost node influences...")

    # Reset tree to root
    tree.reset_to_root()

    # Track the path
    path = [tree.current_node]

    # Make decisions until we reach a leaf node
    while tree.current_node.children:
        print(f"\nCurrent node: '{tree.current_node.content}'")

        # Get available choices
        choices = tree.get_available_choices()
        print(f"Available choices:")
        for i, choice in enumerate(choices):
            print(f"  {i+1}. {choice.content} (weight: {choice.weight:.2f})")

        # Get active ghost nodes
        active_ghosts = tree._get_active_ghost_nodes()
        if active_ghosts:
            print(f"Active ghost nodes:")
            for ghost in active_ghosts:
                print(f"  - {ghost.content} (influence: {ghost.influence:.2f})")

        # Simulate decision making with ghost node influence
        # In a real scenario, this would be more complex
        if active_ghosts:
            # Apply ghost node influences to choice weights
            choice_weights = {choice.node_id: choice.weight for choice in choices}

            for ghost in active_ghosts:
                choice_weights = ghost.apply_influence(choice_weights)
                print(f"  After '{ghost.content}' influence:")
                for i, choice in enumerate(choices):
                    print(
                        f"    {i+1}. {choice.content} (adjusted weight: {choice_weights[choice.node_id]:.2f})"
                    )

            # Select based on adjusted weights
            weights = [choice_weights[choice.node_id] for choice in choices]
            total = sum(weights)
            probabilities = [w / total for w in weights]

            selected_index = np.random.choice(len(choices), p=probabilities)
            selected = choices[selected_index]
        else:
            # Select based on original weights
            weights = [choice.weight for choice in choices]
            total = sum(weights)
            probabilities = [w / total for w in weights]

            selected_index = np.random.choice(len(choices), p=probabilities)
            selected = choices[selected_index]

        print(f"Selected: '{selected.content}'")

        # Make the choice
        tree.make_choice(selected.node_id)
        path.append(tree.current_node)

    print("\nDecision path:")
    for i, node in enumerate(path):
        print(f"  {i+1}. {node.content}")

    return path


def visualize_results(tree, path, performance_metrics, output_dir="examples/outputs"):
    """
    Visualize the results of the simulation.

    Args:
        tree (MemorialTree): The tree used in the simulation.
        path (list): The path taken in the decision process.
        performance_metrics (dict): Performance metrics for different backends.
        output_dir (str): Directory to save visualizations.
    """
    print("\nVisualizing results...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create visualizer
    visualizer = TreeVisualizer(output_dir=output_dir)

    # Visualize the tree with path highlighted
    path_ids = [node.node_id for node in path]
    tree_fig = visualizer.visualize_tree(
        tree=tree,
        highlight_path=path_ids,
        show_ghost_nodes=True,
        save_path=f"{output_dir}/advanced_tree.png",
    )

    # Visualize just the path
    path_fig = visualizer.visualize_path(
        tree=tree, path=path_ids, save_path=f"{output_dir}/advanced_path.png"
    )

    # Visualize ghost node influence
    ghost_fig = visualizer.visualize_ghost_influence(
        tree=tree, save_path=f"{output_dir}/advanced_ghost_influence.png"
    )

    # Visualize backend performance comparison
    if all(metric is not None for metric in performance_metrics.values()):
        plt.figure(figsize=(10, 6))
        backends = list(performance_metrics.keys())
        times = [performance_metrics[b] for b in backends]

        plt.bar(backends, times, color=["blue", "green", "red"])
        plt.title("Backend Performance Comparison")
        plt.xlabel("Backend")
        plt.ylabel("Execution Time (seconds)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        for i, v in enumerate(times):
            plt.text(i, v + 0.01, f"{v:.4f}s", ha="center")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/backend_performance.png")

    print(f"\nVisualizations saved to {output_dir}/")
    print("- Full tree: advanced_tree.png")
    print("- Decision path: advanced_path.png")
    print("- Ghost influence: advanced_ghost_influence.png")
    print("- Backend performance: backend_performance.png")


def main():
    """
    Main function to run the advanced features example.
    """
    print("Memorial Tree - Advanced Features Example")
    print("========================================")

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create complex decision tree with NumPy backend
    tree = create_complex_decision_tree(backend="numpy")

    # Add ghost nodes with custom trigger conditions
    tree = add_ghost_nodes_with_triggers(tree)

    # Demonstrate backend switching
    performance_metrics = demonstrate_backend_switching(tree)

    # Switch back to NumPy for the simulation
    tree.backend_manager.switch_backend("numpy")

    # Simulate decision process with ghost node influences
    path = simulate_decision_process(tree)

    # Visualize results
    visualize_results(tree, path, performance_metrics)

    print("\nAdvanced features example completed successfully!")


if __name__ == "__main__":
    main()
