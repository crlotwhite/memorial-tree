"""
Example demonstrating the Depression Model in Memorial Tree.

This example shows how the DepressionModel affects decision-making in a simple scenario.
"""

import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.memorial_tree.core import MemorialTree
from src.memorial_tree.models import DepressionModel


def main():
    """Run the depression model example."""
    print("Memorial Tree - Depression Model Example")
    print("=======================================")

    # Create a tree for decision-making
    tree = MemorialTree("What should I do today?")
    print(f"Created tree with root: '{tree.root.content}'")

    # Add some options with varying emotional content
    option1 = tree.add_thought(tree.root.node_id, "Go out and meet friends", weight=1.0)
    option2 = tree.add_thought(
        tree.root.node_id, "Work on a personal project", weight=1.0
    )
    option3 = tree.add_thought(
        tree.root.node_id,
        "Stay in bed, it's pointless to try anything today",
        weight=1.0,
    )
    option4 = tree.add_thought(
        tree.root.node_id, "Think about past failures and mistakes", weight=1.0
    )

    print("\nOptions:")
    print(f"1. {option1.content}")
    print(f"2. {option2.content}")
    print(f"3. {option3.content}")
    print(f"4. {option4.content}")

    # Get original weights
    original_weights = {
        node.node_id: node.weight for node in tree.get_available_choices()
    }
    print("\nOriginal decision weights:")
    for i, (node_id, weight) in enumerate(original_weights.items(), 1):
        print(f"Option {i}: {weight:.2f}")

    # Create a depression model
    depression_model = DepressionModel(
        negative_bias=0.7,
        decision_delay=2.0,
        energy_level=0.3,
        rumination=0.6,
    )
    print(f"\nCreated depression model: {depression_model}")

    # Apply the depression model to the decision process
    depression_model.modify_decision_process(tree, tree.root)

    # Get modified weights
    modified_weights = tree.metadata.get("depression_modified_weights", {})
    print("\nDepression-modified decision weights:")
    for i, node in enumerate(tree.get_available_choices(), 1):
        original = original_weights[node.node_id]
        modified = modified_weights.get(node.node_id, original)
        change = ((modified - original) / original) * 100 if original > 0 else 0
        print(f"Option {i}: {modified:.2f} ({change:+.1f}%)")

    # Analyze the results
    print("\nAnalysis of Depression Model Effects:")
    print("-----------------------------------")
    print("1. Negative Bias:")
    print(
        "   - Options containing negative words (like 'pointless', 'failures', 'mistakes')"
    )
    print("     receive increased weight due to negative bias")

    print("\n2. Energy Depletion:")
    print("   - All options have reduced weights due to low energy level")
    print("   - This reflects the general lack of motivation in depression")

    print("\n3. Rumination:")
    print("   - The model may focus on past negative experiences")
    print("   - This can lead to avoiding certain options or fixating on negative ones")

    print("\n4. Decision Delay:")
    print("   - Decision-making is slowed (simulated in the model)")
    print("   - This reflects the difficulty in making choices when depressed")

    print("\nKey Depression Characteristics Modeled:")
    print("- Negative thought patterns")
    print("- Reduced motivation and energy")
    print("- Rumination on negative experiences")
    print("- Slowed cognitive processing")


if __name__ == "__main__":
    main()
