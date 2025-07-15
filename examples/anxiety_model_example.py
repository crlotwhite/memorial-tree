"""
Example demonstrating the Anxiety Model in Memorial Tree.

This example shows how the AnxietyModel affects decision-making in a simple scenario.
"""

import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.memorial_tree.core import MemorialTree
from src.memorial_tree.models import AnxietyModel


def main():
    """Run the anxiety model example."""
    print("Memorial Tree - Anxiety Model Example")
    print("=====================================")

    # Create a tree for decision-making
    tree = MemorialTree("Should I go to the party?")
    print(f"Created tree with root: '{tree.root.content}'")

    # Add some options
    option1 = tree.add_thought(
        tree.root.node_id, "Yes, definitely go to the party", weight=1.0
    )
    option2 = tree.add_thought(
        tree.root.node_id, "Maybe go but leave early if uncomfortable", weight=1.0
    )
    option3 = tree.add_thought(
        tree.root.node_id,
        "No, stay home where it's safe and avoid the risk",
        weight=1.0,
    )
    option4 = tree.add_thought(
        tree.root.node_id, "Uncertain, might be fun but could be stressful", weight=1.0
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

    # Create an anxiety model
    anxiety_model = AnxietyModel(
        worry_amplification=0.8,
        risk_aversion=0.9,
        rumination_cycles=3,
        uncertainty_intolerance=0.7,
    )
    print(f"\nCreated anxiety model: {anxiety_model}")

    # Apply the anxiety model to the decision process
    anxiety_model.modify_decision_process(tree, tree.root)

    # Get modified weights
    modified_weights = tree.metadata.get("anxiety_modified_weights", {})
    print("\nAnxiety-modified decision weights:")
    for i, node in enumerate(tree.get_available_choices(), 1):
        original = original_weights[node.node_id]
        modified = modified_weights.get(node.node_id, original)
        change = ((modified - original) / original) * 100 if original > 0 else 0
        print(f"Option {i}: {modified:.2f} ({change:+.1f}%)")

    # Analyze the results
    print("\nAnalysis:")
    print("- The 'safe' option likely has increased weight due to risk aversion")
    print(
        "- The 'uncertain' option likely has decreased weight due to uncertainty intolerance"
    )
    print("- The option with 'risk' likely has decreased weight but might also have")
    print("  increased focus due to worry amplification")
    print("- The weights may show some randomness due to rumination cycles")


if __name__ == "__main__":
    main()
