"""
Example demonstrating the use of the ADHD model with Memorial Tree.

This example shows how to create a decision tree and apply the ADHD model
to modify the decision-making process.
"""

import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memorial_tree import MemorialTree, ADHDModel


def main():
    """Run the ADHD model example."""
    print("Memorial Tree with ADHD Model Example")
    print("=====================================")

    # Create a new tree
    tree = MemorialTree("일상 계획")
    print(f"Created tree: {tree}")

    # Add some thoughts representing a daily planning scenario
    work = tree.add_thought(tree.root.node_id, "업무 시작하기", weight=2.0)
    print(f"Added thought: {work.content} (weight: {work.weight})")

    social = tree.add_thought(tree.root.node_id, "친구 만나기", weight=1.5)
    print(f"Added thought: {social.content} (weight: {social.weight})")

    hobby = tree.add_thought(tree.root.node_id, "취미 활동하기", weight=1.0)
    print(f"Added thought: {hobby.content} (weight: {hobby.weight})")

    # Add some ghost nodes representing unconscious influences
    anxiety = tree.add_ghost_node("불안감", influence=0.7, visibility=0.2)
    print(f"Added ghost node: {anxiety.content} (influence: {anxiety.influence})")

    # Create an ADHD model with custom parameters
    adhd_model = ADHDModel(
        attention_span=0.3,  # 낮은 주의 지속시간
        impulsivity=0.8,  # 높은 충동성
        distraction_rate=0.6,  # 주의 분산 확률
        hyperactivity=0.7,  # 높은 활동성
    )
    print(f"Created ADHD model: {adhd_model}")

    # Get the current state before applying the ADHD model
    print("\nBefore applying ADHD model:")
    print("---------------------------")
    choices = tree.get_available_choices()
    print("Available choices:")
    for choice in choices:
        print(f"- {choice.content} (weight: {choice.weight})")

    # Apply the ADHD model to the decision process
    print("\nApplying ADHD model...")
    adhd_model.modify_decision_process(tree, tree.current_node)

    # Get the modified weights from the tree's metadata
    print("\nAfter applying ADHD model:")
    print("--------------------------")
    if "adhd_modified_weights" in tree.metadata:
        modified_weights = tree.metadata["adhd_modified_weights"]
        print("Modified decision weights:")
        for node_id, weight in modified_weights.items():
            node = tree.find_node(node_id)
            print(f"- {node.content}: {weight:.2f} (original: {node.weight:.2f})")

        # Analyze the changes
        print("\nADHD Model Effects Analysis:")
        print("---------------------------")
        for node_id, weight in modified_weights.items():
            node = tree.find_node(node_id)
            change = weight - node.weight
            percent = (change / node.weight) * 100
            effect = "increased" if change > 0 else "decreased"
            print(f"- {node.content}: {effect} by {abs(percent):.1f}%")

        # Explain the ADHD characteristics reflected
        print("\nADHD Characteristics Reflected:")
        print("------------------------------")
        print(f"- Attention Deficit: Difficulty maintaining focus on optimal choices")
        print(
            f"- Impulsivity: Tendency to make quick, potentially suboptimal decisions"
        )
        print(f"- Distractibility: Easily distracted by less relevant options")
        print(
            f"- Hyperactivity: Increased randomness and variability in decision-making"
        )

    else:
        print("No ADHD model effects were recorded.")


if __name__ == "__main__":
    main()
