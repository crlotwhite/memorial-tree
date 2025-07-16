"""
Example demonstrating the comparison of different mental illness models in Memorial Tree.

This example creates a common decision-making scenario and applies different mental
health models (ADHD, Depression, Anxiety) to show how each affects the decision process
differently.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np


# Define a simple table formatter if tabulate is not available
def simple_table(data, headers, tablefmt=None):
    """Create a simple text table without external dependencies."""
    result = []

    # Calculate column widths
    col_widths = [
        max(len(str(row[i])) for row in [headers] + data) for i in range(len(headers))
    ]

    # Create header row
    header_row = " | ".join(
        f"{headers[i]:{col_widths[i]}}" for i in range(len(headers))
    )
    result.append(header_row)

    # Create separator
    separator = "-+-".join("-" * width for width in col_widths)
    result.append(separator)

    # Create data rows
    for row in data:
        data_row = " | ".join(f"{str(row[i]):{col_widths[i]}}" for i in range(len(row)))
        result.append(data_row)

    return "\n".join(result)


# Try to import tabulate, use simple_table as fallback
try:
    from tabulate import tabulate
except ImportError:
    tabulate = simple_table

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.memorial_tree.core import MemorialTree
from src.memorial_tree.models import ADHDModel, DepressionModel, AnxietyModel


def create_decision_scenario():
    """Create a common decision scenario for comparing mental illness models."""
    # Create a tree with a common everyday decision
    tree = MemorialTree("How should I handle my work presentation tomorrow?")

    # Add first-level options with neutral weights
    option1 = tree.add_thought(
        tree.root.node_id, "Prepare thoroughly tonight", weight=1.0
    )

    option2 = tree.add_thought(
        tree.root.node_id, "Wing it and rely on my knowledge", weight=1.0
    )

    option3 = tree.add_thought(
        tree.root.node_id, "Ask to postpone the presentation", weight=1.0
    )

    option4 = tree.add_thought(
        tree.root.node_id, "Prepare a basic outline and improvise the rest", weight=1.0
    )

    # Add second-level options to "Prepare thoroughly"
    tree.add_thought(
        option1.node_id, "Create detailed slides with all information", weight=1.0
    )
    tree.add_thought(
        option1.node_id, "Practice the presentation multiple times", weight=1.0
    )
    tree.add_thought(
        option1.node_id, "Research additional material to be extra prepared", weight=1.0
    )

    # Add second-level options to "Wing it"
    tree.add_thought(
        option2.node_id, "Focus on key points I already know well", weight=1.0
    )
    tree.add_thought(option2.node_id, "Use my natural speaking abilities", weight=1.0)
    tree.add_thought(
        option2.node_id, "Rely on my experience with the subject", weight=1.0
    )

    # Add second-level options to "Ask to postpone"
    tree.add_thought(
        option3.node_id, "Email my manager explaining I need more time", weight=1.0
    )
    tree.add_thought(
        option3.node_id, "Suggest someone else present instead", weight=1.0
    )
    tree.add_thought(
        option3.node_id, "Claim I'm feeling unwell tomorrow morning", weight=1.0
    )

    # Add second-level options to "Basic outline"
    tree.add_thought(
        option4.node_id, "Create simple bullet points for structure", weight=1.0
    )
    tree.add_thought(
        option4.node_id, "Prepare opening and closing statements only", weight=1.0
    )
    tree.add_thought(
        option4.node_id, "Make notes on key statistics and facts", weight=1.0
    )

    return tree


def apply_models_and_compare(tree):
    """Apply different mental illness models and compare their effects."""
    # Create models with characteristic parameters
    adhd_model = ADHDModel(
        attention_span=0.3, impulsivity=0.8, distraction_rate=0.6, hyperactivity=0.7
    )

    depression_model = DepressionModel(
        negative_bias=0.7, decision_delay=2.0, energy_level=0.3, rumination=0.6
    )

    anxiety_model = AnxietyModel(
        worry_amplification=0.8,
        risk_aversion=0.9,
        rumination_cycles=3,
        uncertainty_intolerance=0.7,
    )

    # Get original weights for comparison
    choices = tree.get_available_choices()
    original_weights = {node.node_id: node.weight for node in choices}

    # Create copies of the tree for each model
    # (In a real application, we'd use the ModelComparison class,
    # but for clarity in this example, we'll apply models directly)

    # Apply ADHD model
    adhd_model.modify_decision_process(tree, tree.root)
    adhd_weights = tree.metadata.get("adhd_modified_weights", {}).copy()

    # Clear metadata before applying next model
    tree.metadata = {}

    # Apply Depression model
    depression_model.modify_decision_process(tree, tree.root)
    depression_weights = tree.metadata.get("depression_modified_weights", {}).copy()

    # Clear metadata before applying next model
    tree.metadata = {}

    # Apply Anxiety model
    anxiety_model.modify_decision_process(tree, tree.root)
    anxiety_weights = tree.metadata.get("anxiety_modified_weights", {}).copy()

    return {
        "original": original_weights,
        "adhd": adhd_weights,
        "depression": depression_weights,
        "anxiety": anxiety_weights,
        "choices": choices,
    }


def display_comparison_results(results):
    """Display the comparison results in a readable format."""
    choices = results["choices"]

    # Prepare data for tabular display
    table_data = []
    headers = ["Option", "Original", "ADHD", "Depression", "Anxiety"]

    for i, node in enumerate(choices, 1):
        option_name = (
            f"Option {i}: {node.content[:30]}..."
            if len(node.content) > 30
            else f"Option {i}: {node.content}"
        )
        original = results["original"][node.node_id]
        adhd = results["adhd"].get(node.node_id, original)
        depression = results["depression"].get(node.node_id, original)
        anxiety = results["anxiety"].get(node.node_id, original)

        # Calculate percentage changes
        adhd_change = ((adhd - original) / original) * 100 if original > 0 else 0
        depression_change = (
            ((depression - original) / original) * 100 if original > 0 else 0
        )
        anxiety_change = ((anxiety - original) / original) * 100 if original > 0 else 0

        row = [
            option_name,
            f"{original:.2f}",
            f"{adhd:.2f} ({adhd_change:+.1f}%)",
            f"{depression:.2f} ({depression_change:+.1f}%)",
            f"{anxiety:.2f} ({anxiety_change:+.1f}%)",
        ]
        table_data.append(row)

    # Display table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Create visualization if matplotlib is available
    try:
        visualize_comparison(results)
    except Exception as e:
        print(f"\nVisualization error: {e}")
        print("Continuing with text analysis...")


def visualize_comparison(results):
    """Create a bar chart comparing the effects of different models."""
    choices = results["choices"]

    # Prepare data for visualization
    labels = [f"Option {i+1}" for i in range(len(choices))]
    original_values = [results["original"][node.node_id] for node in choices]
    adhd_values = [
        results["adhd"].get(node.node_id, results["original"][node.node_id])
        for node in choices
    ]
    depression_values = [
        results["depression"].get(node.node_id, results["original"][node.node_id])
        for node in choices
    ]
    anxiety_values = [
        results["anxiety"].get(node.node_id, results["original"][node.node_id])
        for node in choices
    ]

    # Set up the figure
    plt.figure(figsize=(12, 8))

    # Set width of bars
    bar_width = 0.2

    # Set position of bars on x axis
    r1 = np.arange(len(labels))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    # Create bars
    plt.bar(r1, original_values, width=bar_width, label="Original", color="gray")
    plt.bar(r2, adhd_values, width=bar_width, label="ADHD", color="orange")
    plt.bar(r3, depression_values, width=bar_width, label="Depression", color="blue")
    plt.bar(r4, anxiety_values, width=bar_width, label="Anxiety", color="red")

    # Add labels and title
    plt.xlabel("Options")
    plt.ylabel("Decision Weight")
    plt.title("Comparison of Mental Illness Models on Decision Weights")
    plt.xticks([r + bar_width * 1.5 for r in range(len(labels))], labels)

    # Add legend
    plt.legend()

    # Save the figure
    output_dir = "examples/outputs"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/mental_illness_comparison.png")
    print(f"\nVisualization saved to {output_dir}/mental_illness_comparison.png")

    # Close the figure to free memory
    plt.close()


def analyze_model_differences():
    """Provide analysis of the key differences between mental illness models."""
    print("\nKey Differences Between Mental Illness Models:")
    print("===========================================")

    print("\n1. ADHD Model Characteristics:")
    print(
        "   - Increased impulsivity: Tendency to choose quick, potentially suboptimal options"
    )
    print(
        "   - Attention deficit: Difficulty focusing on the most important aspects of decisions"
    )
    print(
        "   - Distractibility: May be drawn to novel or stimulating options regardless of relevance"
    )
    print(
        "   - Hyperactivity: Introduces randomness and variability in decision-making"
    )
    print(
        "   - Overall effect: Inconsistent decision patterns with preference for immediate action"
    )

    print("\n2. Depression Model Characteristics:")
    print("   - Negative bias: Increased focus on negative aspects and outcomes")
    print("   - Energy depletion: Overall reduction in motivation across all options")
    print(
        "   - Rumination: Dwelling on past negative experiences, affecting current choices"
    )
    print("   - Decision delay: Slowed decision-making process")
    print("   - Overall effect: Pessimistic outlook with reduced motivation to act")

    print("\n3. Anxiety Model Characteristics:")
    print("   - Worry amplification: Excessive focus on potential risks and threats")
    print("   - Risk aversion: Strong preference for safer, more certain options")
    print("   - Rumination cycles: Overthinking decisions through multiple cycles")
    print("   - Uncertainty intolerance: Avoidance of ambiguous or uncertain options")
    print(
        "   - Overall effect: Cautious decision-making with avoidance of perceived risks"
    )

    print("\nComparative Impact on Decision-Making:")
    print("------------------------------------")
    print("- ADHD tends to make decisions quickly but inconsistently")
    print("- Depression tends to delay decisions and favor negative interpretations")
    print("- Anxiety tends to overthink decisions and avoid perceived risks")

    print("\nReal-World Implications:")
    print("----------------------")
    print(
        "- These models help understand how mental health conditions affect daily choices"
    )
    print("- They can inform therapeutic approaches tailored to specific conditions")
    print(
        "- The models demonstrate why the same situation may be perceived differently"
    )
    print("  by individuals with different mental health conditions")


def main():
    """Run the mental illness models comparison example."""
    print("Memorial Tree - Mental Illness Models Comparison")
    print("=============================================")

    # Create common decision scenario
    tree = create_decision_scenario()
    print(f"Created decision tree with {tree.get_tree_size()} nodes")
    print(f"Root question: '{tree.root.content}'")

    # Display first-level options
    print("\nFirst-level options:")
    for i, node in enumerate(tree.get_available_choices(), 1):
        print(f"{i}. {node.content}")

    # Apply models and compare results
    print("\nApplying mental illness models...")
    results = apply_models_and_compare(tree)

    # Display comparison results
    print("\nComparison of decision weights across models:")
    display_comparison_results(results)

    # Analyze differences between models
    analyze_model_differences()

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
