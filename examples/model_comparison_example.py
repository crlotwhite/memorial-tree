"""
Example script demonstrating model comparison and visualization functionality.

This example creates a decision tree, applies different mental health models,
compares their effects, and visualizes the results.
"""

import os
import matplotlib.pyplot as plt
from datetime import datetime

from memorial_tree.core.memorial_tree import MemorialTree
from memorial_tree.models.adhd_model import ADHDModel
from memorial_tree.models.depression_model import DepressionModel
from memorial_tree.models.anxiety_model import AnxietyModel
from memorial_tree.models.model_comparison import ModelComparison
from memorial_tree.visualization.model_visualizer import ModelVisualizer
from memorial_tree.visualization.statistical_analyzer import StatisticalAnalyzer


def create_sample_tree():
    """Create a sample decision tree for demonstration."""
    # Create tree with root node
    tree = MemorialTree("Should I go to the party tonight?")

    # Add first level choices
    go_option = tree.add_thought(
        tree.root.node_id, "Yes, I'll go to the party", weight=1.0
    )

    stay_option = tree.add_thought(tree.root.node_id, "No, I'll stay home", weight=1.0)

    maybe_option = tree.add_thought(
        tree.root.node_id, "Maybe I'll decide later", weight=1.0
    )

    # Add second level choices to "go" option
    tree.add_thought(go_option.node_id, "I'll go early and leave early", weight=1.0)

    tree.add_thought(
        go_option.node_id, "I'll go late and stay until the end", weight=1.0
    )

    tree.add_thought(go_option.node_id, "I'll go with a friend for support", weight=1.0)

    # Add second level choices to "stay" option
    tree.add_thought(stay_option.node_id, "I'll watch a movie instead", weight=1.0)

    tree.add_thought(stay_option.node_id, "I'll catch up on work", weight=1.0)

    tree.add_thought(stay_option.node_id, "I'll call a friend to chat", weight=1.0)

    # Add second level choices to "maybe" option
    tree.add_thought(
        maybe_option.node_id, "I'll decide based on how I feel later", weight=1.0
    )

    tree.add_thought(
        maybe_option.node_id, "I'll see if any friends are going", weight=1.0
    )

    tree.add_thought(
        maybe_option.node_id, "I'll check the weather forecast", weight=1.0
    )

    # Add ghost nodes (unconscious influences)
    tree.add_ghost_node("Fear of social judgment", influence=0.7, visibility=0.3)

    tree.add_ghost_node("Past negative party experience", influence=0.5, visibility=0.2)

    tree.add_ghost_node("FOMO (Fear of Missing Out)", influence=0.6, visibility=0.4)

    return tree


def main():
    """Run the model comparison example."""
    # Create output directory
    output_dir = "model_comparison_output"
    os.makedirs(output_dir, exist_ok=True)

    # Create sample tree
    tree = create_sample_tree()
    print(
        f"Created decision tree with {tree.get_tree_size()} nodes and {len(tree.ghost_nodes)} ghost nodes"
    )

    # Create model comparison
    comparison = ModelComparison(tree)

    # Add mental health models with different parameters
    comparison.add_model(
        "adhd",
        ADHDModel(
            attention_span=0.3, impulsivity=0.8, distraction_rate=0.6, hyperactivity=0.7
        ),
    )

    comparison.add_model(
        "depression",
        DepressionModel(
            negative_bias=0.7, decision_delay=2.0, energy_level=0.3, rumination=0.6
        ),
    )

    comparison.add_model(
        "anxiety",
        AnxietyModel(
            worry_amplification=0.8,
            risk_aversion=0.9,
            rumination_cycles=3,
            uncertainty_intolerance=0.7,
        ),
    )

    print(f"Added {len(comparison.models)} mental health models for comparison")

    # Define a decision path to analyze
    decision_path = [
        tree.root.node_id,  # Root: "Should I go to the party tonight?"
        tree.root.children[0].node_id,  # First option: "Yes, I'll go to the party"
    ]

    # Run comparison
    results = comparison.run_comparison(decision_path)
    print(
        f"Comparison completed with {len(results['model_effects'])} model effects analyzed"
    )

    # Create visualizer
    visualizer = ModelVisualizer(comparison, output_dir)

    # Visualize weight differences
    print("Generating weight differences visualization...")
    visualizer.visualize_weight_differences()

    # Visualize decision tree
    print("Generating decision tree visualization...")
    visualizer.visualize_decision_tree(tree, decision_path)

    # Visualize model characteristics
    print("Generating model characteristics visualization...")
    visualizer.visualize_model_characteristics()

    # Visualize statistical comparison
    print("Generating statistical comparison visualization...")
    visualizer.visualize_statistical_comparison()

    # Create comprehensive dashboard
    print("Generating comprehensive comparison dashboard...")
    visualizer.create_comparison_dashboard(decision_path)

    # Create statistical analyzer
    analyzer = StatisticalAnalyzer(comparison, output_dir)

    # Analyze weight differences
    print("Analyzing weight differences...")
    weight_analysis = analyzer.analyze_weight_differences()

    # Compare models
    print("Comparing models...")
    model_comparison = analyzer.compare_models()

    # Generate reports in different formats
    print("Generating reports...")
    json_report = analyzer.generate_report("json")
    csv_report = analyzer.generate_report("csv")
    text_report = analyzer.generate_report("text")

    print(f"\nAll outputs saved to '{output_dir}' directory")
    print(f"JSON report: {os.path.basename(json_report)}")
    print(f"CSV report: {os.path.basename(csv_report)}")
    print(f"Text report: {os.path.basename(text_report)}")

    # Get comparison summary
    summary = comparison.get_comparison_summary()

    print("\nComparison Summary:")
    print(f"- Most impactful model: {summary.get('most_impactful_model', 'N/A')}")
    print(f"- Models compared: {', '.join(summary.get('models_compared', []))}")
    print(f"- Decision path length: {summary.get('decision_path_length', 0)}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
