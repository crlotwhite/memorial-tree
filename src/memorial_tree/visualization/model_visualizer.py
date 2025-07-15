"""
Model Visualizer module for Memorial Tree.

This module provides functionality for visualizing comparisons between different
mental health models and their effects on decision-making processes.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import networkx as nx
from datetime import datetime
import os

from ..models.model_comparison import ModelComparison
from ..core.memorial_tree import MemorialTree
from ..core.thought_node import ThoughtNode


class ModelVisualizer:
    """
    Class for visualizing comparisons between different mental health models.

    This class provides methods for creating visualizations that highlight the
    differences between mental health models in decision-making processes.

    Attributes:
        comparison (ModelComparison): The model comparison to visualize.
        output_dir (str): Directory for saving visualization outputs.
        color_map (Dict[str, str]): Color mapping for different models.
    """

    def __init__(
        self, comparison: ModelComparison, output_dir: str = "./visualizations"
    ):
        """
        Initialize a new ModelVisualizer.

        Args:
            comparison (ModelComparison): The model comparison to visualize.
            output_dir (str): Directory for saving visualization outputs.
        """
        self.comparison = comparison
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define color map for different models
        self.color_map = {
            "control": "#808080",  # Gray
            "adhd": "#FF5733",  # Red-Orange
            "depression": "#3366FF",  # Blue
            "anxiety": "#FFCC00",  # Yellow
        }

        # Default figure size
        self.figure_size = (12, 8)

    def visualize_weight_differences(
        self, step: Optional[str] = None, save_path: Optional[str] = None
    ) -> Figure:
        """
        Visualize weight differences between control and models.

        Args:
            step (Optional[str]): Specific step to visualize (e.g., "step_0").
                If None, visualizes overall differences.
            save_path (Optional[str]): Path to save the visualization.
                If None, uses default path in output_dir.

        Returns:
            Figure: The matplotlib figure object.

        Raises:
            ValueError: If comparison results are not available.
        """
        if not self.comparison.comparison_results:
            raise ValueError("No comparison results available. Run comparison first.")

        results = self.comparison.comparison_results

        # Determine which steps to visualize
        steps = (
            [step] if step else [s for s in results["control_weights"] if s != "error"]
        )

        # Create figure
        fig, axes = plt.subplots(len(steps), 1, figsize=self.figure_size, squeeze=False)

        # Process each step
        for i, current_step in enumerate(steps):
            ax = axes[i, 0]

            # Get control weights for this step
            if current_step not in results["control_weights"]:
                continue

            control_weights = results["control_weights"][current_step]

            # Prepare data for plotting
            node_ids = list(control_weights.keys())
            x_positions = np.arange(len(node_ids))
            width = 0.8 / (len(self.comparison.models) + 1)  # Bar width

            # Plot control weights
            ax.bar(
                x_positions - 0.4 + width / 2,
                [control_weights[node_id] for node_id in node_ids],
                width=width,
                label="Control",
                color=self.color_map["control"],
                alpha=0.7,
            )

            # Plot model weights
            for j, (model_name, model) in enumerate(self.comparison.models.items()):
                if model_name not in results["model_effects"]:
                    continue

                model_weights = results["model_effects"][model_name].get(
                    current_step, {}
                )

                # Get weights for each node
                weights = [model_weights.get(node_id, 0) for node_id in node_ids]

                # Plot model weights
                ax.bar(
                    x_positions - 0.4 + (j + 1.5) * width,
                    weights,
                    width=width,
                    label=model_name,
                    color=self.color_map.get(model_name.lower(), "#000000"),
                    alpha=0.7,
                )

            # Set labels and title
            ax.set_xlabel("Node ID")
            ax.set_ylabel("Weight")
            ax.set_title(f"Weight Comparison - {current_step}")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(node_ids, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        elif self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            step_str = f"_{step}" if step else ""
            plt.savefig(
                f"{self.output_dir}/weight_diff{step_str}_{timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )

        return fig

    def visualize_decision_tree(
        self,
        tree: MemorialTree,
        highlight_path: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Visualize a decision tree with optional path highlighting.

        Args:
            tree (MemorialTree): The tree to visualize.
            highlight_path (Optional[List[str]]): List of node IDs to highlight.
            save_path (Optional[str]): Path to save the visualization.

        Returns:
            Figure: The matplotlib figure object.
        """
        # Create directed graph
        G = nx.DiGraph()

        # Add nodes and edges
        nodes = tree.get_all_nodes()

        for node in nodes:
            # Add node with attributes
            G.add_node(
                node.node_id,
                content=node.content,
                weight=node.weight,
                is_ghost=isinstance(node, ThoughtNode),
            )

            # Add edges to children
            for child in node.children:
                G.add_edge(node.node_id, child.node_id, weight=child.weight)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Define node colors
        node_colors = []
        for node_id in G.nodes():
            if highlight_path and node_id in highlight_path:
                node_colors.append("#FF9900")  # Orange for highlighted path
            elif G.nodes[node_id].get("is_ghost", False):
                node_colors.append("#CC99FF")  # Purple for ghost nodes
            else:
                node_colors.append("#66CCFF")  # Light blue for regular nodes

        # Define edge colors
        edge_colors = []
        for u, v in G.edges():
            if highlight_path and u in highlight_path and v in highlight_path:
                edge_colors.append("#FF9900")  # Orange for highlighted path
            else:
                edge_colors.append("#CCCCCC")  # Gray for regular edges

        # Define layout
        pos = nx.spring_layout(G, seed=42)  # Consistent layout

        # Draw the graph
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=True,
            node_color=node_colors,
            edge_color=edge_colors,
            node_size=500,
            font_size=8,
            font_weight="bold",
            ax=ax,
        )

        # Add node content as labels
        node_labels = {
            node_id: (
                G.nodes[node_id]["content"][:20] + "..."
                if len(G.nodes[node_id]["content"]) > 20
                else G.nodes[node_id]["content"]
            )
            for node_id in G.nodes()
        }

        nx.draw_networkx_labels(
            G, pos=pos, labels=node_labels, font_size=6, font_color="black"
        )

        # Set title
        ax.set_title(f"Decision Tree Visualization")

        # Remove axis
        ax.axis("off")

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        elif self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(
                f"{self.output_dir}/decision_tree_{timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )

        return fig

    def visualize_model_characteristics(
        self, save_path: Optional[str] = None
    ) -> Figure:
        """
        Visualize the characteristic parameters of each model.

        Args:
            save_path (Optional[str]): Path to save the visualization.

        Returns:
            Figure: The matplotlib figure object.
        """
        # Get model characteristics
        characteristics = self.comparison.get_model_characteristics()

        if not characteristics:
            raise ValueError("No model characteristics available")

        # Create figure
        fig, axes = plt.subplots(
            len(characteristics), 1, figsize=self.figure_size, squeeze=False
        )

        # Process each model
        for i, (model_name, params) in enumerate(characteristics.items()):
            ax = axes[i, 0]

            # Prepare data for plotting
            param_names = list(params.keys())
            param_values = list(params.values())
            x_positions = np.arange(len(param_names))

            # Plot parameters
            bars = ax.bar(
                x_positions,
                param_values,
                color=self.color_map.get(model_name.lower(), "#000000"),
                alpha=0.7,
            )

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.02,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            # Set labels and title
            ax.set_xlabel("Parameter")
            ax.set_ylabel("Value")
            ax.set_title(f"{model_name} Characteristics")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(param_names, rotation=45, ha="right")
            ax.set_ylim(0, 1.1 * max(param_values))
            ax.grid(True, linestyle="--", alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        elif self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(
                f"{self.output_dir}/model_characteristics_{timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )

        return fig

    def visualize_statistical_comparison(
        self, metric: str = "overall_mean_abs_diff", save_path: Optional[str] = None
    ) -> Figure:
        """
        Visualize statistical comparison between models.

        Args:
            metric (str): The metric to visualize.
            save_path (Optional[str]): Path to save the visualization.

        Returns:
            Figure: The matplotlib figure object.

        Raises:
            ValueError: If comparison results are not available.
        """
        if not self.comparison.comparison_results:
            raise ValueError("No comparison results available. Run comparison first.")

        results = self.comparison.comparison_results

        if "statistical_metrics" not in results:
            raise ValueError("No statistical metrics available in comparison results")

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Prepare data for plotting
        model_names = []
        metric_values = []

        for model_name, metrics in results["statistical_metrics"].items():
            if metric in metrics:
                model_names.append(model_name)
                metric_values.append(metrics[metric])

        if not model_names:
            raise ValueError(f"No data available for metric '{metric}'")

        # Plot data
        x_positions = np.arange(len(model_names))
        bars = ax.bar(
            x_positions,
            metric_values,
            color=[self.color_map.get(name.lower(), "#000000") for name in model_names],
            alpha=0.7,
        )

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02 * max(metric_values),
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Set labels and title
        ax.set_xlabel("Model")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Model Comparison - {metric.replace('_', ' ').title()}")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.grid(True, linestyle="--", alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        elif self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(
                f"{self.output_dir}/stat_comparison_{metric}_{timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )

        return fig

    def create_comparison_dashboard(
        self, decision_path: List[str], save_path: Optional[str] = None
    ) -> Figure:
        """
        Create a comprehensive dashboard visualizing model comparisons.

        Args:
            decision_path (List[str]): The decision path used for comparison.
            save_path (Optional[str]): Path to save the visualization.

        Returns:
            Figure: The matplotlib figure object.

        Raises:
            ValueError: If comparison results are not available.
        """
        if not self.comparison.comparison_results:
            # Run comparison if not already run
            self.comparison.run_comparison(decision_path)

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # Define grid layout
        gs = fig.add_gridspec(3, 3)

        # 1. Model characteristics (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_model_characteristics_radar(ax1)

        # 2. Weight differences overview (top center)
        ax2 = fig.add_subplot(gs[0, 1:])
        self._plot_weight_differences_overview(ax2)

        # 3. Statistical metrics comparison (middle row)
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_statistical_metrics(ax3)

        # 4. Decision path visualization (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_decision_path(ax4, decision_path)

        # 5. Impact summary (bottom right)
        ax5 = fig.add_subplot(gs[2, 1:])
        self._plot_impact_summary(ax5)

        # Set overall title
        fig.suptitle("Mental Health Model Comparison Dashboard", fontsize=16)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        elif self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(
                f"{self.output_dir}/comparison_dashboard_{timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )

        return fig

    def _plot_model_characteristics_radar(self, ax: Axes) -> None:
        """
        Plot model characteristics as radar charts.

        Args:
            ax (Axes): The matplotlib axes to plot on.
        """
        characteristics = self.comparison.get_model_characteristics()

        if not characteristics:
            ax.text(
                0.5, 0.5, "No model characteristics available", ha="center", va="center"
            )
            return

        # Use the first model to determine categories
        first_model = next(iter(characteristics.values()))
        categories = list(first_model.keys())

        # Number of categories
        N = len(categories)

        # Create angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Initialize radar chart
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw category labels on axis
        plt.xticks(angles[:-1], categories, size=8)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], size=7)
        plt.ylim(0, 1)

        # Plot each model
        for model_name, params in characteristics.items():
            values = list(params.values())
            values += values[:1]  # Close the loop

            # Plot values
            ax.plot(
                angles,
                values,
                linewidth=1,
                linestyle="solid",
                label=model_name,
                color=self.color_map.get(model_name.lower(), "#000000"),
            )
            ax.fill(
                angles,
                values,
                alpha=0.1,
                color=self.color_map.get(model_name.lower(), "#000000"),
            )

        # Add legend
        ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        ax.set_title("Model Characteristics", size=10)

    def _plot_weight_differences_overview(self, ax: Axes) -> None:
        """
        Plot overview of weight differences between models.

        Args:
            ax (Axes): The matplotlib axes to plot on.
        """
        if not self.comparison.comparison_results:
            ax.text(
                0.5, 0.5, "No comparison results available", ha="center", va="center"
            )
            return

        results = self.comparison.comparison_results

        if "weight_differences" not in results:
            ax.text(
                0.5, 0.5, "No weight differences available", ha="center", va="center"
            )
            return

        # Prepare data for heatmap
        model_names = list(results["weight_differences"].keys())
        steps = []
        for model_diffs in results["weight_differences"].values():
            steps.extend(model_diffs.keys())
        steps = sorted(set(steps))

        # Create data matrix
        data = np.zeros((len(model_names), len(steps)))

        for i, model_name in enumerate(model_names):
            model_diffs = results["weight_differences"].get(model_name, {})
            for j, step in enumerate(steps):
                if step in model_diffs:
                    # Calculate average absolute difference for this step
                    step_diffs = model_diffs[step]
                    abs_diffs = [d["absolute_diff"] for d in step_diffs.values()]
                    data[i, j] = np.mean(abs_diffs) if abs_diffs else 0

        # Plot heatmap
        im = ax.imshow(data, cmap="YlOrRd")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(
            "Average Absolute Weight Difference", rotation=-90, va="bottom"
        )

        # Set ticks and labels
        ax.set_xticks(np.arange(len(steps)))
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels(steps)
        ax.set_yticklabels(model_names)

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        for i in range(len(model_names)):
            for j in range(len(steps)):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black" if data[i, j] < 0.5 else "white",
                )

        ax.set_title("Weight Differences Overview")
        ax.set_xlabel("Decision Steps")
        ax.set_ylabel("Models")

    def _plot_statistical_metrics(self, ax: Axes) -> None:
        """
        Plot statistical metrics comparison between models.

        Args:
            ax (Axes): The matplotlib axes to plot on.
        """
        if not self.comparison.comparison_results:
            ax.text(
                0.5, 0.5, "No comparison results available", ha="center", va="center"
            )
            return

        results = self.comparison.comparison_results

        if "statistical_metrics" not in results:
            ax.text(
                0.5, 0.5, "No statistical metrics available", ha="center", va="center"
            )
            return

        # Select key metrics to display
        key_metrics = [
            "overall_mean_abs_diff",
            "overall_max_abs_diff",
            "overall_mean_rel_diff",
        ]

        # Prepare data
        model_names = list(results["statistical_metrics"].keys())
        metrics_data = []

        for metric in key_metrics:
            metric_values = []
            for model_name in model_names:
                model_metrics = results["statistical_metrics"].get(model_name, {})
                metric_values.append(model_metrics.get(metric, 0))
            metrics_data.append(metric_values)

        # Set width of bars
        barWidth = 0.25

        # Set position of bars on X axis
        r = np.arange(len(model_names))
        positions = [r]
        for i in range(1, len(key_metrics)):
            positions.append([x + barWidth for x in positions[i - 1]])

        # Create bars
        for i, (metric, data) in enumerate(zip(key_metrics, metrics_data)):
            ax.bar(
                positions[i],
                data,
                width=barWidth,
                label=metric.replace("_", " ").title(),
                alpha=0.7,
            )

        # Add labels and title
        ax.set_xlabel("Models")
        ax.set_ylabel("Metric Value")
        ax.set_title("Statistical Metrics Comparison")
        ax.set_xticks([r + barWidth for r in range(len(model_names))])
        ax.set_xticklabels(model_names)
        ax.legend()

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.7, axis="y")

    def _plot_decision_path(self, ax: Axes, decision_path: List[str]) -> None:
        """
        Plot visualization of the decision path.

        Args:
            ax (Axes): The matplotlib axes to plot on.
            decision_path (List[str]): The decision path to visualize.
        """
        if not decision_path:
            ax.text(0.5, 0.5, "No decision path available", ha="center", va="center")
            return

        # Create a simple directed graph for the path
        G = nx.DiGraph()

        # Add nodes and edges for the path
        for i in range(len(decision_path) - 1):
            G.add_edge(decision_path[i], decision_path[i + 1])

        # Define layout
        pos = nx.spring_layout(G, seed=42)

        # Draw the graph
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=True,
            node_color="#FF9900",
            edge_color="#FF9900",
            node_size=300,
            font_size=8,
            font_weight="bold",
            ax=ax,
        )

        # Set title
        ax.set_title("Decision Path")

        # Remove axis
        ax.axis("off")

    def _plot_impact_summary(self, ax: Axes) -> None:
        """
        Plot summary of model impacts.

        Args:
            ax (Axes): The matplotlib axes to plot on.
        """
        if not self.comparison.comparison_results:
            ax.text(
                0.5, 0.5, "No comparison results available", ha="center", va="center"
            )
            return

        # Get comparison summary
        summary = self.comparison.get_comparison_summary()

        if "key_differences" not in summary:
            ax.text(0.5, 0.5, "No key differences available", ha="center", va="center")
            return

        # Prepare data
        model_names = list(summary["key_differences"].keys())
        impact_data = [
            [
                summary["key_differences"][model].get("relative_impact", 0)
                for model in model_names
            ]
        ]

        # Create horizontal bar chart
        y_pos = np.arange(len(model_names))

        # Plot horizontal bars
        bars = ax.barh(
            y_pos,
            impact_data[0],
            color=[self.color_map.get(name.lower(), "#000000") for name in model_names],
            alpha=0.7,
        )

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}",
                ha="left",
                va="center",
                fontsize=10,
            )

        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names)
        ax.set_xlabel("Relative Impact")
        ax.set_title("Model Impact Summary")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.7, axis="x")

        # Add text summary
        text_y = -0.3
        ax.text(
            0,
            text_y,
            f"Total models compared: {len(model_names)}",
            transform=ax.transAxes,
        )
        ax.text(
            0,
            text_y - 0.1,
            f"Decision path length: {summary.get('decision_path_length', 0)}",
            transform=ax.transAxes,
        )
        ax.text(
            0,
            text_y - 0.2,
            f"Timestamp: {summary.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}",
            transform=ax.transAxes,
        )
