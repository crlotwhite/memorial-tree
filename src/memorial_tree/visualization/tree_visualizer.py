"""
Tree Visualizer module for Memorial Tree.

This module provides functionality for visualizing tree structures in the Memorial Tree
framework, with support for different node types, interactive features, and customization.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Set
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
from matplotlib.widgets import Button, Slider
import numpy as np
from datetime import datetime
import os
import warnings

from ..core.memorial_tree import MemorialTree
from ..core.thought_node import ThoughtNode
from ..core.ghost_node import GhostNode


class TreeVisualizer:
    """
    Class for visualizing Memorial Tree structures.

    This class provides methods for creating visualizations of tree structures,
    with support for different node types, highlighting paths, and interactive features.

    Attributes:
        output_dir (str): Directory for saving visualization outputs.
        node_colors (Dict[str, str]): Color mapping for different node types.
        node_sizes (Dict[str, int]): Size mapping for different node types.
        edge_styles (Dict[str, Dict]): Style mapping for different edge types.
        interactive (bool): Whether to enable interactive features.
    """

    def __init__(self, output_dir: str = "./visualizations", interactive: bool = False):
        """
        Initialize a new TreeVisualizer.

        Args:
            output_dir (str): Directory for saving visualization outputs.
            interactive (bool): Whether to enable interactive features.
        """
        self.output_dir = output_dir
        self.interactive = interactive

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define color scheme for different node types
        self.node_colors = {
            "regular": "#66CCFF",  # Light blue for regular nodes
            "ghost": "#CC99FF",  # Purple for ghost nodes
            "current": "#FF9900",  # Orange for current node
            "highlight": "#FF3333",  # Red for highlighted nodes
            "root": "#00CC66",  # Green for root node
        }

        # Define node sizes for different node types
        self.node_sizes = {
            "regular": 500,
            "ghost": 400,
            "current": 700,
            "root": 800,
        }

        # Define edge styles for different edge types
        self.edge_styles = {
            "regular": {"color": "#CCCCCC", "width": 1.0, "style": "solid"},
            "highlight": {"color": "#FF9900", "width": 2.0, "style": "solid"},
            "ghost": {"color": "#CC99FF", "width": 1.0, "style": "dashed"},
        }

        # Default figure size
        self.figure_size = (12, 8)

    def visualize_tree(
        self,
        tree: MemorialTree,
        highlight_path: Optional[List[str]] = None,
        highlight_nodes: Optional[List[str]] = None,
        show_ghost_nodes: bool = True,
        layout_type: str = "spring",
        node_labels: bool = True,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Visualize a Memorial Tree structure.

        Args:
            tree (MemorialTree): The tree to visualize.
            highlight_path (Optional[List[str]]): List of node IDs to highlight as a path.
            highlight_nodes (Optional[List[str]]): List of individual node IDs to highlight.
            show_ghost_nodes (bool): Whether to show ghost nodes in the visualization.
            layout_type (str): Type of layout to use ('spring', 'kamada_kawai', 'planar', 'circular').
            node_labels (bool): Whether to show node labels.
            save_path (Optional[str]): Path to save the visualization.

        Returns:
            Figure: The matplotlib figure object.
        """
        # Create directed graph
        G = nx.DiGraph()

        # Add nodes and edges
        nodes = tree.get_all_nodes()

        # Track ghost nodes for special styling
        ghost_node_ids = set()

        for node in nodes:
            # Skip ghost nodes if not showing them
            if isinstance(node, GhostNode) and not show_ghost_nodes:
                continue

            # Add node with attributes
            G.add_node(
                node.node_id,
                content=node.content,
                weight=node.weight,
                is_ghost=isinstance(node, GhostNode),
            )

            # Track ghost nodes
            if isinstance(node, GhostNode):
                ghost_node_ids.add(node.node_id)

            # Add edges to children
            for child in node.children:
                # Skip ghost node children if not showing them
                if isinstance(child, GhostNode) and not show_ghost_nodes:
                    continue
                G.add_edge(node.node_id, child.node_id, weight=child.weight)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Define node colors
        node_colors = []
        node_sizes = []

        # Set of nodes in the highlight path (if provided)
        highlight_path_set = set(highlight_path) if highlight_path else set()

        # Set of individually highlighted nodes (if provided)
        highlight_nodes_set = set(highlight_nodes) if highlight_nodes else set()

        for node_id in G.nodes():
            # Determine node color based on type and highlighting
            if node_id == tree.root.node_id:
                node_colors.append(self.node_colors["root"])
                node_sizes.append(self.node_sizes["root"])
            elif node_id == tree.current_node.node_id:
                node_colors.append(self.node_colors["current"])
                node_sizes.append(self.node_sizes["current"])
            elif node_id in highlight_path_set or node_id in highlight_nodes_set:
                node_colors.append(self.node_colors["highlight"])
                node_sizes.append(self.node_sizes["regular"] * 1.2)  # Slightly larger
            elif node_id in ghost_node_ids:
                node_colors.append(self.node_colors["ghost"])
                node_sizes.append(self.node_sizes["ghost"])
            else:
                node_colors.append(self.node_colors["regular"])
                node_sizes.append(self.node_sizes["regular"])

        # Define edge colors and styles
        edge_colors = []
        edge_widths = []
        edge_styles = []

        for u, v in G.edges():
            if highlight_path and u in highlight_path_set and v in highlight_path_set:
                # Check if these nodes are adjacent in the highlight path
                if highlight_path.index(v) == highlight_path.index(u) + 1:
                    edge_colors.append(self.edge_styles["highlight"]["color"])
                    edge_widths.append(self.edge_styles["highlight"]["width"])
                    edge_styles.append(self.edge_styles["highlight"]["style"])
                    continue

            if u in ghost_node_ids or v in ghost_node_ids:
                edge_colors.append(self.edge_styles["ghost"]["color"])
                edge_widths.append(self.edge_styles["ghost"]["width"])
                edge_styles.append(self.edge_styles["ghost"]["style"])
            else:
                edge_colors.append(self.edge_styles["regular"]["color"])
                edge_widths.append(self.edge_styles["regular"]["width"])
                edge_styles.append(self.edge_styles["regular"]["style"])

        # Define layout based on layout_type
        if layout_type == "spring":
            pos = nx.spring_layout(G, seed=42)  # Consistent layout with seed
        elif layout_type == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout_type == "planar":
            try:
                pos = nx.planar_layout(G)
            except nx.NetworkXException:
                warnings.warn("Graph is not planar, falling back to spring layout")
                pos = nx.spring_layout(G, seed=42)
        elif layout_type == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)  # Default to spring layout

        # Draw the graph
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            node_color=node_colors,
            node_size=node_sizes,
            ax=ax,
        )

        # Draw edges with different styles
        for i, (u, v) in enumerate(G.edges()):
            nx.draw_networkx_edges(
                G,
                pos=pos,
                edgelist=[(u, v)],
                width=edge_widths[i],
                edge_color=edge_colors[i],
                style=edge_styles[i],
                ax=ax,
            )

        # Add node labels if requested
        if node_labels:
            # Create custom node labels
            node_labels = {}
            for node_id in G.nodes():
                content = G.nodes[node_id]["content"]
                # Truncate long content
                if len(content) > 20:
                    content = content[:17] + "..."
                node_labels[node_id] = content

            nx.draw_networkx_labels(
                G, pos=pos, labels=node_labels, font_size=8, font_weight="bold", ax=ax
            )

        # Set title
        ax.set_title("Memorial Tree Visualization")

        # Remove axis
        ax.axis("off")

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.node_colors["root"],
                markersize=10,
                label="Root Node",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.node_colors["regular"],
                markersize=10,
                label="Regular Node",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.node_colors["current"],
                markersize=10,
                label="Current Node",
            ),
        ]

        if highlight_path or highlight_nodes:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=self.node_colors["highlight"],
                    markersize=10,
                    label="Highlighted Node",
                )
            )

        if show_ghost_nodes and any(isinstance(node, GhostNode) for node in nodes):
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=self.node_colors["ghost"],
                    markersize=10,
                    label="Ghost Node",
                )
            )

        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.1, 1))

        # Add interactive features if enabled
        if self.interactive:
            self._add_interactive_features(fig, ax, G, pos, tree)

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        elif self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(
                f"{self.output_dir}/tree_visualization_{timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )

        return fig

    def visualize_path(
        self,
        tree: MemorialTree,
        path: List[str],
        show_weights: bool = True,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Visualize a specific path through the tree.

        Args:
            tree (MemorialTree): The tree containing the path.
            path (List[str]): List of node IDs in the path.
            show_weights (bool): Whether to show edge weights.
            save_path (Optional[str]): Path to save the visualization.

        Returns:
            Figure: The matplotlib figure object.
        """
        # Create directed graph for the path
        G = nx.DiGraph()

        # Add nodes and edges for the path
        for i in range(len(path)):
            node = tree.find_node(path[i])
            if not node:
                continue

            # Add node
            G.add_node(
                node.node_id,
                content=node.content,
                weight=node.weight,
                is_ghost=isinstance(node, GhostNode),
            )

            # Add edge to next node in path
            if i < len(path) - 1:
                next_node = tree.find_node(path[i + 1])
                if next_node:
                    G.add_edge(node.node_id, next_node.node_id, weight=next_node.weight)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Define node colors
        node_colors = []
        for node_id in G.nodes():
            if node_id == path[0]:  # First node (usually root)
                node_colors.append(self.node_colors["root"])
            elif node_id == path[-1]:  # Last node (current)
                node_colors.append(self.node_colors["current"])
            elif tree.find_node(node_id) and isinstance(
                tree.find_node(node_id), GhostNode
            ):
                node_colors.append(self.node_colors["ghost"])
            else:
                node_colors.append(self.node_colors["highlight"])

        # Define layout - for paths, a hierarchical layout works well
        pos = nx.spring_layout(G, seed=42)

        # Draw the graph
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            node_color=node_colors,
            node_size=self.node_sizes["regular"],
            ax=ax,
        )

        nx.draw_networkx_edges(
            G,
            pos=pos,
            edge_color=self.edge_styles["highlight"]["color"],
            width=self.edge_styles["highlight"]["width"],
            ax=ax,
        )

        # Add node labels
        node_labels = {}
        for node_id in G.nodes():
            node = tree.find_node(node_id)
            if node:
                content = node.content
                if len(content) > 20:
                    content = content[:17] + "..."
                node_labels[node_id] = content

        nx.draw_networkx_labels(
            G, pos=pos, labels=node_labels, font_size=8, font_weight="bold", ax=ax
        )

        # Add edge weights if requested
        if show_weights:
            edge_labels = {}
            for u, v, data in G.edges(data=True):
                edge_labels[(u, v)] = f"{data['weight']:.2f}"

            nx.draw_networkx_edge_labels(
                G, pos=pos, edge_labels=edge_labels, font_size=8, ax=ax
            )

        # Set title
        ax.set_title("Decision Path Visualization")

        # Remove axis
        ax.axis("off")

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        elif self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(
                f"{self.output_dir}/path_visualization_{timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )

        return fig

    def visualize_ghost_influence(
        self,
        tree: MemorialTree,
        target_node_id: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Visualize the influence of ghost nodes on the tree or a specific node.

        Args:
            tree (MemorialTree): The tree to visualize.
            target_node_id (Optional[str]): ID of a specific node to analyze.
                If None, analyzes the current node.
            save_path (Optional[str]): Path to save the visualization.

        Returns:
            Figure: The matplotlib figure object.
        """
        # Get target node
        target_node = (
            tree.find_node(target_node_id) if target_node_id else tree.current_node
        )

        # Get active ghost nodes
        active_ghost_nodes = tree._get_active_ghost_nodes()

        if not active_ghost_nodes:
            # If no active ghost nodes, use all ghost nodes for visualization
            if tree.ghost_nodes:
                active_ghost_nodes = tree.ghost_nodes
            else:
                # Create a simple figure with a message if no ghost nodes at all
                fig, ax = plt.subplots(figsize=self.figure_size)
                ax.text(
                    0.5,
                    0.5,
                    "No ghost nodes found",
                    ha="center",
                    va="center",
                    fontsize=14,
                )
                ax.axis("off")

                # Save figure if path provided
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches="tight")
                elif self.output_dir:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    plt.savefig(
                        f"{self.output_dir}/ghost_influence_{timestamp}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )

                return fig

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 1. Network visualization showing ghost nodes and their connections
        G = nx.DiGraph()

        # Add target node
        G.add_node(
            target_node.node_id,
            content=target_node.content,
            weight=target_node.weight,
            is_ghost=False,
        )

        # Add ghost nodes and edges to target
        for ghost_node in active_ghost_nodes:
            G.add_node(
                ghost_node.node_id,
                content=ghost_node.content,
                weight=ghost_node.weight,
                is_ghost=True,
                influence=ghost_node.influence,
            )
            G.add_edge(
                ghost_node.node_id,
                target_node.node_id,
                weight=ghost_node.influence,
                influence=ghost_node.influence,
            )

        # Define node colors
        node_colors = []
        node_sizes = []
        for node_id in G.nodes():
            if node_id == target_node.node_id:
                node_colors.append(self.node_colors["current"])
                node_sizes.append(self.node_sizes["current"])
            else:
                node_colors.append(self.node_colors["ghost"])
                node_sizes.append(self.node_sizes["ghost"])

        # Define layout
        pos = nx.spring_layout(G, seed=42)

        # Draw the graph
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            node_color=node_colors,
            node_size=node_sizes,
            ax=ax1,
        )

        # Draw edges with width proportional to influence
        for u, v, data in G.edges(data=True):
            nx.draw_networkx_edges(
                G,
                pos=pos,
                edgelist=[(u, v)],
                width=data["influence"] * 5,  # Scale influence for visibility
                edge_color=self.edge_styles["ghost"]["color"],
                style=self.edge_styles["ghost"]["style"],
                ax=ax1,
                alpha=data["influence"],  # Transparency based on influence
            )

        # Add node labels
        node_labels = {}
        for node_id in G.nodes():
            content = G.nodes[node_id]["content"]
            if len(content) > 20:
                content = content[:17] + "..."
            node_labels[node_id] = content

        nx.draw_networkx_labels(
            G, pos=pos, labels=node_labels, font_size=8, font_weight="bold", ax=ax1
        )

        # Add edge labels showing influence values
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            edge_labels[(u, v)] = f"{data['influence']:.2f}"

        nx.draw_networkx_edge_labels(
            G, pos=pos, edge_labels=edge_labels, font_size=8, ax=ax1
        )

        ax1.set_title("Ghost Node Influence Network")
        ax1.axis("off")

        # 2. Bar chart showing influence values
        ghost_names = [
            ghost.content[:20] + "..." if len(ghost.content) > 20 else ghost.content
            for ghost in active_ghost_nodes
        ]
        influence_values = [ghost.influence for ghost in active_ghost_nodes]
        visibility_values = [ghost.visibility for ghost in active_ghost_nodes]

        x = np.arange(len(ghost_names))
        width = 0.35

        ax2.bar(
            x - width / 2,
            influence_values,
            width,
            label="Influence",
            color=self.node_colors["ghost"],
        )
        ax2.bar(
            x + width / 2,
            visibility_values,
            width,
            label="Visibility",
            color=self.edge_styles["ghost"]["color"],
        )

        ax2.set_xlabel("Ghost Nodes")
        ax2.set_ylabel("Value")
        ax2.set_title("Ghost Node Properties")
        ax2.set_xticks(x)
        ax2.set_xticklabels(ghost_names, rotation=45, ha="right")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        elif self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(
                f"{self.output_dir}/ghost_influence_{timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )

        return fig

    def _add_interactive_features(
        self, fig: Figure, ax: plt.Axes, G: nx.DiGraph, pos: Dict, tree: MemorialTree
    ) -> None:
        """
        Add interactive features to the visualization.

        Args:
            fig (Figure): The matplotlib figure.
            ax (plt.Axes): The matplotlib axes.
            G (nx.DiGraph): The networkx graph.
            pos (Dict): The node positions.
            tree (MemorialTree): The Memorial Tree.
        """
        # Add zoom slider
        ax_zoom = plt.axes([0.2, 0.01, 0.65, 0.03])
        zoom_slider = Slider(ax_zoom, "Zoom", 0.5, 2.0, valinit=1.0)

        def update_zoom(val):
            # Update the axis limits based on zoom level
            x_center = sum(p[0] for p in pos.values()) / len(pos)
            y_center = sum(p[1] for p in pos.values()) / len(pos)

            # Calculate current axis limits
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()

            # Calculate new limits based on zoom
            x_range = (x_max - x_min) / (2 * val)
            y_range = (y_max - y_min) / (2 * val)

            ax.set_xlim(x_center - x_range, x_center + x_range)
            ax.set_ylim(y_center - y_range, y_center + y_range)
            fig.canvas.draw_idle()

        zoom_slider.on_changed(update_zoom)

        # Add button to toggle ghost nodes
        ax_toggle = plt.axes([0.8, 0.05, 0.15, 0.05])
        toggle_button = Button(ax_toggle, "Toggle Ghost Nodes")

        # Store visibility state
        toggle_button.ghost_visible = True

        def toggle_ghost_nodes(event):
            toggle_button.ghost_visible = not toggle_button.ghost_visible

            # Clear the current axes
            ax.clear()

            # Redraw with updated visibility
            self.visualize_tree(
                tree,
                show_ghost_nodes=toggle_button.ghost_visible,
                layout_type="spring",  # Use the same layout type
            )

            fig.canvas.draw_idle()

        toggle_button.on_clicked(toggle_ghost_nodes)

        # Add reset button
        ax_reset = plt.axes([0.65, 0.05, 0.15, 0.05])
        reset_button = Button(ax_reset, "Reset View")

        def reset_view(event):
            zoom_slider.set_val(1.0)
            ax.relim()  # Reset limits
            ax.autoscale_view()  # Autoscale
            fig.canvas.draw_idle()

        reset_button.on_clicked(reset_view)
