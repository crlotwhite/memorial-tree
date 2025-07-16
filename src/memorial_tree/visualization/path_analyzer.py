"""
Path Analyzer module for Memorial Tree.

This module provides functionality for visualizing and analyzing decision paths
in the Memorial Tree framework, including path highlighting, ghost node influence
visualization, and path evolution animation.
"""

from typing import Dict, List, Optional, Tuple, Union, Set
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import numpy as np
from datetime import datetime
import os
import warnings
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from ..core.memorial_tree import MemorialTree
from ..core.ghost_node import GhostNode


class PathAnalyzer:
    """
    Class for analyzing and visualizing decision paths in Memorial Tree structures.

    This class provides methods for highlighting decision paths, visualizing ghost node
    influences on paths, and creating animations of path evolution over time.

    Attributes:
        output_dir (str): Directory for saving visualization outputs.
        node_colors (Dict[str, str]): Color mapping for different node types.
        edge_styles (Dict[str, Dict]): Style mapping for different edge types.
        animation_interval (int): Interval between animation frames in milliseconds.
    """

    def __init__(self, output_dir: str = "./visualizations"):
        """
        Initialize a new PathAnalyzer.

        Args:
            output_dir (str): Directory for saving visualization outputs.
        """
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define color scheme for different node types
        self.node_colors = {
            "regular": "#66CCFF",  # Light blue for regular nodes
            "ghost": "#CC99FF",  # Purple for ghost nodes
            "current": "#FF9900",  # Orange for current node
            "highlight": "#FF3333",  # Red for highlighted nodes
            "root": "#00CC66",  # Green for root node
            "influenced": "#FFCC00",  # Yellow for nodes influenced by ghost nodes
        }

        # Define edge styles for different edge types
        self.edge_styles = {
            "regular": {"color": "#CCCCCC", "width": 1.0, "style": "solid"},
            "highlight": {"color": "#FF9900", "width": 2.0, "style": "solid"},
            "ghost": {"color": "#CC99FF", "width": 1.0, "style": "dashed"},
            "influence": {"color": "#FFCC00", "width": 1.5, "style": "dotted"},
        }

        # Animation settings
        self.animation_interval = 500  # milliseconds

        # Default figure size
        self.figure_size = (12, 8)

    def highlight_decision_path(
        self,
        tree: MemorialTree,
        path: List[str],
        show_ghost_influence: bool = True,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Visualize a decision path with highlighting and optional ghost node influence.

        Args:
            tree (MemorialTree): The tree containing the path.
            path (List[str]): List of node IDs in the path.
            show_ghost_influence (bool): Whether to show ghost node influence on the path.
            save_path (Optional[str]): Path to save the visualization.

        Returns:
            Figure: The matplotlib figure object.
        """
        # Create directed graph
        G = nx.DiGraph()

        # Add nodes and edges for the path
        path_nodes = []
        for node_id in path:
            node = tree.find_node(node_id)
            if not node:
                continue
            path_nodes.append(node)
            G.add_node(
                node.node_id,
                content=node.content,
                weight=node.weight,
                is_ghost=isinstance(node, GhostNode),
            )

        # Add edges between path nodes
        for i in range(len(path_nodes) - 1):
            G.add_edge(
                path_nodes[i].node_id,
                path_nodes[i + 1].node_id,
                weight=path_nodes[i + 1].weight,
            )

        # Add ghost nodes and their influence edges if requested
        ghost_influence_edges = []
        if show_ghost_influence:
            active_ghost_nodes = tree._get_active_ghost_nodes()
            for ghost_node in active_ghost_nodes:
                G.add_node(
                    ghost_node.node_id,
                    content=ghost_node.content,
                    weight=ghost_node.weight,
                    is_ghost=True,
                    influence=ghost_node.influence,
                )

                # Add influence edges to path nodes
                # In a real implementation, we would determine which nodes are influenced
                # Here we'll add influence to all nodes in the path for demonstration
                for path_node in path_nodes:
                    influence_strength = (
                        ghost_node.influence * 0.5
                    )  # Scale for visibility
                    G.add_edge(
                        ghost_node.node_id,
                        path_node.node_id,
                        weight=influence_strength,
                        influence=influence_strength,
                        is_influence=True,
                    )
                    ghost_influence_edges.append(
                        (ghost_node.node_id, path_node.node_id)
                    )

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Define node colors
        node_colors = []
        node_sizes = []
        for node_id in G.nodes():
            if node_id == path[0]:  # First node (usually root)
                node_colors.append(self.node_colors["root"])
                node_sizes.append(800)
            elif node_id == path[-1]:  # Last node (current)
                node_colors.append(self.node_colors["current"])
                node_sizes.append(700)
            elif G.nodes[node_id].get("is_ghost", False):
                node_colors.append(self.node_colors["ghost"])
                node_sizes.append(400)
            else:
                node_colors.append(self.node_colors["highlight"])
                node_sizes.append(500)

        # Define layout - hierarchical layout works well for paths
        pos = nx.spring_layout(G, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            node_color=node_colors,
            node_size=node_sizes,
            ax=ax,
        )

        # Draw path edges
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(
            G,
            pos=pos,
            edgelist=path_edges,
            width=self.edge_styles["highlight"]["width"],
            edge_color=self.edge_styles["highlight"]["color"],
            style=self.edge_styles["highlight"]["style"],
            ax=ax,
        )

        # Draw ghost influence edges if requested
        if show_ghost_influence and ghost_influence_edges:
            # Draw with varying width based on influence strength
            for u, v in ghost_influence_edges:
                influence = G.edges[u, v].get("influence", 0.5)
                nx.draw_networkx_edges(
                    G,
                    pos=pos,
                    edgelist=[(u, v)],
                    width=influence * 3,  # Scale width by influence
                    edge_color=self.edge_styles["influence"]["color"],
                    style=self.edge_styles["influence"]["style"],
                    alpha=influence,  # Transparency based on influence
                    ax=ax,
                )

        # Add node labels
        node_labels = {}
        for node_id in G.nodes():
            content = G.nodes[node_id]["content"]
            if len(content) > 20:
                content = content[:17] + "..."
            node_labels[node_id] = content

        nx.draw_networkx_labels(
            G, pos=pos, labels=node_labels, font_size=8, font_weight="bold", ax=ax
        )

        # Set title
        ax.set_title("Decision Path Analysis")

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
                label="Start Node",
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
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.node_colors["highlight"],
                markersize=10,
                label="Path Node",
            ),
            Line2D(
                [0],
                [0],
                color=self.edge_styles["highlight"]["color"],
                lw=self.edge_styles["highlight"]["width"],
                label="Decision Path",
            ),
        ]

        if show_ghost_influence:
            legend_elements.extend(
                [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=self.node_colors["ghost"],
                        markersize=10,
                        label="Ghost Node",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=self.edge_styles["influence"]["color"],
                        lw=self.edge_styles["influence"]["width"],
                        linestyle=self.edge_styles["influence"]["style"],
                        label="Ghost Influence",
                    ),
                ]
            )

        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.1, 1))

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        elif self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(
                f"{self.output_dir}/path_analysis_{timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )

        return fig

    def visualize_ghost_influence_on_path(
        self,
        tree: MemorialTree,
        path: List[str],
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Visualize the influence of ghost nodes on a specific decision path.

        Args:
            tree (MemorialTree): The tree containing the path.
            path (List[str]): List of node IDs in the path.
            save_path (Optional[str]): Path to save the visualization.

        Returns:
            Figure: The matplotlib figure object.
        """
        # Get active ghost nodes
        active_ghost_nodes = tree._get_active_ghost_nodes()

        # If no active ghost nodes, use all ghost nodes for visualization
        if not active_ghost_nodes:
            active_ghost_nodes = tree.ghost_nodes

        if not active_ghost_nodes:
            # Create a simple figure with a message if no ghost nodes
            fig, ax = plt.subplots(figsize=self.figure_size)
            ax.text(
                0.5,
                0.5,
                "No ghost nodes found to analyze influence",
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
                    f"{self.output_dir}/ghost_path_influence_{timestamp}.png",
                    dpi=300,
                    bbox_inches="tight",
                )

            return fig

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 1. Network visualization showing path and ghost influences
        G = nx.DiGraph()

        # Add path nodes
        path_nodes = []
        for node_id in path:
            node = tree.find_node(node_id)
            if not node:
                continue
            path_nodes.append(node)
            G.add_node(
                node.node_id,
                content=node.content,
                weight=node.weight,
                is_ghost=False,
                step=path.index(node_id),  # Track position in path
            )

        # Add path edges
        for i in range(len(path_nodes) - 1):
            G.add_edge(
                path_nodes[i].node_id,
                path_nodes[i + 1].node_id,
                weight=path_nodes[i + 1].weight,
                is_path=True,
            )

        # Add ghost nodes and influence edges
        for ghost_node in active_ghost_nodes:
            G.add_node(
                ghost_node.node_id,
                content=ghost_node.content,
                weight=ghost_node.weight,
                is_ghost=True,
                influence=ghost_node.influence,
            )

            # Calculate influence on each path node
            # In a real implementation, this would be based on more complex logic
            for path_node in path_nodes:
                # Simple influence calculation for demonstration
                influence_strength = ghost_node.influence * (
                    1 - 0.1 * G.nodes[path_node.node_id]["step"]
                )
                if influence_strength > 0.1:  # Only show significant influences
                    G.add_edge(
                        ghost_node.node_id,
                        path_node.node_id,
                        weight=influence_strength,
                        influence=influence_strength,
                    )

        # Define node colors and sizes
        node_colors = []
        node_sizes = []
        for node_id in G.nodes():
            if G.nodes[node_id].get("is_ghost", False):
                node_colors.append(self.node_colors["ghost"])
                node_sizes.append(400)
            elif node_id == path[0]:
                node_colors.append(self.node_colors["root"])
                node_sizes.append(800)
            elif node_id == path[-1]:
                node_colors.append(self.node_colors["current"])
                node_sizes.append(700)
            else:
                node_colors.append(self.node_colors["highlight"])
                node_sizes.append(500)

        # Define layout
        pos = nx.spring_layout(G, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            node_color=node_colors,
            node_size=node_sizes,
            ax=ax1,
        )

        # Draw path edges
        path_edges = [
            (path[i], path[i + 1])
            for i in range(len(path) - 1)
            if path[i] in G.nodes and path[i + 1] in G.nodes
        ]
        nx.draw_networkx_edges(
            G,
            pos=pos,
            edgelist=path_edges,
            width=self.edge_styles["highlight"]["width"],
            edge_color=self.edge_styles["highlight"]["color"],
            style=self.edge_styles["highlight"]["style"],
            ax=ax1,
        )

        # Draw influence edges with varying width and transparency
        for u, v, data in G.edges(data=True):
            if not data.get("is_path", False):
                influence = data.get("influence", 0.5)
                nx.draw_networkx_edges(
                    G,
                    pos=pos,
                    edgelist=[(u, v)],
                    width=influence * 3,
                    edge_color=self.edge_styles["influence"]["color"],
                    style=self.edge_styles["influence"]["style"],
                    alpha=influence,
                    ax=ax1,
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

        ax1.set_title("Ghost Node Influence on Decision Path")
        ax1.axis("off")

        # 2. Heatmap showing influence strength along the path
        # Create a matrix of ghost influences on each path node
        ghost_names = [
            ghost.content[:15] + "..." if len(ghost.content) > 15 else ghost.content
            for ghost in active_ghost_nodes
        ]
        path_labels = [
            node.content[:15] + "..." if len(node.content) > 15 else node.content
            for node in path_nodes
        ]

        influence_matrix = np.zeros((len(active_ghost_nodes), len(path_nodes)))

        # Fill the matrix with influence values
        for i, ghost in enumerate(active_ghost_nodes):
            for j, path_node in enumerate(path_nodes):
                # Check if there's an edge between ghost and path node
                if G.has_edge(ghost.node_id, path_node.node_id):
                    influence_matrix[i, j] = G.edges[ghost.node_id, path_node.node_id][
                        "influence"
                    ]

        # Create heatmap
        im = ax2.imshow(influence_matrix, cmap="YlOrRd")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label("Influence Strength")

        # Add labels
        ax2.set_xticks(np.arange(len(path_labels)))
        ax2.set_yticks(np.arange(len(ghost_names)))
        ax2.set_xticklabels(path_labels, rotation=45, ha="right")
        ax2.set_yticklabels(ghost_names)

        # Add grid lines
        ax2.set_xticks(np.arange(-0.5, len(path_labels), 1), minor=True)
        ax2.set_yticks(np.arange(-0.5, len(ghost_names), 1), minor=True)
        ax2.grid(which="minor", color="w", linestyle="-", linewidth=1)

        ax2.set_title("Ghost Node Influence Heatmap")
        ax2.set_xlabel("Path Nodes")
        ax2.set_ylabel("Ghost Nodes")

        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        elif self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(
                f"{self.output_dir}/ghost_path_influence_{timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )

        return fig

    def animate_path_evolution(
        self,
        tree: MemorialTree,
        path_history: List[List[str]],
        show_ghost_influence: bool = True,
        interval: int = 1000,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Create an animation showing the evolution of a decision path over time.

        Args:
            tree (MemorialTree): The tree containing the paths.
            path_history (List[List[str]]): List of paths at different time points.
            show_ghost_influence (bool): Whether to show ghost node influence.
            interval (int): Time between frames in milliseconds.
            save_path (Optional[str]): Path to save the animation.

        Returns:
            Figure: The matplotlib figure object.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Create a directed graph for the full tree structure
        G = nx.DiGraph()

        # Add all nodes that appear in any path
        all_nodes = set()
        for path in path_history:
            all_nodes.update(path)

        for node_id in all_nodes:
            node = tree.find_node(node_id)
            if node:
                G.add_node(
                    node.node_id,
                    content=node.content,
                    weight=node.weight,
                    is_ghost=isinstance(node, GhostNode),
                )

        # Add all edges between nodes in paths
        for path in path_history:
            for i in range(len(path) - 1):
                if path[i] in G.nodes and path[i + 1] in G.nodes:
                    G.add_edge(path[i], path[i + 1], weight=1.0)

        # Add ghost nodes if requested
        ghost_nodes = []
        if show_ghost_influence:
            for ghost_node in tree.ghost_nodes:
                G.add_node(
                    ghost_node.node_id,
                    content=ghost_node.content,
                    weight=ghost_node.weight,
                    is_ghost=True,
                    influence=ghost_node.influence,
                )
                ghost_nodes.append(ghost_node.node_id)

        # Define layout - use the same layout for all frames
        pos = nx.spring_layout(G, seed=42)

        # Define base node colors and sizes
        base_node_colors = {}
        base_node_sizes = {}

        for node_id in G.nodes():
            if G.nodes[node_id].get("is_ghost", False):
                base_node_colors[node_id] = self.node_colors["ghost"]
                base_node_sizes[node_id] = 400
            elif node_id == tree.root.node_id:
                base_node_colors[node_id] = self.node_colors["root"]
                base_node_sizes[node_id] = 800
            else:
                base_node_colors[node_id] = self.node_colors["regular"]
                base_node_sizes[node_id] = 500

        # Animation update function
        def update(frame):
            ax.clear()

            current_path = path_history[frame]

            # Update node colors and sizes for this frame
            node_colors = base_node_colors.copy()
            node_sizes = base_node_sizes.copy()

            # Highlight current path
            for i, node_id in enumerate(current_path):
                if i == len(current_path) - 1:  # Current node
                    node_colors[node_id] = self.node_colors["current"]
                    node_sizes[node_id] = 700
                else:
                    node_colors[node_id] = self.node_colors["highlight"]
                    node_sizes[node_id] = 600

            # Draw nodes
            nx.draw_networkx_nodes(
                G,
                pos=pos,
                node_color=[
                    node_colors.get(n, self.node_colors["regular"]) for n in G.nodes()
                ],
                node_size=[node_sizes.get(n, 500) for n in G.nodes()],
                ax=ax,
            )

            # Draw regular edges
            nx.draw_networkx_edges(
                G,
                pos=pos,
                edge_color=self.edge_styles["regular"]["color"],
                width=self.edge_styles["regular"]["width"],
                style=self.edge_styles["regular"]["style"],
                ax=ax,
            )

            # Draw highlighted path edges
            path_edges = [
                (current_path[i], current_path[i + 1])
                for i in range(len(current_path) - 1)
            ]
            if path_edges:
                nx.draw_networkx_edges(
                    G,
                    pos=pos,
                    edgelist=path_edges,
                    edge_color=self.edge_styles["highlight"]["color"],
                    width=self.edge_styles["highlight"]["width"],
                    style=self.edge_styles["highlight"]["style"],
                    ax=ax,
                )

            # Draw ghost influence edges if requested
            if show_ghost_influence:
                for ghost_id in ghost_nodes:
                    # In a real implementation, we would calculate actual influence
                    # Here we'll just show influence to the current node
                    if current_path:
                        current_node_id = current_path[-1]
                        ghost_node = tree.find_node(ghost_id)
                        if ghost_node and isinstance(ghost_node, GhostNode):
                            influence = ghost_node.influence
                            nx.draw_networkx_edges(
                                G,
                                pos=pos,
                                edgelist=[(ghost_id, current_node_id)],
                                edge_color=self.edge_styles["influence"]["color"],
                                width=influence * 3,
                                style=self.edge_styles["influence"]["style"],
                                alpha=influence,
                                ax=ax,
                            )

            # Add node labels
            node_labels = {}
            for node_id in G.nodes():
                content = G.nodes[node_id]["content"]
                if len(content) > 20:
                    content = content[:17] + "..."
                node_labels[node_id] = content

            nx.draw_networkx_labels(
                G, pos=pos, labels=node_labels, font_size=8, font_weight="bold", ax=ax
            )

            # Set title with frame number
            ax.set_title(
                f"Decision Path Evolution - Step {frame+1}/{len(path_history)}"
            )

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
                    markerfacecolor=self.node_colors["current"],
                    markersize=10,
                    label="Current Node",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=self.node_colors["highlight"],
                    markersize=10,
                    label="Path Node",
                ),
                Line2D(
                    [0],
                    [0],
                    color=self.edge_styles["highlight"]["color"],
                    lw=self.edge_styles["highlight"]["width"],
                    label="Decision Path",
                ),
            ]

            if show_ghost_influence:
                legend_elements.extend(
                    [
                        plt.Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=self.node_colors["ghost"],
                            markersize=10,
                            label="Ghost Node",
                        ),
                        Line2D(
                            [0],
                            [0],
                            color=self.edge_styles["influence"]["color"],
                            lw=self.edge_styles["influence"]["width"],
                            linestyle=self.edge_styles["influence"]["style"],
                            label="Ghost Influence",
                        ),
                    ]
                )

            ax.legend(
                handles=legend_elements, loc="upper right", bbox_to_anchor=(1.1, 1)
            )

        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=len(path_history),
            interval=interval,
            repeat=True,
            repeat_delay=1000,
        )

        # Save animation if path provided
        if save_path:
            if save_path.endswith(".gif"):
                anim.save(save_path, writer="pillow", fps=1000 // interval)
            else:
                anim.save(save_path, writer="ffmpeg", fps=1000 // interval)
        elif self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            anim.save(
                f"{self.output_dir}/path_evolution_{timestamp}.gif",
                writer="pillow",
                fps=1000 // interval,
            )

        return fig
