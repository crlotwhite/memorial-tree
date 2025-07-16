"""
Tests for the tree visualization functionality.
"""

import pytest
import os
import tempfile
import matplotlib

# Use non-interactive backend for testing
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.memorial_tree.core.memorial_tree import MemorialTree
from src.memorial_tree.core.ghost_node import GhostNode
from src.memorial_tree.visualization.tree_visualizer import TreeVisualizer


class TestTreeVisualizer:
    """Test suite for the TreeVisualizer class."""

    @pytest.fixture
    def sample_tree(self):
        """Create a sample tree for testing."""
        tree = MemorialTree("Root Decision")

        # Add first level choices
        choice1 = tree.add_thought(tree.root.node_id, "Option A", weight=1.0)
        choice2 = tree.add_thought(tree.root.node_id, "Option B", weight=1.0)
        choice3 = tree.add_thought(tree.root.node_id, "Option C", weight=1.0)

        # Add second level choices to Option A
        tree.add_thought(choice1.node_id, "Option A.1", weight=1.0)
        tree.add_thought(choice1.node_id, "Option A.2", weight=1.0)

        # Add second level choices to Option B
        tree.add_thought(choice2.node_id, "Option B.1", weight=1.0)
        tree.add_thought(choice2.node_id, "Option B.2", weight=1.0)

        # Add second level choices to Option C
        tree.add_thought(choice3.node_id, "Option C.1", weight=1.0)
        tree.add_thought(choice3.node_id, "Option C.2", weight=1.0)

        # Add ghost nodes
        ghost1 = tree.add_ghost_node("Hidden Fear", influence=0.4, visibility=0.2)
        ghost2 = tree.add_ghost_node(
            "Subconscious Desire", influence=0.6, visibility=0.3
        )

        return tree

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_initialization(self, temp_output_dir):
        """Test initialization of TreeVisualizer."""
        visualizer = TreeVisualizer(temp_output_dir)

        assert visualizer.output_dir == temp_output_dir
        assert isinstance(visualizer.node_colors, dict)
        assert isinstance(visualizer.node_sizes, dict)
        assert isinstance(visualizer.edge_styles, dict)
        assert visualizer.interactive is False

        # Test with interactive mode
        interactive_visualizer = TreeVisualizer(temp_output_dir, interactive=True)
        assert interactive_visualizer.interactive is True

    def test_visualize_tree(self, sample_tree, temp_output_dir):
        """Test visualizing a tree."""
        visualizer = TreeVisualizer(temp_output_dir)

        # Test basic visualization
        fig = visualizer.visualize_tree(sample_tree)
        assert isinstance(fig, Figure)

        # Test with highlight path
        highlight_path = [
            sample_tree.root.node_id,
            sample_tree.root.children[0].node_id,
        ]
        fig = visualizer.visualize_tree(sample_tree, highlight_path=highlight_path)
        assert isinstance(fig, Figure)

        # Test with highlight nodes
        highlight_nodes = [node.node_id for node in sample_tree.root.children]
        fig = visualizer.visualize_tree(sample_tree, highlight_nodes=highlight_nodes)
        assert isinstance(fig, Figure)

        # Test with ghost nodes hidden
        fig = visualizer.visualize_tree(sample_tree, show_ghost_nodes=False)
        assert isinstance(fig, Figure)

        # Test with different layout types
        for layout_type in ["spring", "kamada_kawai", "circular"]:
            fig = visualizer.visualize_tree(sample_tree, layout_type=layout_type)
            assert isinstance(fig, Figure)

        # Test with node labels disabled
        fig = visualizer.visualize_tree(sample_tree, node_labels=False)
        assert isinstance(fig, Figure)

        # Test with save path
        save_path = os.path.join(temp_output_dir, "test_tree.png")
        fig = visualizer.visualize_tree(sample_tree, save_path=save_path)
        assert os.path.exists(save_path)

        # Close figures to avoid warnings
        plt.close("all")

    def test_visualize_path(self, sample_tree, temp_output_dir):
        """Test visualizing a path through the tree."""
        visualizer = TreeVisualizer(temp_output_dir)

        # Create a path
        path = [
            sample_tree.root.node_id,
            sample_tree.root.children[0].node_id,
            sample_tree.root.children[0].children[0].node_id,
        ]

        # Test basic path visualization
        fig = visualizer.visualize_path(sample_tree, path)
        assert isinstance(fig, Figure)

        # Test with weights hidden
        fig = visualizer.visualize_path(sample_tree, path, show_weights=False)
        assert isinstance(fig, Figure)

        # Test with save path
        save_path = os.path.join(temp_output_dir, "test_path.png")
        fig = visualizer.visualize_path(sample_tree, path, save_path=save_path)
        assert os.path.exists(save_path)

        # Close figures to avoid warnings
        plt.close("all")

    def test_visualize_ghost_influence(self, sample_tree, temp_output_dir):
        """Test visualizing ghost node influence."""
        visualizer = TreeVisualizer(temp_output_dir)

        # Test ghost influence visualization
        fig = visualizer.visualize_ghost_influence(sample_tree)
        assert isinstance(fig, Figure)

        # Test with specific target node
        target_node_id = sample_tree.root.children[0].node_id
        fig = visualizer.visualize_ghost_influence(sample_tree, target_node_id)
        assert isinstance(fig, Figure)

        # Test with save path
        save_path = os.path.join(temp_output_dir, "test_ghost.png")
        fig = visualizer.visualize_ghost_influence(sample_tree, save_path=save_path)
        assert os.path.exists(save_path)

        # Close figures to avoid warnings
        plt.close("all")

    def test_tree_with_no_ghost_nodes(self, temp_output_dir):
        """Test visualizing a tree with no ghost nodes."""
        # Create a tree without ghost nodes
        tree = MemorialTree("Root Decision")
        choice1 = tree.add_thought(tree.root.node_id, "Option A", weight=1.0)
        tree.add_thought(choice1.node_id, "Option A.1", weight=1.0)

        visualizer = TreeVisualizer(temp_output_dir)

        # Test ghost influence visualization with no ghost nodes
        fig = visualizer.visualize_ghost_influence(tree)
        assert isinstance(fig, Figure)

        # Close figures to avoid warnings
        plt.close("all")
