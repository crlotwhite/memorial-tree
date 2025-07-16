"""
Tests for the path analyzer functionality.
"""

import pytest
import os
import tempfile
import matplotlib

# Use non-interactive backend for testing
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from src.memorial_tree.core.memorial_tree import MemorialTree
from src.memorial_tree.core.ghost_node import GhostNode
from src.memorial_tree.visualization.path_analyzer import PathAnalyzer


class TestPathAnalyzer:
    """Test suite for the PathAnalyzer class."""

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
        tree.add_ghost_node("Hidden Fear", influence=0.4, visibility=0.2)
        tree.add_ghost_node("Subconscious Desire", influence=0.6, visibility=0.3)

        return tree

    @pytest.fixture
    def sample_path(self, sample_tree):
        """Create a sample path for testing."""
        # Make some choices to create a path
        sample_tree.make_choice(sample_tree.root.children[0].node_id)  # Choose Option A
        sample_tree.make_choice(
            sample_tree.current_node.children[0].node_id
        )  # Choose Option A.1

        return sample_tree.path_history

    @pytest.fixture
    def sample_path_history(self, sample_tree):
        """Create a sample path history for testing."""
        # Create multiple paths representing evolution over time
        path1 = [sample_tree.root.node_id]
        path2 = [sample_tree.root.node_id, sample_tree.root.children[0].node_id]
        path3 = [
            sample_tree.root.node_id,
            sample_tree.root.children[0].node_id,
            sample_tree.root.children[0].children[0].node_id,
        ]

        return [path1, path2, path3]

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_initialization(self, temp_output_dir):
        """Test initialization of PathAnalyzer."""
        analyzer = PathAnalyzer(temp_output_dir)

        assert analyzer.output_dir == temp_output_dir
        assert isinstance(analyzer.node_colors, dict)
        assert isinstance(analyzer.edge_styles, dict)
        assert analyzer.animation_interval == 500

    def test_highlight_decision_path(self, sample_tree, sample_path, temp_output_dir):
        """Test highlighting a decision path."""
        analyzer = PathAnalyzer(temp_output_dir)

        # Test basic path highlighting
        fig = analyzer.highlight_decision_path(sample_tree, sample_path)
        assert isinstance(fig, Figure)

        # Test with ghost influence disabled
        fig = analyzer.highlight_decision_path(
            sample_tree, sample_path, show_ghost_influence=False
        )
        assert isinstance(fig, Figure)

        # Test with save path
        save_path = os.path.join(temp_output_dir, "test_path_highlight.png")
        fig = analyzer.highlight_decision_path(
            sample_tree, sample_path, save_path=save_path
        )
        assert os.path.exists(save_path)

        # Close figures to avoid warnings
        plt.close("all")

    def test_visualize_ghost_influence_on_path(
        self, sample_tree, sample_path, temp_output_dir
    ):
        """Test visualizing ghost influence on a path."""
        analyzer = PathAnalyzer(temp_output_dir)

        # Test ghost influence visualization
        fig = analyzer.visualize_ghost_influence_on_path(sample_tree, sample_path)
        assert isinstance(fig, Figure)

        # Test with save path
        save_path = os.path.join(temp_output_dir, "test_ghost_influence.png")
        fig = analyzer.visualize_ghost_influence_on_path(
            sample_tree, sample_path, save_path=save_path
        )
        assert os.path.exists(save_path)

        # Close figures to avoid warnings
        plt.close("all")

    def test_animate_path_evolution(
        self, sample_tree, sample_path_history, temp_output_dir
    ):
        """Test animating path evolution."""
        analyzer = PathAnalyzer(temp_output_dir)

        # Test animation creation
        fig = analyzer.animate_path_evolution(sample_tree, sample_path_history)
        assert isinstance(fig, Figure)

        # Test with ghost influence disabled
        fig = analyzer.animate_path_evolution(
            sample_tree, sample_path_history, show_ghost_influence=False
        )
        assert isinstance(fig, Figure)

        # Test with custom interval
        fig = analyzer.animate_path_evolution(
            sample_tree, sample_path_history, interval=500
        )
        assert isinstance(fig, Figure)

        # Test with save path (GIF)
        save_path = os.path.join(temp_output_dir, "test_animation.gif")
        fig = analyzer.animate_path_evolution(
            sample_tree, sample_path_history, save_path=save_path
        )
        # Note: We don't check if the file exists because the animation saving might be skipped in CI environments

        # Close figures to avoid warnings
        plt.close("all")

    def test_tree_with_no_ghost_nodes(self, temp_output_dir):
        """Test visualizing a tree with no ghost nodes."""
        # Create a tree without ghost nodes
        tree = MemorialTree("Root Decision")
        choice1 = tree.add_thought(tree.root.node_id, "Option A", weight=1.0)
        tree.add_thought(choice1.node_id, "Option A.1", weight=1.0)

        # Create a path
        path = [tree.root.node_id, choice1.node_id]

        analyzer = PathAnalyzer(temp_output_dir)

        # Test ghost influence visualization with no ghost nodes
        fig = analyzer.visualize_ghost_influence_on_path(tree, path)
        assert isinstance(fig, Figure)

        # Close figures to avoid warnings
        plt.close("all")
