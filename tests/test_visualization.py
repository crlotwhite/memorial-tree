"""
Tests for the visualization functionality.
"""

import pytest
import os
import tempfile
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.memorial_tree.core.memorial_tree import MemorialTree
from src.memorial_tree.models.adhd_model import ADHDModel
from src.memorial_tree.models.depression_model import DepressionModel
from src.memorial_tree.models.anxiety_model import AnxietyModel
from src.memorial_tree.models.model_comparison import ModelComparison
from src.memorial_tree.visualization.model_visualizer import ModelVisualizer
from src.memorial_tree.visualization.statistical_analyzer import StatisticalAnalyzer


class TestModelVisualizer:
    """Test suite for the ModelVisualizer class."""

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

        return tree

    @pytest.fixture
    def comparison_with_results(self, sample_tree):
        """Create a model comparison with results."""
        comparison = ModelComparison(sample_tree)

        # Add models
        comparison.add_model("adhd", ADHDModel())
        comparison.add_model("depression", DepressionModel())
        comparison.add_model("anxiety", AnxietyModel())

        # Run comparison
        decision_path = [sample_tree.root.node_id, sample_tree.root.children[0].node_id]
        comparison.run_comparison(decision_path)

        return comparison

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_initialization(self, comparison_with_results, temp_output_dir):
        """Test initialization of ModelVisualizer."""
        visualizer = ModelVisualizer(comparison_with_results, temp_output_dir)

        assert visualizer.comparison == comparison_with_results
        assert visualizer.output_dir == temp_output_dir
        assert isinstance(visualizer.color_map, dict)
        assert "control" in visualizer.color_map
        assert "adhd" in visualizer.color_map
        assert "depression" in visualizer.color_map
        assert "anxiety" in visualizer.color_map

    def test_visualize_weight_differences(
        self, comparison_with_results, temp_output_dir
    ):
        """Test visualizing weight differences."""
        visualizer = ModelVisualizer(comparison_with_results, temp_output_dir)

        # Test with specific step
        fig = visualizer.visualize_weight_differences("step_0")
        assert isinstance(fig, Figure)

        # Test with no step (overall)
        fig = visualizer.visualize_weight_differences()
        assert isinstance(fig, Figure)

        # Test with save path
        save_path = os.path.join(temp_output_dir, "test_weights.png")
        fig = visualizer.visualize_weight_differences(save_path=save_path)
        assert os.path.exists(save_path)

        # Close figures to avoid warnings
        plt.close("all")

    def test_visualize_decision_tree(
        self, comparison_with_results, sample_tree, temp_output_dir
    ):
        """Test visualizing decision tree."""
        visualizer = ModelVisualizer(comparison_with_results, temp_output_dir)

        # Test without highlight path
        fig = visualizer.visualize_decision_tree(sample_tree)
        assert isinstance(fig, Figure)

        # Test with highlight path
        highlight_path = [
            sample_tree.root.node_id,
            sample_tree.root.children[0].node_id,
        ]
        fig = visualizer.visualize_decision_tree(sample_tree, highlight_path)
        assert isinstance(fig, Figure)

        # Test with save path
        save_path = os.path.join(temp_output_dir, "test_tree.png")
        fig = visualizer.visualize_decision_tree(sample_tree, save_path=save_path)
        assert os.path.exists(save_path)

        # Close figures to avoid warnings
        plt.close("all")

    def test_visualize_model_characteristics(
        self, comparison_with_results, temp_output_dir
    ):
        """Test visualizing model characteristics."""
        visualizer = ModelVisualizer(comparison_with_results, temp_output_dir)

        # Test visualization
        fig = visualizer.visualize_model_characteristics()
        assert isinstance(fig, Figure)

        # Test with save path
        save_path = os.path.join(temp_output_dir, "test_chars.png")
        fig = visualizer.visualize_model_characteristics(save_path=save_path)
        assert os.path.exists(save_path)

        # Close figures to avoid warnings
        plt.close("all")

    def test_visualize_statistical_comparison(
        self, comparison_with_results, temp_output_dir
    ):
        """Test visualizing statistical comparison."""
        visualizer = ModelVisualizer(comparison_with_results, temp_output_dir)

        # Test with default metric
        fig = visualizer.visualize_statistical_comparison()
        assert isinstance(fig, Figure)

        # Test with specific metric
        fig = visualizer.visualize_statistical_comparison("overall_max_abs_diff")
        assert isinstance(fig, Figure)

        # Test with save path
        save_path = os.path.join(temp_output_dir, "test_stats.png")
        fig = visualizer.visualize_statistical_comparison(save_path=save_path)
        assert os.path.exists(save_path)

        # Close figures to avoid warnings
        plt.close("all")

    def test_create_comparison_dashboard(
        self, comparison_with_results, sample_tree, temp_output_dir
    ):
        """Test creating comparison dashboard."""
        visualizer = ModelVisualizer(comparison_with_results, temp_output_dir)

        # Define decision path
        decision_path = [sample_tree.root.node_id, sample_tree.root.children[0].node_id]

        # Test dashboard creation
        fig = visualizer.create_comparison_dashboard(decision_path)
        assert isinstance(fig, Figure)

        # Test with save path
        save_path = os.path.join(temp_output_dir, "test_dashboard.png")
        fig = visualizer.create_comparison_dashboard(decision_path, save_path=save_path)
        assert os.path.exists(save_path)

        # Close figures to avoid warnings
        plt.close("all")


class TestStatisticalAnalyzer:
    """Test suite for the StatisticalAnalyzer class."""

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

        return tree

    @pytest.fixture
    def comparison_with_results(self, sample_tree):
        """Create a model comparison with results."""
        comparison = ModelComparison(sample_tree)

        # Add models
        comparison.add_model("adhd", ADHDModel())
        comparison.add_model("depression", DepressionModel())
        comparison.add_model("anxiety", AnxietyModel())

        # Run comparison
        decision_path = [sample_tree.root.node_id, sample_tree.root.children[0].node_id]
        comparison.run_comparison(decision_path)

        return comparison

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_initialization(self, comparison_with_results, temp_output_dir):
        """Test initialization of StatisticalAnalyzer."""
        analyzer = StatisticalAnalyzer(comparison_with_results, temp_output_dir)

        assert analyzer.comparison == comparison_with_results
        assert analyzer.output_dir == temp_output_dir
        assert analyzer.significance_level == 0.05

    def test_analyze_weight_differences(self, comparison_with_results):
        """Test analyzing weight differences."""
        analyzer = StatisticalAnalyzer(comparison_with_results)

        # Run analysis
        analysis = analyzer.analyze_weight_differences()

        # Check structure
        assert "timestamp" in analysis
        assert "model_analysis" in analysis
        assert "significant_differences" in analysis
        assert "effect_sizes" in analysis

        # Check model analysis
        assert "adhd" in analysis["model_analysis"]
        assert "depression" in analysis["model_analysis"]
        assert "anxiety" in analysis["model_analysis"]

    def test_compare_models(self, comparison_with_results):
        """Test comparing models."""
        analyzer = StatisticalAnalyzer(comparison_with_results)

        # Run comparison
        comparison = analyzer.compare_models()

        # Check structure
        assert "timestamp" in comparison
        assert "pairwise_comparisons" in comparison
        assert "ranking" in comparison

        # Check pairwise comparisons
        assert "adhd_vs_depression" in comparison["pairwise_comparisons"]
        assert "adhd_vs_anxiety" in comparison["pairwise_comparisons"]
        assert "depression_vs_anxiety" in comparison["pairwise_comparisons"]

        # Check ranking
        assert "by_impact" in comparison["ranking"]

    def test_generate_report(self, comparison_with_results, temp_output_dir):
        """Test generating reports in different formats."""
        analyzer = StatisticalAnalyzer(comparison_with_results, temp_output_dir)

        # Test JSON format
        json_path = analyzer.generate_report("json")
        assert os.path.exists(json_path)
        assert json_path.endswith(".json")

        # Test CSV format
        csv_path = analyzer.generate_report("csv")
        assert os.path.exists(csv_path)
        assert csv_path.endswith(".csv")

        # Test text format
        text_path = analyzer.generate_report("text")
        assert os.path.exists(text_path)
        assert text_path.endswith(".txt")

        # Test invalid format
        with pytest.raises(ValueError):
            analyzer.generate_report("invalid")
