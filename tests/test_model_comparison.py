"""
Tests for the model comparison functionality.
"""

import pytest
import numpy as np
from datetime import datetime

from src.memorial_tree.core.memorial_tree import MemorialTree
from src.memorial_tree.models.adhd_model import ADHDModel
from src.memorial_tree.models.depression_model import DepressionModel
from src.memorial_tree.models.anxiety_model import AnxietyModel
from src.memorial_tree.models.model_comparison import ModelComparison


class TestModelComparison:
    """Test suite for the ModelComparison class."""

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
    def comparison_with_models(self, sample_tree):
        """Create a model comparison with models added."""
        comparison = ModelComparison(sample_tree)

        # Add models
        comparison.add_model(
            "adhd",
            ADHDModel(
                attention_span=0.3,
                impulsivity=0.8,
                distraction_rate=0.6,
                hyperactivity=0.7,
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

        return comparison

    def test_initialization(self, sample_tree):
        """Test initialization of ModelComparison."""
        comparison = ModelComparison(sample_tree)

        assert comparison.control_tree == sample_tree
        assert comparison.models == {}
        assert comparison.model_trees == {}
        assert isinstance(comparison.comparison_results, dict)
        assert isinstance(comparison.timestamp, datetime)

    def test_add_model(self, sample_tree):
        """Test adding models to comparison."""
        comparison = ModelComparison(sample_tree)

        # Add ADHD model
        adhd_model = ADHDModel()
        comparison.add_model("adhd", adhd_model)

        assert "adhd" in comparison.models
        assert comparison.models["adhd"] == adhd_model
        assert "adhd" in comparison.model_trees
        assert comparison.model_trees["adhd"] is not sample_tree  # Should be a clone

        # Add Depression model
        depression_model = DepressionModel()
        comparison.add_model("depression", depression_model)

        assert "depression" in comparison.models
        assert comparison.models["depression"] == depression_model

        # Test adding duplicate model name
        with pytest.raises(ValueError):
            comparison.add_model("adhd", ADHDModel())

        # Test adding unsupported model type
        with pytest.raises(ValueError):
            comparison.add_model("invalid", object())

    def test_clone_tree(self, sample_tree):
        """Test cloning a tree."""
        comparison = ModelComparison(sample_tree)
        cloned_tree = comparison._clone_tree(sample_tree)

        # Check that it's a different object
        assert cloned_tree is not sample_tree

        # Check that structure is the same
        assert cloned_tree.get_tree_size() == sample_tree.get_tree_size()
        assert cloned_tree.get_tree_depth() == sample_tree.get_tree_depth()

        # Check root content is the same
        assert cloned_tree.root.content == sample_tree.root.content

    def test_run_comparison(self, comparison_with_models, sample_tree):
        """Test running a comparison."""
        # Define a decision path
        decision_path = [sample_tree.root.node_id, sample_tree.root.children[0].node_id]

        # Run comparison
        results = comparison_with_models.run_comparison(decision_path)

        # Check results structure
        assert "timestamp" in results
        assert "decision_path" in results
        assert "model_effects" in results
        assert "weight_differences" in results
        assert "statistical_metrics" in results

        # Check model effects
        assert "adhd" in results["model_effects"]
        assert "depression" in results["model_effects"]
        assert "anxiety" in results["model_effects"]

        # Check weight differences
        assert "adhd" in results["weight_differences"]
        assert "depression" in results["weight_differences"]
        assert "anxiety" in results["weight_differences"]

        # Check statistical metrics
        assert "adhd" in results["statistical_metrics"]
        assert "depression" in results["statistical_metrics"]
        assert "anxiety" in results["statistical_metrics"]

    def test_get_model_characteristics(self, comparison_with_models):
        """Test getting model characteristics."""
        characteristics = comparison_with_models.get_model_characteristics()

        # Check structure
        assert "adhd" in characteristics
        assert "depression" in characteristics
        assert "anxiety" in characteristics

        # Check ADHD characteristics
        adhd_chars = characteristics["adhd"]
        assert "attention_span" in adhd_chars
        assert "impulsivity" in adhd_chars
        assert "distraction_rate" in adhd_chars
        assert "hyperactivity" in adhd_chars

        # Check Depression characteristics
        depression_chars = characteristics["depression"]
        assert "negative_bias" in depression_chars
        assert "decision_delay" in depression_chars
        assert "energy_level" in depression_chars
        assert "rumination" in depression_chars

        # Check Anxiety characteristics
        anxiety_chars = characteristics["anxiety"]
        assert "worry_amplification" in anxiety_chars
        assert "risk_aversion" in anxiety_chars
        assert "rumination_cycles" in anxiety_chars
        assert "uncertainty_intolerance" in anxiety_chars

    def test_get_comparison_summary(self, comparison_with_models, sample_tree):
        """Test getting comparison summary."""
        # Run comparison first
        decision_path = [sample_tree.root.node_id, sample_tree.root.children[0].node_id]
        comparison_with_models.run_comparison(decision_path)

        # Get summary
        summary = comparison_with_models.get_comparison_summary()

        # Check structure
        assert "timestamp" in summary
        assert "models_compared" in summary
        assert "decision_path_length" in summary
        assert "model_characteristics" in summary
        assert "key_differences" in summary

        # Check models compared
        assert "adhd" in summary["models_compared"]
        assert "depression" in summary["models_compared"]
        assert "anxiety" in summary["models_compared"]

        # Check decision path length
        assert summary["decision_path_length"] == len(decision_path)

        # Check model characteristics
        assert "adhd" in summary["model_characteristics"]
        assert "depression" in summary["model_characteristics"]
        assert "anxiety" in summary["model_characteristics"]

        # Check key differences
        assert "adhd" in summary["key_differences"]
        assert "depression" in summary["key_differences"]
        assert "anxiety" in summary["key_differences"]

    def test_no_comparison_summary(self):
        """Test getting summary without running comparison."""
        comparison = ModelComparison()
        summary = comparison.get_comparison_summary()

        assert "error" in summary
