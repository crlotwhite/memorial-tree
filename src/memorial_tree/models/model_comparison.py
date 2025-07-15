"""
Model Comparison module for Memorial Tree.

This module provides functionality for comparing different mental health models
and analyzing their effects on decision-making processes.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from datetime import datetime
import copy

from ..core.memorial_tree import MemorialTree
from ..core.thought_node import ThoughtNode
from .adhd_model import ADHDModel
from .depression_model import DepressionModel
from .anxiety_model import AnxietyModel


class ModelComparison:
    """
    Class for comparing different mental health models and their effects on decision-making.

    This class provides methods for running simulations with different models,
    comparing their decision patterns, and analyzing the differences.

    Attributes:
        control_tree (MemorialTree): The control tree without any model modifications.
        models (Dict[str, Any]): Dictionary of mental health models to compare.
        model_trees (Dict[str, MemorialTree]): Dictionary of trees modified by each model.
        comparison_results (Dict[str, Any]): Results of the comparison analysis.
    """

    def __init__(self, base_tree: Optional[MemorialTree] = None):
        """
        Initialize a new ModelComparison instance.

        Args:
            base_tree (Optional[MemorialTree]): Base tree to use for comparison.
                If None, a new tree will be created.
        """
        # Initialize base tree (control)
        self.control_tree = base_tree if base_tree else MemorialTree("Root")

        # Initialize models dictionary
        self.models: Dict[str, Any] = {}

        # Initialize model trees dictionary
        self.model_trees: Dict[str, MemorialTree] = {}

        # Initialize comparison results
        self.comparison_results: Dict[str, Any] = {}

        # Initialize timestamp
        self.timestamp = datetime.now()

    def add_model(self, model_name: str, model: Any) -> None:
        """
        Add a mental health model to the comparison.

        Args:
            model_name (str): Name identifier for the model.
            model (Any): The model instance (ADHDModel, DepressionModel, etc.).

        Raises:
            ValueError: If model_name already exists or model type is not supported.
        """
        # Check if model name already exists
        if model_name in self.models:
            raise ValueError(f"Model name '{model_name}' already exists")

        # Check if model type is supported
        if not isinstance(model, (ADHDModel, DepressionModel, AnxietyModel)):
            raise ValueError(f"Unsupported model type: {type(model).__name__}")

        # Add model to dictionary
        self.models[model_name] = model

        # Create a copy of the control tree for this model
        self.model_trees[model_name] = self._clone_tree(self.control_tree)

    def run_comparison(self, decision_path: List[str]) -> Dict[str, Any]:
        """
        Run a comparison of models by simulating a decision path.

        This method applies each model to its corresponding tree and simulates
        the same decision path, then analyzes the differences in outcomes.

        Args:
            decision_path (List[str]): List of node IDs representing the decision path to simulate.

        Returns:
            Dict[str, Any]: Results of the comparison.

        Raises:
            ValueError: If no models have been added or decision path is invalid.
        """
        if not self.models:
            raise ValueError("No models added for comparison")

        if not decision_path:
            raise ValueError("Decision path cannot be empty")

        # Reset all trees to root
        self._reset_all_trees()

        # Initialize results dictionary
        results = {
            "timestamp": datetime.now(),
            "decision_path": decision_path,
            "model_effects": {},
            "weight_differences": {},
            "path_deviations": {},
            "statistical_metrics": {},
        }

        # Process control tree first
        control_weights = self._process_decision_path(self.control_tree, decision_path)
        results["control_weights"] = control_weights

        # Process each model tree
        for model_name, model in self.models.items():
            tree = self.model_trees[model_name]

            # Apply model to tree
            self._apply_model_to_tree(model, tree)

            # Process decision path
            model_weights = self._process_decision_path(tree, decision_path)

            # Calculate weight differences from control
            weight_diffs = self._calculate_weight_differences(
                control_weights, model_weights
            )

            # Store results
            results["model_effects"][model_name] = model_weights
            results["weight_differences"][model_name] = weight_diffs

        # Calculate statistical metrics
        results["statistical_metrics"] = self._calculate_statistical_metrics(results)

        # Store results
        self.comparison_results = results

        return results

    def _clone_tree(self, tree: MemorialTree) -> MemorialTree:
        """
        Create a deep copy of a tree for independent model application.

        Args:
            tree (MemorialTree): The tree to clone.

        Returns:
            MemorialTree: A new tree with the same structure.
        """
        # Create new tree with same root content
        new_tree = MemorialTree(tree.root.content, tree.backend_manager.backend_type)

        # Clone the tree structure (recursive helper function)
        def clone_node_children(original_parent_id: str, new_parent_id: str) -> None:
            original_parent = tree.find_node(original_parent_id)
            if not original_parent:
                return

            for child in original_parent.children:
                # Create new child in new tree
                new_child = new_tree.add_thought(
                    new_parent_id, child.content, child.weight
                )

                # Recursively clone children
                clone_node_children(child.node_id, new_child.node_id)

        # Start cloning from root
        clone_node_children(tree.root.node_id, new_tree.root.node_id)

        # Clone ghost nodes
        for ghost_node in tree.ghost_nodes:
            new_tree.add_ghost_node(
                ghost_node.content,
                ghost_node.influence,
                ghost_node.visibility,
                ghost_node.weight,
            )

        return new_tree

    def _reset_all_trees(self) -> None:
        """
        Reset all trees to their root nodes.
        """
        self.control_tree.reset_to_root()
        for tree in self.model_trees.values():
            tree.reset_to_root()

    def _apply_model_to_tree(self, model: Any, tree: MemorialTree) -> None:
        """
        Apply a mental health model to a tree.

        Args:
            model (Any): The model to apply.
            tree (MemorialTree): The tree to modify.
        """
        # Apply model to current node
        model.modify_decision_process(tree, tree.current_node)

    def _process_decision_path(
        self, tree: MemorialTree, decision_path: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Process a decision path on a tree and collect weights at each step.

        Args:
            tree (MemorialTree): The tree to process.
            decision_path (List[str]): List of node IDs representing the path.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary of weights at each step.
        """
        weights_by_step = {}

        # Start from root
        tree.reset_to_root()

        # Process each step in the path
        for i, node_id in enumerate(decision_path):
            # Get current node ID
            current_id = tree.current_node.node_id

            # Get available choices
            choices = tree.get_available_choices()

            # Collect weights
            step_weights = {node.node_id: node.weight for node in choices}
            weights_by_step[f"step_{i}"] = step_weights

            # Make choice if not the last step
            if i < len(decision_path) - 1:
                try:
                    tree.make_choice(node_id)
                except (ValueError, Exception) as e:
                    # If choice is invalid, break and return partial results
                    weights_by_step["error"] = f"Error at step {i}: {str(e)}"
                    break

        return weights_by_step

    def _calculate_weight_differences(
        self,
        control_weights: Dict[str, Dict[str, float]],
        model_weights: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate differences between control weights and model weights.

        Args:
            control_weights (Dict[str, Dict[str, float]]): Control weights by step.
            model_weights (Dict[str, Dict[str, float]]): Model weights by step.

        Returns:
            Dict[str, Dict[str, float]]: Weight differences by step.
        """
        differences = {}

        # Process each step
        for step in control_weights:
            if step == "error" or step not in model_weights:
                continue

            step_diffs = {}

            # Calculate differences for each node
            for node_id, control_weight in control_weights[step].items():
                if node_id in model_weights[step]:
                    model_weight = model_weights[step][node_id]
                    # Calculate absolute and relative differences
                    abs_diff = model_weight - control_weight
                    rel_diff = abs_diff / control_weight if control_weight != 0 else 0

                    step_diffs[node_id] = {
                        "absolute_diff": abs_diff,
                        "relative_diff": rel_diff,
                        "control_weight": control_weight,
                        "model_weight": model_weight,
                    }

            differences[step] = step_diffs

        return differences

    def _calculate_statistical_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate statistical metrics for the comparison results.

        Args:
            results (Dict[str, Any]): Comparison results.

        Returns:
            Dict[str, Any]: Statistical metrics.
        """
        metrics = {}

        # Process each model
        for model_name in self.models:
            model_metrics = {}

            # Skip if model has no weight differences
            if model_name not in results["weight_differences"]:
                continue

            weight_diffs = results["weight_differences"][model_name]

            # Calculate metrics across all steps
            all_abs_diffs = []
            all_rel_diffs = []

            for step, diffs in weight_diffs.items():
                step_abs_diffs = [d["absolute_diff"] for d in diffs.values()]
                step_rel_diffs = [d["relative_diff"] for d in diffs.values()]

                all_abs_diffs.extend(step_abs_diffs)
                all_rel_diffs.extend(step_rel_diffs)

                # Calculate step-specific metrics
                model_metrics[f"{step}_mean_abs_diff"] = (
                    np.mean(step_abs_diffs) if step_abs_diffs else 0
                )
                model_metrics[f"{step}_std_abs_diff"] = (
                    np.std(step_abs_diffs) if step_abs_diffs else 0
                )
                model_metrics[f"{step}_max_abs_diff"] = (
                    max(step_abs_diffs) if step_abs_diffs else 0
                )

                model_metrics[f"{step}_mean_rel_diff"] = (
                    np.mean(step_rel_diffs) if step_rel_diffs else 0
                )
                model_metrics[f"{step}_std_rel_diff"] = (
                    np.std(step_rel_diffs) if step_rel_diffs else 0
                )
                model_metrics[f"{step}_max_rel_diff"] = (
                    max(step_rel_diffs) if step_rel_diffs else 0
                )

            # Calculate overall metrics
            model_metrics["overall_mean_abs_diff"] = (
                np.mean(all_abs_diffs) if all_abs_diffs else 0
            )
            model_metrics["overall_std_abs_diff"] = (
                np.std(all_abs_diffs) if all_abs_diffs else 0
            )
            model_metrics["overall_max_abs_diff"] = (
                max(all_abs_diffs) if all_abs_diffs else 0
            )

            model_metrics["overall_mean_rel_diff"] = (
                np.mean(all_rel_diffs) if all_rel_diffs else 0
            )
            model_metrics["overall_std_rel_diff"] = (
                np.std(all_rel_diffs) if all_rel_diffs else 0
            )
            model_metrics["overall_max_rel_diff"] = (
                max(all_rel_diffs) if all_rel_diffs else 0
            )

            metrics[model_name] = model_metrics

        return metrics

    def get_model_characteristics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the characteristic parameters of each model.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of model characteristics.
        """
        characteristics = {}

        for model_name, model in self.models.items():
            model_chars = {}

            if isinstance(model, ADHDModel):
                model_chars = {
                    "attention_span": model.attention_span,
                    "impulsivity": model.impulsivity,
                    "distraction_rate": model.distraction_rate,
                    "hyperactivity": model.hyperactivity,
                }
            elif isinstance(model, DepressionModel):
                model_chars = {
                    "negative_bias": model.negative_bias,
                    "decision_delay": model.decision_delay,
                    "energy_level": model.energy_level,
                    "rumination": model.rumination,
                }
            elif isinstance(model, AnxietyModel):
                model_chars = {
                    "worry_amplification": model.worry_amplification,
                    "risk_aversion": model.risk_aversion,
                    "rumination_cycles": model.rumination_cycles,
                    "uncertainty_intolerance": model.uncertainty_intolerance,
                }

            characteristics[model_name] = model_chars

        return characteristics

    def get_comparison_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the comparison results.

        Returns:
            Dict[str, Any]: Summary of comparison results.
        """
        if not self.comparison_results:
            return {"error": "No comparison has been run yet"}

        summary = {
            "timestamp": self.comparison_results.get("timestamp", datetime.now()),
            "models_compared": list(self.models.keys()),
            "decision_path_length": len(
                self.comparison_results.get("decision_path", [])
            ),
            "model_characteristics": self.get_model_characteristics(),
            "key_differences": {},
        }

        # Extract key differences for each model
        for model_name in self.models:
            if model_name not in self.comparison_results.get("statistical_metrics", {}):
                continue

            metrics = self.comparison_results["statistical_metrics"][model_name]

            summary["key_differences"][model_name] = {
                "mean_weight_difference": metrics.get("overall_mean_abs_diff", 0),
                "max_weight_difference": metrics.get("overall_max_abs_diff", 0),
                "relative_impact": metrics.get("overall_mean_rel_diff", 0),
            }

        return summary
