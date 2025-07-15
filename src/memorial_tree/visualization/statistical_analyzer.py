"""
Statistical Analyzer module for Memorial Tree.

This module provides functionality for statistical analysis of model comparisons
and generating reports on the differences between mental health models.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from datetime import datetime
import pandas as pd
import os
import json
from scipy import stats

from ..models.model_comparison import ModelComparison


class StatisticalAnalyzer:
    """
    Class for statistical analysis of model comparisons.

    This class provides methods for analyzing the statistical significance of
    differences between mental health models and generating reports.

    Attributes:
        comparison (ModelComparison): The model comparison to analyze.
        output_dir (str): Directory for saving analysis outputs.
        significance_level (float): P-value threshold for statistical significance.
    """

    def __init__(
        self,
        comparison: ModelComparison,
        output_dir: str = "./analysis",
        significance_level: float = 0.05,
    ):
        """
        Initialize a new StatisticalAnalyzer.

        Args:
            comparison (ModelComparison): The model comparison to analyze.
            output_dir (str): Directory for saving analysis outputs.
            significance_level (float): P-value threshold for statistical significance.
        """
        self.comparison = comparison
        self.output_dir = output_dir
        self.significance_level = significance_level

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def analyze_weight_differences(self) -> Dict[str, Any]:
        """
        Analyze the statistical significance of weight differences between models.

        Returns:
            Dict[str, Any]: Analysis results.

        Raises:
            ValueError: If comparison results are not available.
        """
        if not self.comparison.comparison_results:
            raise ValueError("No comparison results available. Run comparison first.")

        results = self.comparison.comparison_results

        if "weight_differences" not in results:
            raise ValueError("No weight differences available in comparison results")

        # Initialize analysis results
        analysis = {
            "timestamp": datetime.now(),
            "model_analysis": {},
            "significant_differences": {},
            "effect_sizes": {},
        }

        # Analyze each model
        for model_name, weight_diffs in results["weight_differences"].items():
            model_analysis = {}
            significant_diffs = {}
            effect_sizes = {}

            # Analyze each step
            for step, diffs in weight_diffs.items():
                # Extract absolute differences
                abs_diffs = [d["absolute_diff"] for d in diffs.values()]

                if not abs_diffs:
                    continue

                # Basic statistics
                step_analysis = {
                    "mean": np.mean(abs_diffs),
                    "median": np.median(abs_diffs),
                    "std": np.std(abs_diffs),
                    "min": np.min(abs_diffs),
                    "max": np.max(abs_diffs),
                    "count": len(abs_diffs),
                }

                # One-sample t-test to check if differences are significant
                t_stat, p_value = stats.ttest_1samp(abs_diffs, 0)
                step_analysis["t_statistic"] = t_stat
                step_analysis["p_value"] = p_value
                step_analysis["significant"] = p_value < self.significance_level

                # Cohen's d effect size
                if step_analysis["std"] > 0:
                    cohen_d = step_analysis["mean"] / step_analysis["std"]
                else:
                    cohen_d = 0
                step_analysis["effect_size"] = cohen_d

                # Store results
                model_analysis[step] = step_analysis

                # Track significant differences
                if step_analysis["significant"]:
                    significant_diffs[step] = {
                        "p_value": p_value,
                        "mean_diff": step_analysis["mean"],
                    }

                # Track effect sizes
                effect_sizes[step] = {
                    "cohen_d": cohen_d,
                    "interpretation": self._interpret_effect_size(cohen_d),
                }

            # Store model results
            analysis["model_analysis"][model_name] = model_analysis
            analysis["significant_differences"][model_name] = significant_diffs
            analysis["effect_sizes"][model_name] = effect_sizes

        return analysis

    def _interpret_effect_size(self, cohen_d: float) -> str:
        """
        Interpret Cohen's d effect size.

        Args:
            cohen_d (float): Cohen's d effect size.

        Returns:
            str: Interpretation of effect size.
        """
        abs_d = abs(cohen_d)

        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def compare_models(self) -> Dict[str, Any]:
        """
        Compare models against each other to find significant differences.

        Returns:
            Dict[str, Any]: Comparison results.

        Raises:
            ValueError: If comparison results are not available.
        """
        if not self.comparison.comparison_results:
            raise ValueError("No comparison results available. Run comparison first.")

        results = self.comparison.comparison_results

        if "model_effects" not in results:
            raise ValueError("No model effects available in comparison results")

        # Initialize comparison results
        comparison = {
            "timestamp": datetime.now(),
            "pairwise_comparisons": {},
            "ranking": {},
        }

        # Get model names
        model_names = list(results["model_effects"].keys())

        if len(model_names) < 2:
            return comparison  # Not enough models to compare

        # Perform pairwise comparisons
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i + 1 :]:
                pair_key = f"{model1}_vs_{model2}"
                comparison["pairwise_comparisons"][pair_key] = self._compare_two_models(
                    model1, model2
                )

        # Rank models based on overall impact
        ranking = {}
        for model_name in model_names:
            if model_name in results["statistical_metrics"]:
                metrics = results["statistical_metrics"][model_name]
                ranking[model_name] = metrics.get("overall_mean_abs_diff", 0)

        # Sort by impact (descending)
        sorted_ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)

        comparison["ranking"] = {
            "by_impact": [
                {"model": model, "impact": impact} for model, impact in sorted_ranking
            ]
        }

        return comparison

    def _compare_two_models(self, model1: str, model2: str) -> Dict[str, Any]:
        """
        Compare two models to find significant differences.

        Args:
            model1 (str): Name of first model.
            model2 (str): Name of second model.

        Returns:
            Dict[str, Any]: Comparison results.
        """
        results = self.comparison.comparison_results

        # Initialize comparison results
        comparison = {
            "steps": {},
            "overall": {
                "significant": False,
                "p_value": 1.0,
                "effect_size": 0.0,
                "stronger_model": None,
            },
        }

        # Get model effects
        model1_effects = results["model_effects"].get(model1, {})
        model2_effects = results["model_effects"].get(model2, {})

        # Find common steps
        common_steps = set(model1_effects.keys()) & set(model2_effects.keys())

        # Collect all differences for overall comparison
        all_diffs = []

        # Compare each step
        for step in common_steps:
            step_comparison = {}

            # Get weights for this step
            weights1 = model1_effects[step]
            weights2 = model2_effects[step]

            # Find common nodes
            common_nodes = set(weights1.keys()) & set(weights2.keys())

            if not common_nodes:
                continue

            # Calculate differences
            diffs = [weights1[node] - weights2[node] for node in common_nodes]
            all_diffs.extend(diffs)

            # Perform t-test
            t_stat, p_value = stats.ttest_1samp(diffs, 0)

            # Calculate effect size
            if len(diffs) > 1 and np.std(diffs) > 0:
                cohen_d = np.mean(diffs) / np.std(diffs)
            else:
                cohen_d = 0

            # Determine which model has stronger effect
            stronger = model1 if np.mean(diffs) > 0 else model2

            # Store results
            step_comparison = {
                "mean_diff": np.mean(diffs),
                "std_diff": np.std(diffs) if len(diffs) > 1 else 0,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < self.significance_level,
                "effect_size": cohen_d,
                "effect_interpretation": self._interpret_effect_size(cohen_d),
                "stronger_model": (
                    stronger if p_value < self.significance_level else None
                ),
            }

            comparison["steps"][step] = step_comparison

        # Overall comparison
        if all_diffs:
            t_stat, p_value = stats.ttest_1samp(all_diffs, 0)

            if len(all_diffs) > 1 and np.std(all_diffs) > 0:
                cohen_d = np.mean(all_diffs) / np.std(all_diffs)
            else:
                cohen_d = 0

            stronger = model1 if np.mean(all_diffs) > 0 else model2

            comparison["overall"] = {
                "mean_diff": np.mean(all_diffs),
                "std_diff": np.std(all_diffs) if len(all_diffs) > 1 else 0,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < self.significance_level,
                "effect_size": cohen_d,
                "effect_interpretation": self._interpret_effect_size(cohen_d),
                "stronger_model": (
                    stronger if p_value < self.significance_level else None
                ),
            }

        return comparison

    def generate_report(self, format: str = "json") -> str:
        """
        Generate a comprehensive statistical report.

        Args:
            format (str): Output format ("json", "csv", or "text").

        Returns:
            str: Path to the generated report file.

        Raises:
            ValueError: If comparison results are not available or format is invalid.
        """
        if not self.comparison.comparison_results:
            raise ValueError("No comparison results available. Run comparison first.")

        if format not in ["json", "csv", "text"]:
            raise ValueError(
                f"Invalid format: {format}. Must be 'json', 'csv', or 'text'"
            )

        # Get analysis results
        weight_analysis = self.analyze_weight_differences()
        model_comparison = self.compare_models()

        # Combine results
        report_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models_analyzed": list(self.comparison.models.keys()),
            "weight_analysis": weight_analysis,
            "model_comparison": model_comparison,
            "summary": self._generate_summary(weight_analysis, model_comparison),
        }

        # Generate report based on format
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "json":
            report_path = f"{self.output_dir}/report_{timestamp_str}.json"
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2, default=str)

        elif format == "csv":
            report_path = f"{self.output_dir}/report_{timestamp_str}.csv"

            # Convert to DataFrame for CSV export
            summary_df = pd.DataFrame(report_data["summary"]["model_summary"])
            summary_df.to_csv(report_path, index=False)

            # Export additional CSVs
            for model_name in report_data["models_analyzed"]:
                if model_name in weight_analysis["model_analysis"]:
                    model_df = pd.DataFrame.from_dict(
                        weight_analysis["model_analysis"][model_name], orient="index"
                    )
                    model_df.to_csv(
                        f"{self.output_dir}/model_{model_name}_{timestamp_str}.csv"
                    )

        elif format == "text":
            report_path = f"{self.output_dir}/report_{timestamp_str}.txt"

            with open(report_path, "w") as f:
                f.write("=" * 80 + "\n")
                f.write("MEMORIAL TREE MODEL COMPARISON REPORT\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Generated: {report_data['timestamp']}\n")
                f.write(
                    f"Models Analyzed: {', '.join(report_data['models_analyzed'])}\n\n"
                )

                f.write("-" * 80 + "\n")
                f.write("SUMMARY\n")
                f.write("-" * 80 + "\n\n")

                # Write summary
                summary = report_data["summary"]
                f.write(f"Most Impactful Model: {summary['most_impactful_model']}\n")
                f.write(f"Least Impactful Model: {summary['least_impactful_model']}\n")
                f.write(
                    f"Significant Differences Found: {summary['significant_differences_found']}\n\n"
                )

                # Write model summary table
                f.write("Model Summary:\n")
                f.write("-" * 60 + "\n")
                f.write(
                    f"{'Model':<15} {'Mean Impact':<12} {'Effect Size':<12} {'Significance':<12}\n"
                )
                f.write("-" * 60 + "\n")

                for model in summary["model_summary"]:
                    f.write(
                        f"{model['model']:<15} "
                        f"{model['mean_impact']:<12.4f} "
                        f"{model['effect_size']:<12.4f} "
                        f"{'Yes' if model['significant'] else 'No':<12}\n"
                    )

                f.write("\n")

                # Write pairwise comparisons
                f.write("-" * 80 + "\n")
                f.write("PAIRWISE COMPARISONS\n")
                f.write("-" * 80 + "\n\n")

                for pair, comparison in model_comparison[
                    "pairwise_comparisons"
                ].items():
                    f.write(f"Comparison: {pair}\n")
                    f.write(
                        f"  Significant: {'Yes' if comparison['overall']['significant'] else 'No'}\n"
                    )
                    f.write(f"  P-value: {comparison['overall']['p_value']:.4f}\n")
                    f.write(
                        f"  Effect Size: {comparison['overall']['effect_size']:.4f} "
                        f"({comparison['overall']['effect_interpretation']})\n"
                    )

                    if comparison["overall"]["significant"]:
                        f.write(
                            f"  Stronger Model: {comparison['overall']['stronger_model']}\n"
                        )

                    f.write("\n")

        return report_path

    def _generate_summary(
        self, weight_analysis: Dict[str, Any], model_comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a summary of the analysis results.

        Args:
            weight_analysis (Dict[str, Any]): Weight difference analysis results.
            model_comparison (Dict[str, Any]): Model comparison results.

        Returns:
            Dict[str, Any]: Summary of results.
        """
        summary = {
            "model_summary": [],
            "significant_differences_found": False,
            "most_impactful_model": None,
            "least_impactful_model": None,
        }

        # Check if there are any significant differences
        for model_name, sig_diffs in weight_analysis["significant_differences"].items():
            if sig_diffs:
                summary["significant_differences_found"] = True
                break

        # Get model rankings
        if "ranking" in model_comparison and "by_impact" in model_comparison["ranking"]:
            rankings = model_comparison["ranking"]["by_impact"]

            if rankings:
                summary["most_impactful_model"] = rankings[0]["model"]
                summary["least_impactful_model"] = rankings[-1]["model"]

        # Generate model summary
        for model_name in self.comparison.models:
            model_summary = {
                "model": model_name,
                "mean_impact": 0.0,
                "effect_size": 0.0,
                "significant": False,
            }

            # Get mean impact from statistical metrics
            if (
                "statistical_metrics" in self.comparison.comparison_results
                and model_name
                in self.comparison.comparison_results["statistical_metrics"]
            ):
                metrics = self.comparison.comparison_results["statistical_metrics"][
                    model_name
                ]
                model_summary["mean_impact"] = metrics.get("overall_mean_abs_diff", 0)

            # Get effect size and significance
            if model_name in weight_analysis["model_analysis"]:
                # Average effect size across steps
                effect_sizes = []
                significant_steps = 0

                for step, analysis in weight_analysis["model_analysis"][
                    model_name
                ].items():
                    effect_sizes.append(analysis.get("effect_size", 0))
                    if analysis.get("significant", False):
                        significant_steps += 1

                model_summary["effect_size"] = (
                    np.mean(effect_sizes) if effect_sizes else 0
                )
                model_summary["significant"] = significant_steps > 0

            summary["model_summary"].append(model_summary)

        return summary
