#!/usr/bin/env python
"""
Command-line script for running Memorial Tree benchmarks.

This script provides a command-line interface for running performance benchmarks
on the Memorial Tree system and generating reports.
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from .performance_benchmark import PerformanceBenchmark, MemoryOptimizer


def main():
    """Run the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Run Memorial Tree performance benchmarks"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Directory for saving benchmark results and plots",
    )

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="Tree sizes to benchmark (e.g., --sizes 10 50 100 500)",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        choices=[
            "all",
            "tree_creation",
            "tree_traversal",
            "backend_operations",
            "backend_switching",
            "memory_usage",
            "model_application",
            "visualization",
        ],
        default="all",
        help="Specific benchmark to run (default: all)",
    )

    parser.add_argument(
        "--report", action="store_true", help="Generate a detailed HTML report"
    )

    args = parser.parse_args()

    # Create benchmark instance
    benchmark = PerformanceBenchmark(args.output_dir)

    # Run specified benchmark(s)
    start_time = time.time()

    if args.benchmark == "all":
        results = benchmark.run_all_benchmarks(args.sizes)
    elif args.benchmark == "tree_creation":
        results = benchmark.benchmark_tree_creation(args.sizes)
    elif args.benchmark == "tree_traversal":
        results = benchmark.benchmark_tree_traversal(args.sizes)
    elif args.benchmark == "backend_operations":
        results = benchmark.benchmark_backend_operations(args.sizes)
    elif args.benchmark == "backend_switching":
        results = benchmark.benchmark_backend_switching(args.sizes)
    elif args.benchmark == "memory_usage":
        results = benchmark.benchmark_memory_usage(args.sizes)
    elif args.benchmark == "model_application":
        results = benchmark.benchmark_model_application(args.sizes)
    elif args.benchmark == "visualization":
        results = benchmark.benchmark_visualization(args.sizes)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nBenchmarks completed in {total_time:.2f} seconds")
    print(f"Results saved to {args.output_dir}")

    # Generate HTML report if requested
    if args.report:
        generate_html_report(benchmark.results, args.output_dir, total_time)
        print(
            f"HTML report generated at {os.path.join(args.output_dir, 'report.html')}"
        )


def generate_html_report(
    results: Dict[str, Any], output_dir: str, total_time: float
) -> None:
    """
    Generate an HTML report from benchmark results.

    Args:
        results (Dict[str, Any]): Benchmark results.
        output_dir (str): Directory to save the report.
        total_time (float): Total benchmark execution time.
    """
    # Get current date and time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Start HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Memorial Tree Performance Benchmark Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #fff;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
            }}
            .image-container {{
                margin: 20px 0;
                text-align: center;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .summary {{
                font-weight: bold;
                margin-top: 10px;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                font-size: 0.9em;
                color: #777;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Memorial Tree Performance Benchmark Report</h1>
                <p>Generated on: {now}</p>
                <p>Total benchmark time: {total_time:.2f} seconds</p>
            </div>
    """

    # Add sections for each benchmark type
    if "tree_creation" in results:
        html_content += _generate_tree_creation_section(
            results["tree_creation"], output_dir
        )

    if "tree_traversal" in results:
        html_content += _generate_tree_traversal_section(
            results["tree_traversal"], output_dir
        )

    if "backend_operations" in results:
        html_content += _generate_backend_operations_section(
            results["backend_operations"], output_dir
        )

    if "backend_switching" in results:
        html_content += _generate_backend_switching_section(
            results["backend_switching"], output_dir
        )

    if "memory_usage" in results:
        html_content += _generate_memory_usage_section(
            results["memory_usage"], output_dir
        )

    if "model_application" in results:
        html_content += _generate_model_application_section(
            results["model_application"], output_dir
        )

    if "visualization" in results:
        html_content += _generate_visualization_section(
            results["visualization"], output_dir
        )

    # Add summary section
    html_content += """
            <div class="section">
                <h2>Summary and Recommendations</h2>
                <p>Based on the benchmark results, here are some recommendations for optimizing Memorial Tree performance:</p>
                <ul>
                    <li>For large trees (>500 nodes), consider using the NumPy backend for best performance.</li>
                    <li>Minimize backend switching operations during performance-critical code paths.</li>
                    <li>For visualization of large trees, consider using simplified layouts or partial tree views.</li>
                    <li>When working with memory-constrained environments, use the MemoryOptimizer to reduce memory usage.</li>
                    <li>For real-time applications, keep tree sizes below 200 nodes for optimal responsiveness.</li>
                </ul>
            </div>

            <div class="footer">
                <p>Memorial Tree Performance Benchmark Report</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Write HTML to file
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(html_content)


def _generate_tree_creation_section(results: Dict[str, Any], output_dir: str) -> str:
    """Generate HTML section for tree creation benchmark."""
    sizes = results["sizes"]
    creation_times = results["creation_times"]
    nodes_per_second = results["nodes_per_second"]
    scaling_factor = results["scaling_factor"]

    html = """
            <div class="section">
                <h2>Tree Creation Performance</h2>
                <p>This benchmark measures how quickly trees of different sizes can be created.</p>

                <table>
                    <tr>
                        <th>Tree Size (nodes)</th>
                        <th>Creation Time (seconds)</th>
                        <th>Performance (nodes/sec)</th>
                    </tr>
    """

    for i, size in enumerate(sizes):
        html += f"""
                    <tr>
                        <td>{size}</td>
                        <td>{creation_times[i]:.4f}</td>
                        <td>{nodes_per_second[i]:.1f}</td>
                    </tr>
        """

    html += f"""
                </table>

                <p class="summary">Scaling Factor: {scaling_factor:.2f} (1.0 is perfect linear scaling)</p>

                <div class="image-container">
                    <img src="tree_creation_performance.png" alt="Tree Creation Performance Graph">
                </div>
            </div>
    """

    return html


def _generate_tree_traversal_section(results: Dict[str, Any], output_dir: str) -> str:
    """Generate HTML section for tree traversal benchmark."""
    sizes = results["sizes"]
    traversal_times = results["traversal_times"]
    nodes_per_second = results["nodes_per_second"]
    scaling_factor = results["scaling_factor"]

    html = """
            <div class="section">
                <h2>Tree Traversal Performance</h2>
                <p>This benchmark measures how quickly trees of different sizes can be traversed.</p>

                <table>
                    <tr>
                        <th>Tree Size (nodes)</th>
                        <th>Traversal Time (seconds)</th>
                        <th>Performance (nodes/sec)</th>
                    </tr>
    """

    for i, size in enumerate(sizes):
        html += f"""
                    <tr>
                        <td>{size}</td>
                        <td>{traversal_times[i]:.4f}</td>
                        <td>{nodes_per_second[i]:.1f}</td>
                    </tr>
        """

    html += f"""
                </table>

                <p class="summary">Scaling Factor: {scaling_factor:.2f} (1.0 is perfect linear scaling)</p>

                <div class="image-container">
                    <img src="tree_traversal_performance.png" alt="Tree Traversal Performance Graph">
                </div>
            </div>
    """

    return html


def _generate_backend_operations_section(
    results: Dict[str, Any], output_dir: str
) -> str:
    """Generate HTML section for backend operations benchmark."""
    html = """
            <div class="section">
                <h2>Backend Operations Performance</h2>
                <p>This benchmark compares the performance of different numerical computation backends.</p>

                <div class="image-container">
                    <img src="backend_creation_comparison.png" alt="Tensor Creation Performance">
                    <p>Tensor Creation Performance Comparison</p>
                </div>

                <div class="image-container">
                    <img src="backend_softmax_comparison.png" alt="Softmax Operation Performance">
                    <p>Softmax Operation Performance Comparison</p>
                </div>

                <div class="image-container">
                    <img src="backend_weights_comparison.png" alt="Weight Calculation Performance">
                    <p>Weight Calculation Performance Comparison</p>
                </div>
            </div>
    """

    return html


def _generate_backend_switching_section(
    results: Dict[str, Any], output_dir: str
) -> str:
    """Generate HTML section for backend switching benchmark."""
    html = """
            <div class="section">
                <h2>Backend Switching Performance</h2>
                <p>This benchmark measures the time required to switch between different backends.</p>

                <div class="image-container">
                    <img src="backend_switching_performance.png" alt="Backend Switching Performance">
                </div>
            </div>
    """

    return html


def _generate_memory_usage_section(results: Dict[str, Any], output_dir: str) -> str:
    """Generate HTML section for memory usage benchmark."""
    html = """
            <div class="section">
                <h2>Memory Usage Analysis</h2>
                <p>This benchmark analyzes memory usage for different operations and tree sizes.</p>

                <div class="image-container">
                    <img src="memory_usage_comparison.png" alt="Memory Usage Comparison">
                </div>
            </div>
    """

    return html


def _generate_model_application_section(
    results: Dict[str, Any], output_dir: str
) -> str:
    """Generate HTML section for model application benchmark."""
    html = """
            <div class="section">
                <h2>Mental Health Model Application Performance</h2>
                <p>This benchmark measures the performance of applying different mental health models to trees.</p>

                <div class="image-container">
                    <img src="model_application_performance.png" alt="Model Application Performance">
                </div>
            </div>
    """

    return html


def _generate_visualization_section(results: Dict[str, Any], output_dir: str) -> str:
    """Generate HTML section for visualization benchmark."""
    sizes = results["sizes"]
    visualization_times = results["visualization_times"]
    nodes_per_second = results["nodes_per_second"]
    scaling_factor = results["scaling_factor"]

    html = """
            <div class="section">
                <h2>Visualization Performance</h2>
                <p>This benchmark measures the performance of tree visualization operations.</p>

                <table>
                    <tr>
                        <th>Tree Size (nodes)</th>
                        <th>Visualization Time (seconds)</th>
                        <th>Performance (nodes/sec)</th>
                    </tr>
    """

    for i, size in enumerate(sizes):
        html += f"""
                    <tr>
                        <td>{size}</td>
                        <td>{visualization_times[i]:.4f}</td>
                        <td>{nodes_per_second[i]:.1f}</td>
                    </tr>
        """

    html += f"""
                </table>

                <p class="summary">Scaling Factor: {scaling_factor:.2f} (1.0 is perfect linear scaling)</p>

                <div class="image-container">
                    <img src="visualization_performance.png" alt="Visualization Performance Graph">
                </div>
            </div>
    """

    return html


if __name__ == "__main__":
    main()
