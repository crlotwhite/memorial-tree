"""
Integration tests for the Memorial Tree system.

These tests verify the entire workflow from tree creation to modeling and visualization,
ensuring all components work together correctly.
"""

import unittest
import os
import tempfile
import shutil
import gc
import sys
import time
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import memory_usage

from src.memorial_tree.core import MemorialTree
from src.memorial_tree.models import ADHDModel, DepressionModel, AnxietyModel
from src.memorial_tree.visualization import TreeVisualizer, PathAnalyzer
from src.memorial_tree.models.model_comparison import ModelComparison


class TestFullWorkflow(unittest.TestCase):
    """Test the full workflow from tree creation to visualization."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
        # Close all matplotlib figures
        plt.close("all")
        # Force garbage collection to clean up resources
        gc.collect()

    def test_basic_workflow(self):
        """Test the basic workflow: create tree, add nodes, make decisions, visualize."""
        # 1. Create a tree
        tree = MemorialTree("Should I learn a new programming language?")

        # 2. Add thoughts (choices)
        yes_id = tree.add_thought(
            parent_id=tree.root.node_id,
            content="Yes, I should learn a new language",
            weight=0.7,
        ).node_id

        no_id = tree.add_thought(
            parent_id=tree.root.node_id,
            content="No, I should focus on what I know",
            weight=0.3,
        ).node_id

        # Add second-level thoughts to "Yes"
        python_id = tree.add_thought(
            parent_id=yes_id, content="Learn Python", weight=0.8
        ).node_id

        js_id = tree.add_thought(
            parent_id=yes_id, content="Learn JavaScript", weight=0.6
        ).node_id

        rust_id = tree.add_thought(
            parent_id=yes_id, content="Learn Rust", weight=0.4
        ).node_id

        # Add second-level thoughts to "No"
        tree.add_thought(parent_id=no_id, content="Improve existing skills", weight=0.7)

        tree.add_thought(parent_id=no_id, content="Work on projects", weight=0.6)

        # 3. Add a ghost node
        tree.add_ghost_node(
            content="Fear of wasting time", influence=0.5, visibility=0.3
        )

        # 4. Make decisions
        tree.make_choice(yes_id)
        tree.make_choice(python_id)

        # 5. Get the path
        path = tree.get_path_from_root()
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0].content, "Should I learn a new programming language?")
        self.assertEqual(path[1].content, "Yes, I should learn a new language")
        self.assertEqual(path[2].content, "Learn Python")

        # 6. Visualize the tree
        visualizer = TreeVisualizer(output_dir=self.test_dir)
        path_ids = [node.node_id for node in path]

        # Test tree visualization
        fig = visualizer.visualize_tree(
            tree=tree,
            highlight_path=path_ids,
            show_ghost_nodes=True,
            save_path=os.path.join(self.test_dir, "test_tree.png"),
        )
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_tree.png")))
        plt.close(fig)

        # Test path visualization
        path_fig = visualizer.visualize_path(
            tree=tree,
            path=path_ids,
            save_path=os.path.join(self.test_dir, "test_path.png"),
        )
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_path.png")))
        plt.close(path_fig)

        # Test ghost influence visualization
        ghost_fig = visualizer.visualize_ghost_influence(
            tree=tree, save_path=os.path.join(self.test_dir, "test_ghost.png")
        )
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_ghost.png")))
        plt.close(ghost_fig)

    def test_mental_health_models_workflow(self):
        """Test workflow with mental health models applied."""
        # 1. Create a tree
        tree = MemorialTree("일상 계획")

        # 2. Add thoughts
        work = tree.add_thought(tree.root.node_id, "업무 시작하기", weight=2.0)
        social = tree.add_thought(tree.root.node_id, "친구 만나기", weight=1.5)
        hobby = tree.add_thought(tree.root.node_id, "취미 활동하기", weight=1.0)

        # Store original weights for comparison
        original_weights = {
            work.node_id: work.weight,
            social.node_id: social.weight,
            hobby.node_id: hobby.weight,
        }

        # 3. Create and apply ADHD model
        adhd_model = ADHDModel(
            attention_span=0.3, impulsivity=0.8, distraction_rate=0.6, hyperactivity=0.7
        )

        # Apply the model
        adhd_model.modify_decision_process(tree, tree.current_node)

        # Verify model effects were recorded
        self.assertIn("adhd_modified_weights", tree.metadata)
        modified_weights = tree.metadata["adhd_modified_weights"]

        # Check that weights were modified
        for node_id, original_weight in original_weights.items():
            self.assertIn(node_id, modified_weights)
            # Weights should be different after model application
            self.assertNotEqual(modified_weights[node_id], original_weight)

        # 4. Reset and apply Depression model
        tree = MemorialTree("일상 계획")
        work = tree.add_thought(tree.root.node_id, "업무 시작하기", weight=2.0)
        social = tree.add_thought(tree.root.node_id, "친구 만나기", weight=1.5)
        hobby = tree.add_thought(tree.root.node_id, "취미 활동하기", weight=1.0)

        depression_model = DepressionModel(
            negative_bias=0.7, decision_delay=2.0, energy_level=0.3
        )

        depression_model.modify_decision_process(tree, tree.current_node)

        # Verify model effects were recorded
        self.assertIn("depression_modified_weights", tree.metadata)

        # 5. Reset and apply Anxiety model
        tree = MemorialTree("일상 계획")
        work = tree.add_thought(tree.root.node_id, "업무 시작하기", weight=2.0)
        social = tree.add_thought(tree.root.node_id, "친구 만나기", weight=1.5)
        hobby = tree.add_thought(tree.root.node_id, "취미 활동하기", weight=1.0)

        anxiety_model = AnxietyModel(
            worry_amplification=0.8, risk_aversion=0.9, rumination_cycles=3
        )

        anxiety_model.modify_decision_process(tree, tree.current_node)

        # Verify model effects were recorded
        self.assertIn("anxiety_modified_weights", tree.metadata)

        # 6. Test model comparison
        comparison = ModelComparison()

        # Create trees for each model
        control_tree = MemorialTree("일상 계획")
        control_tree.add_thought(control_tree.root.node_id, "업무 시작하기", weight=2.0)
        control_tree.add_thought(control_tree.root.node_id, "친구 만나기", weight=1.5)
        control_tree.add_thought(control_tree.root.node_id, "취미 활동하기", weight=1.0)

        adhd_tree = MemorialTree("일상 계획")
        adhd_tree.add_thought(adhd_tree.root.node_id, "업무 시작하기", weight=2.0)
        adhd_tree.add_thought(adhd_tree.root.node_id, "친구 만나기", weight=1.5)
        adhd_tree.add_thought(adhd_tree.root.node_id, "취미 활동하기", weight=1.0)
        adhd_model.modify_decision_process(adhd_tree, adhd_tree.current_node)

        depression_tree = MemorialTree("일상 계획")
        depression_tree.add_thought(
            depression_tree.root.node_id, "업무 시작하기", weight=2.0
        )
        depression_tree.add_thought(
            depression_tree.root.node_id, "친구 만나기", weight=1.5
        )
        depression_tree.add_thought(
            depression_tree.root.node_id, "취미 활동하기", weight=1.0
        )
        depression_model.modify_decision_process(
            depression_tree, depression_tree.current_node
        )

        anxiety_tree = MemorialTree("일상 계획")
        anxiety_tree.add_thought(anxiety_tree.root.node_id, "업무 시작하기", weight=2.0)
        anxiety_tree.add_thought(anxiety_tree.root.node_id, "친구 만나기", weight=1.5)
        anxiety_tree.add_thought(anxiety_tree.root.node_id, "취미 활동하기", weight=1.0)
        anxiety_model.modify_decision_process(anxiety_tree, anxiety_tree.current_node)

        # Compare models
        comparison.add_model("Control", control_tree)
        comparison.add_model("ADHD", adhd_tree)
        comparison.add_model("Depression", depression_tree)
        comparison.add_model("Anxiety", anxiety_tree)

        # Generate comparison visualization
        fig = comparison.visualize_weight_comparison(
            save_path=os.path.join(self.test_dir, "model_comparison.png")
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "model_comparison.png"))
        )
        plt.close(fig)

    def test_backend_switching(self):
        """Test workflow with backend switching."""
        # Skip test if PyTorch or TensorFlow is not available
        try:
            import torch
            import tensorflow as tf
        except ImportError:
            self.skipTest("PyTorch or TensorFlow not available")

        # 1. Create trees with different backends
        tree_numpy = MemorialTree("Backend Test", backend="numpy")
        tree_pytorch = MemorialTree("Backend Test", backend="pytorch")
        tree_tensorflow = MemorialTree("Backend Test", backend="tensorflow")

        # 2. Add identical structure to all trees
        for tree in [tree_numpy, tree_pytorch, tree_tensorflow]:
            yes = tree.add_thought(tree.root.node_id, "Yes", weight=0.7)
            no = tree.add_thought(tree.root.node_id, "No", weight=0.3)

            tree.add_thought(yes.node_id, "Yes-Option1", weight=0.6)
            tree.add_thought(yes.node_id, "Yes-Option2", weight=0.4)

            tree.add_thought(no.node_id, "No-Option1", weight=0.5)
            tree.add_thought(no.node_id, "No-Option2", weight=0.5)

            tree.add_ghost_node("Ghost", influence=0.4)

        # 3. Switch backends
        tree_numpy.switch_backend("pytorch")
        self.assertEqual(tree_numpy.backend_manager.get_backend_name(), "pytorch")

        tree_pytorch.switch_backend("tensorflow")
        self.assertEqual(tree_pytorch.backend_manager.get_backend_name(), "tensorflow")

        tree_tensorflow.switch_backend("numpy")
        self.assertEqual(tree_tensorflow.backend_manager.get_backend_name(), "numpy")

        # 4. Verify tree structure is preserved after switching
        for tree in [tree_numpy, tree_pytorch, tree_tensorflow]:
            self.assertEqual(tree.get_tree_size(), 7)  # root + 6 nodes
            self.assertEqual(len(tree.ghost_nodes), 1)

            # Check children of root
            root_children = tree.root.children
            self.assertEqual(len(root_children), 2)

            # Check content of children
            child_contents = [child.content for child in root_children]
            self.assertIn("Yes", child_contents)
            self.assertIn("No", child_contents)

    def test_path_analysis_workflow(self):
        """Test the path analysis workflow."""
        # 1. Create a tree with multiple paths
        tree = MemorialTree("Career Decision")

        # First level choices
        stay_id = tree.add_thought(
            tree.root.node_id, "Stay at current job", weight=0.6
        ).node_id
        change_id = tree.add_thought(
            tree.root.node_id, "Change career", weight=0.4
        ).node_id

        # Second level for "Stay"
        promotion_id = tree.add_thought(stay_id, "Seek promotion", weight=0.7).node_id
        lateral_id = tree.add_thought(stay_id, "Make lateral move", weight=0.3).node_id

        # Second level for "Change"
        tech_id = tree.add_thought(
            change_id, "Move to tech industry", weight=0.8
        ).node_id
        edu_id = tree.add_thought(change_id, "Go into education", weight=0.5).node_id

        # Add ghost nodes
        tree.add_ghost_node("Fear of change", influence=0.6, visibility=0.3)
        tree.add_ghost_node("Desire for higher income", influence=0.7, visibility=0.4)

        # 2. Create a path by making choices
        tree.make_choice(change_id)
        tree.make_choice(tech_id)

        # 3. Create a path analyzer
        path_analyzer = PathAnalyzer()

        # 4. Analyze the current path
        path_analysis = path_analyzer.analyze_path(tree)

        # 5. Verify analysis results
        self.assertIn("path_length", path_analysis)
        self.assertEqual(path_analysis["path_length"], 3)  # root + 2 choices

        self.assertIn("decision_confidence", path_analysis)
        self.assertIn("ghost_influence", path_analysis)
        self.assertIn("path_nodes", path_analysis)

        # 6. Generate path analysis visualization
        fig = path_analyzer.visualize_path_analysis(
            tree, save_path=os.path.join(self.test_dir, "path_analysis.png")
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "path_analysis.png"))
        )
        plt.close(fig)

        # 7. Compare multiple paths
        # Reset and create a different path
        tree.reset_to_root()
        tree.make_choice(stay_id)
        tree.make_choice(promotion_id)

        # Add this path to the analyzer
        path_analyzer.add_path("Career Change", path_analysis)
        new_path_analysis = path_analyzer.analyze_path(tree)
        path_analyzer.add_path("Stay and Promote", new_path_analysis)

        # Compare paths
        comparison_fig = path_analyzer.compare_paths(
            save_path=os.path.join(self.test_dir, "path_comparison.png")
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "path_comparison.png"))
        )
        plt.close(comparison_fig)


class TestSystemStability(unittest.TestCase):
    """Test system stability under various scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)
        plt.close("all")
        gc.collect()

    def test_large_tree_performance(self):
        """Test performance with a large tree."""
        # Create a large tree
        tree = MemorialTree("Large Tree Root")

        # Track creation time
        start_time = time.time()

        # Create a tree with many nodes (3 levels, branching factor of 5)
        level1_nodes = []
        for i in range(5):
            node = tree.add_thought(tree.root.node_id, f"Level1-{i}", weight=1.0)
            level1_nodes.append(node)

        level2_nodes = []
        for parent in level1_nodes:
            for i in range(5):
                node = tree.add_thought(
                    parent.node_id, f"Level2-{parent.content}-{i}", weight=1.0
                )
                level2_nodes.append(node)

        for parent in level2_nodes:
            for i in range(5):
                tree.add_thought(
                    parent.node_id, f"Level3-{parent.content}-{i}", weight=1.0
                )

        # Add some ghost nodes
        for i in range(10):
            tree.add_ghost_node(f"Ghost-{i}", influence=0.3, visibility=0.2)

        creation_time = time.time() - start_time
        print(f"Large tree creation time: {creation_time:.2f} seconds")

        # Verify tree size
        self.assertEqual(tree.get_tree_size(), 156)  # 1 + 5 + 25 + 125 + 10 ghost nodes

        # Test traversal performance
        start_time = time.time()
        all_nodes = tree.get_all_nodes()
        traversal_time = time.time() - start_time
        print(f"Large tree traversal time: {traversal_time:.2f} seconds")

        # Test visualization performance
        visualizer = TreeVisualizer(output_dir=self.test_dir)

        start_time = time.time()
        fig = visualizer.visualize_tree(
            tree=tree, save_path=os.path.join(self.test_dir, "large_tree.png")
        )
        visualization_time = time.time() - start_time
        print(f"Large tree visualization time: {visualization_time:.2f} seconds")
        plt.close(fig)

        # Performance should be reasonable
        self.assertLess(creation_time, 5.0, "Tree creation took too long")
        self.assertLess(traversal_time, 1.0, "Tree traversal took too long")
        self.assertLess(visualization_time, 10.0, "Tree visualization took too long")

    def test_memory_usage(self):
        """Test memory usage during tree operations."""

        def create_and_operate_on_tree():
            # Create a medium-sized tree
            tree = MemorialTree("Memory Test Root")

            # Add three levels with branching factor of 3
            level1_nodes = []
            for i in range(3):
                node = tree.add_thought(tree.root.node_id, f"Level1-{i}", weight=1.0)
                level1_nodes.append(node)

            level2_nodes = []
            for parent in level1_nodes:
                for i in range(3):
                    node = tree.add_thought(parent.node_id, f"Level2-{i}", weight=1.0)
                    level2_nodes.append(node)

            for parent in level2_nodes:
                for i in range(3):
                    tree.add_thought(parent.node_id, f"Level3-{i}", weight=1.0)

            # Add ghost nodes
            for i in range(5):
                tree.add_ghost_node(f"Ghost-{i}", influence=0.3)

            # Perform operations
            tree.get_all_nodes()
            tree.get_tree_depth()

            # Navigate through the tree
            for _ in range(10):
                choices = tree.get_available_choices()
                if choices:
                    tree.make_choice(choices[0].node_id)

            # Create visualizer and generate visualization
            visualizer = TreeVisualizer(output_dir=self.test_dir)
            fig = visualizer.visualize_tree(tree=tree)
            plt.close(fig)

            # Apply mental health model
            model = ADHDModel()
            model.modify_decision_process(tree, tree.current_node)

            # Force garbage collection to ensure accurate memory measurement
            gc.collect()

            # Hold the tree in memory for a moment
            time.sleep(1)

        # Measure memory usage
        mem_usage = memory_usage(create_and_operate_on_tree)

        # Print memory usage statistics
        peak_memory = max(mem_usage)
        baseline_memory = min(mem_usage)
        memory_increase = peak_memory - baseline_memory

        print(f"Baseline memory: {baseline_memory:.2f} MiB")
        print(f"Peak memory: {peak_memory:.2f} MiB")
        print(f"Memory increase: {memory_increase:.2f} MiB")

        # Check for memory leaks - increase should be reasonable
        self.assertLess(memory_increase, 100.0, "Excessive memory usage detected")

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        tree = MemorialTree("Error Test")

        # Test handling of invalid node IDs
        with self.assertRaises(Exception):
            tree.make_choice("non-existent-id")

        with self.assertRaises(Exception):
            tree.add_thought("non-existent-id", "This should fail")

        # Test handling of invalid backend switching
        with self.assertRaises(Exception):
            tree.switch_backend("invalid-backend")

        # Test handling of circular references
        child = tree.add_thought(tree.root.node_id, "Child")
        grandchild = tree.add_thought(child.node_id, "Grandchild")

        # Attempt to create a circular reference by making grandchild a parent of root
        # This should be prevented by the system
        with self.assertRaises(Exception):
            # We need to bypass normal API to attempt this invalid operation
            grandchild.add_child(tree.root)

        # Test visualization with invalid parameters
        visualizer = TreeVisualizer(output_dir=self.test_dir)

        # Invalid path
        with self.assertRaises(Exception):
            visualizer.visualize_path(tree, ["non-existent-id"])

        # Invalid layout type
        with self.assertRaises(Exception):
            visualizer.visualize_tree(tree, layout_type="invalid-layout")

    def test_complex_scenario(self):
        """Test a complex scenario combining multiple features."""
        # 1. Create a tree with multiple levels
        tree = MemorialTree("Complex Decision", backend="numpy")

        # Add first level choices
        option_a = tree.add_thought(tree.root.node_id, "Option A", weight=0.7)
        option_b = tree.add_thought(tree.root.node_id, "Option B", weight=0.5)
        option_c = tree.add_thought(tree.root.node_id, "Option C", weight=0.3)

        # Add second level choices
        for parent in [option_a, option_b, option_c]:
            for i in range(3):
                tree.add_thought(
                    parent.node_id, f"{parent.content}-{i+1}", weight=0.5 + i * 0.1
                )

        # Add ghost nodes with different influences
        tree.add_ghost_node("Ghost 1", influence=0.7, visibility=0.2)
        tree.add_ghost_node("Ghost 2", influence=0.4, visibility=0.3)
        tree.add_ghost_node("Ghost 3", influence=0.6, visibility=0.1)

        # 2. Switch backend to PyTorch
        try:
            import torch

            tree.switch_backend("pytorch")
        except ImportError:
            print("PyTorch not available, skipping backend switch")

        # 3. Make some decisions
        tree.make_choice(option_b.node_id)
        choices = tree.get_available_choices()
        tree.make_choice(choices[1].node_id)

        # 4. Apply a mental health model
        adhd_model = ADHDModel()
        adhd_model.modify_decision_process(tree, tree.current_node)

        # 5. Switch backend to NumPy
        tree.switch_backend("numpy")

        # 6. Create visualizations
        visualizer = TreeVisualizer(output_dir=self.test_dir)

        # Basic tree visualization
        fig1 = visualizer.visualize_tree(
            tree=tree,
            highlight_path=tree.path_history,
            show_ghost_nodes=True,
            save_path=os.path.join(self.test_dir, "complex_tree.png"),
        )
        plt.close(fig1)

        # Ghost influence visualization
        fig2 = visualizer.visualize_ghost_influence(
            tree=tree, save_path=os.path.join(self.test_dir, "complex_ghost.png")
        )
        plt.close(fig2)

        # 7. Analyze the path
        path_analyzer = PathAnalyzer()
        analysis = path_analyzer.analyze_path(tree)

        fig3 = path_analyzer.visualize_path_analysis(
            tree, save_path=os.path.join(self.test_dir, "complex_path_analysis.png")
        )
        plt.close(fig3)

        # 8. Verify all components worked together
        self.assertEqual(tree.backend_manager.get_backend_name(), "numpy")
        self.assertEqual(len(tree.path_history), 3)  # root + 2 choices
        self.assertIn("adhd_modified_weights", tree.metadata)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "complex_tree.png")))
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "complex_ghost.png"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "complex_path_analysis.png"))
        )


if __name__ == "__main__":
    unittest.main()
