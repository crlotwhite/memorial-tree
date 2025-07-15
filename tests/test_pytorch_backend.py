"""
Tests for the pytorch_backend module of Memorial Tree.
"""

import unittest
import numpy as np

from src.memorial_tree.backends import PyTorchBackend
from src.memorial_tree.backends.backend_manager import BackendInterface


class TestPyTorchBackend(unittest.TestCase):
    """Test cases for the PyTorchBackend class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        try:
            self.backend = PyTorchBackend()
            self.torch = self.backend.torch
        except ImportError:
            self.skipTest("PyTorch not installed, skipping tests")

    def test_interface_implementation(self) -> None:
        """Test that PyTorchBackend implements BackendInterface."""
        self.assertIsInstance(self.backend, BackendInterface)

    def test_create_tensor(self) -> None:
        """Test creating a tensor."""
        data = [1.0, 2.0, 3.0]
        tensor = self.backend.create_tensor(data)

        self.assertIsInstance(tensor, self.torch.Tensor)
        np.testing.assert_array_equal(
            self.backend.to_numpy(tensor), np.array(data, dtype=np.float32)
        )
        self.assertEqual(tensor.dtype, self.torch.float32)

    def test_to_numpy(self) -> None:
        """Test converting a tensor to NumPy."""
        data = [1.0, 2.0, 3.0]
        tensor = self.backend.create_tensor(data)

        result = self.backend.to_numpy(tensor)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(data, dtype=np.float32))

    def test_from_numpy(self) -> None:
        """Test converting from NumPy to a tensor."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result = self.backend.from_numpy(data)
        self.assertIsInstance(result, self.torch.Tensor)
        np.testing.assert_array_equal(self.backend.to_numpy(result), data)

    def test_calculate_weights_default_factors(self) -> None:
        """Test calculating weighted values with default factors."""
        tensors = [
            self.backend.create_tensor([1.0, 2.0, 3.0]),
            self.backend.create_tensor([4.0, 5.0, 6.0]),
            self.backend.create_tensor([7.0, 8.0, 9.0]),
        ]

        # Test with default factors (all 1.0)
        result = self.backend.calculate_weights(tensors)
        expected = np.array([12.0, 15.0, 18.0], dtype=np.float32)
        np.testing.assert_array_equal(self.backend.to_numpy(result), expected)

    def test_calculate_weights_custom_factors(self) -> None:
        """Test calculating weighted values with custom factors."""
        tensors = [
            self.backend.create_tensor([1.0, 2.0, 3.0]),
            self.backend.create_tensor([4.0, 5.0, 6.0]),
            self.backend.create_tensor([7.0, 8.0, 9.0]),
        ]

        # Test with custom factors
        factors = [0.5, 1.0, 2.0]
        result = self.backend.calculate_weights(tensors, factors)
        expected = np.array(
            [
                0.5 * 1.0 + 1.0 * 4.0 + 2.0 * 7.0,
                0.5 * 2.0 + 1.0 * 5.0 + 2.0 * 8.0,
                0.5 * 3.0 + 1.0 * 6.0 + 2.0 * 9.0,
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_almost_equal(self.backend.to_numpy(result), expected)

    def test_calculate_weights_empty_list(self) -> None:
        """Test calculating weighted values with an empty list."""
        result = self.backend.calculate_weights([])
        self.assertEqual(len(self.backend.to_numpy(result)), 0)

    def test_calculate_weights_mismatched_factors(self) -> None:
        """Test calculating weighted values with mismatched factors."""
        tensors = [
            self.backend.create_tensor([1.0, 2.0, 3.0]),
            self.backend.create_tensor([4.0, 5.0, 6.0]),
            self.backend.create_tensor([7.0, 8.0, 9.0]),
        ]

        # Test with mismatched tensors and factors
        with self.assertRaises(ValueError):
            self.backend.calculate_weights(tensors, [1.0, 2.0])  # Missing one factor

    def test_apply_softmax_default_temperature(self) -> None:
        """Test applying softmax function with default temperature."""
        tensor = self.backend.create_tensor([1.0, 2.0, 3.0])

        # Test with default temperature
        result = self.backend.apply_softmax(tensor)
        result_np = self.backend.to_numpy(result)

        # Calculate expected softmax manually
        exp_values = np.exp(
            np.array([1.0, 2.0, 3.0]) - 3.0
        )  # Subtract max for stability
        expected = exp_values / np.sum(exp_values)

        np.testing.assert_array_almost_equal(result_np, expected)

        # Check that result sums to 1
        self.assertAlmostEqual(np.sum(result_np), 1.0)

    def test_apply_softmax_custom_temperature(self) -> None:
        """Test applying softmax function with custom temperature."""
        tensor = self.backend.create_tensor([1.0, 2.0, 3.0])

        # Test with different temperature
        temperature = 2.0
        result = self.backend.apply_softmax(tensor, temperature=temperature)
        result_np = self.backend.to_numpy(result)

        # Calculate expected softmax manually with temperature
        scaled = np.array([1.0, 2.0, 3.0]) / temperature
        exp_values = np.exp(scaled - np.max(scaled))
        expected = exp_values / np.sum(exp_values)

        np.testing.assert_array_almost_equal(result_np, expected)

        # Check that result sums to 1
        self.assertAlmostEqual(np.sum(result_np), 1.0)

    def test_apply_softmax_zero_temperature(self) -> None:
        """Test applying softmax function with zero temperature."""
        tensor = self.backend.create_tensor([1.0, 2.0, 3.0])

        # Test with zero temperature (should use a small epsilon instead)
        result = self.backend.apply_softmax(tensor, temperature=0.0)
        result_np = self.backend.to_numpy(result)

        # Should not raise an error and should still sum to 1
        self.assertAlmostEqual(np.sum(result_np), 1.0)

    def test_get_backend_name(self) -> None:
        """Test getting the backend name."""
        self.assertEqual(self.backend.get_backend_name(), "pytorch")

    def test_enable_disable_grad(self) -> None:
        """Test enabling and disabling gradient computation."""
        # Save original state
        original_state = self.torch.is_grad_enabled()

        # Test enabling
        self.backend.enable_grad()
        self.assertTrue(self.torch.is_grad_enabled())

        # Test disabling
        self.backend.disable_grad()
        self.assertFalse(self.torch.is_grad_enabled())

        # Restore original state
        if original_state:
            self.torch.set_grad_enabled(True)
        else:
            self.torch.set_grad_enabled(False)

    def test_use_gpu(self) -> None:
        """Test GPU usage setting."""
        # This test just verifies the method runs without error
        # Actual GPU usage depends on hardware availability
        result = self.backend.use_gpu(True)

        # If CUDA is available, device should be cuda
        if self.torch.cuda.is_available():
            self.assertTrue(result)
            self.assertEqual(self.backend.device.type, "cuda")
        else:
            self.assertFalse(result)
            self.assertEqual(self.backend.device.type, "cpu")

        # Test disabling GPU
        self.backend.use_gpu(False)
        self.assertEqual(self.backend.device.type, "cpu")

    def test_mixed_tensor_types(self) -> None:
        """Test handling of mixed tensor types in calculate_weights."""
        # Create tensors of different types
        torch_tensor = self.backend.create_tensor([1.0, 2.0, 3.0])
        numpy_array = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        # Mix them in the input
        tensors = [torch_tensor, numpy_array]

        # Should handle the conversion internally
        result = self.backend.calculate_weights(tensors)
        expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(self.backend.to_numpy(result), expected)

    def test_gradient_computation(self) -> None:
        """Test gradient computation with PyTorch tensors."""
        # Skip if running on CPU as this is just a basic test
        if not self.torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping gradient test")

        # Enable gradients
        self.backend.enable_grad()

        # Create tensor with requires_grad=True
        x = self.backend.create_tensor([1.0, 2.0, 3.0])
        x.requires_grad_(True)

        # Simple operation
        y = x * x

        # Sum to get scalar
        z = y.sum()

        # Backward pass
        z.backward()

        # Check gradients (should be 2*x)
        expected_grad = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(
            self.backend.to_numpy(x.grad), expected_grad
        )

        # Disable gradients
        self.backend.disable_grad()


if __name__ == "__main__":
    unittest.main()
