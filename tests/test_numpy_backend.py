"""
Tests for the numpy_backend module of Memorial Tree.
"""

import unittest
import numpy as np

from src.memorial_tree.backends import NumpyBackend
from src.memorial_tree.backends.backend_manager import BackendInterface


class TestNumpyBackend(unittest.TestCase):
    """Test cases for the NumpyBackend class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.backend = NumpyBackend()

    def test_interface_implementation(self) -> None:
        """Test that NumpyBackend implements BackendInterface."""
        self.assertIsInstance(self.backend, BackendInterface)

    def test_create_tensor(self) -> None:
        """Test creating a tensor."""
        data = [1.0, 2.0, 3.0]
        tensor = self.backend.create_tensor(data)

        self.assertIsInstance(tensor, np.ndarray)
        np.testing.assert_array_equal(tensor, np.array(data, dtype=np.float32))
        self.assertEqual(tensor.dtype, np.float32)

    def test_to_numpy(self) -> None:
        """Test converting a tensor to NumPy."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # For NumPy backend, to_numpy should return the same array
        result = self.backend.to_numpy(data)
        self.assertIs(result, data)  # Should be the same object

    def test_from_numpy(self) -> None:
        """Test converting from NumPy to a tensor."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # For NumPy backend, from_numpy should return the same array
        result = self.backend.from_numpy(data)
        self.assertIs(result, data)  # Should be the same object

    def test_calculate_weights_default_factors(self) -> None:
        """Test calculating weighted values with default factors."""
        tensors = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
            np.array([7.0, 8.0, 9.0], dtype=np.float32),
        ]

        # Test with default factors (all 1.0)
        result = self.backend.calculate_weights(tensors)
        expected = np.array([12.0, 15.0, 18.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_calculate_weights_custom_factors(self) -> None:
        """Test calculating weighted values with custom factors."""
        tensors = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
            np.array([7.0, 8.0, 9.0], dtype=np.float32),
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
        np.testing.assert_array_almost_equal(result, expected)

    def test_calculate_weights_empty_list(self) -> None:
        """Test calculating weighted values with an empty list."""
        result = self.backend.calculate_weights([])
        self.assertEqual(len(result), 0)

    def test_calculate_weights_mismatched_factors(self) -> None:
        """Test calculating weighted values with mismatched factors."""
        tensors = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
            np.array([7.0, 8.0, 9.0], dtype=np.float32),
        ]

        # Test with mismatched tensors and factors
        with self.assertRaises(ValueError):
            self.backend.calculate_weights(tensors, [1.0, 2.0])  # Missing one factor

    def test_apply_softmax_default_temperature(self) -> None:
        """Test applying softmax function with default temperature."""
        tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Test with default temperature
        result = self.backend.apply_softmax(tensor)

        # Calculate expected softmax manually
        exp_values = np.exp(
            np.array([1.0, 2.0, 3.0]) - 3.0
        )  # Subtract max for stability
        expected = exp_values / np.sum(exp_values)

        np.testing.assert_array_almost_equal(result, expected)

        # Check that result sums to 1
        self.assertAlmostEqual(np.sum(result), 1.0)

    def test_apply_softmax_custom_temperature(self) -> None:
        """Test applying softmax function with custom temperature."""
        tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Test with different temperature
        temperature = 2.0
        result = self.backend.apply_softmax(tensor, temperature=temperature)

        # Calculate expected softmax manually with temperature
        scaled = np.array([1.0, 2.0, 3.0]) / temperature
        exp_values = np.exp(scaled - np.max(scaled))
        expected = exp_values / np.sum(exp_values)

        np.testing.assert_array_almost_equal(result, expected)

        # Check that result sums to 1
        self.assertAlmostEqual(np.sum(result), 1.0)

    def test_apply_softmax_zero_temperature(self) -> None:
        """Test applying softmax function with zero temperature."""
        tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Test with zero temperature (should use a small epsilon instead)
        result = self.backend.apply_softmax(tensor, temperature=0.0)

        # Should not raise an error and should still sum to 1
        self.assertAlmostEqual(np.sum(result), 1.0)

    def test_get_backend_name(self) -> None:
        """Test getting the backend name."""
        self.assertEqual(self.backend.get_backend_name(), "numpy")


if __name__ == "__main__":
    unittest.main()
