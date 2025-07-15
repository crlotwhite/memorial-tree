"""
Tests for the backend_manager module of Memorial Tree.
"""

import unittest
import numpy as np

from src.memorial_tree.backends import BackendManager, BackendInterface, NumpyBackend


class TestBackendManager(unittest.TestCase):
    """Test cases for the BackendManager class."""

    def test_init(self):
        """Test initialization of BackendManager."""
        # Test with default backend (numpy)
        manager = BackendManager()
        self.assertEqual(manager.backend_type, "numpy")
        self.assertIsInstance(manager.backend, NumpyBackend)

        # Test with explicit numpy backend
        manager = BackendManager("numpy")
        self.assertEqual(manager.backend_type, "numpy")
        self.assertIsInstance(manager.backend, NumpyBackend)

        # Test with invalid backend type
        with self.assertRaises(ValueError):
            BackendManager("invalid_backend")

    def test_switch_backend(self):
        """Test switching backends."""
        manager = BackendManager("numpy")

        # Switch to the same backend (should be a no-op)
        manager.switch_backend("numpy")
        self.assertEqual(manager.backend_type, "numpy")
        self.assertIsInstance(manager.backend, NumpyBackend)

        # Test switching to an invalid backend
        with self.assertRaises(ValueError):
            manager.switch_backend("invalid_backend")

        # Note: We can't test switching to PyTorch or TensorFlow here
        # since we haven't implemented those backends yet

    def test_create_tensor(self):
        """Test creating a tensor."""
        manager = BackendManager()
        data = [1.0, 2.0, 3.0]
        tensor = manager.create_tensor(data)

        # Check that the tensor is a NumPy array with the correct values
        self.assertIsInstance(tensor, np.ndarray)
        np.testing.assert_array_equal(tensor, np.array(data, dtype=np.float32))

    def test_to_numpy(self):
        """Test converting a tensor to NumPy."""
        manager = BackendManager()
        data = [1.0, 2.0, 3.0]
        tensor = manager.create_tensor(data)

        # For NumPy backend, to_numpy should return the same array
        numpy_array = manager.to_numpy(tensor)
        self.assertIsInstance(numpy_array, np.ndarray)
        np.testing.assert_array_equal(numpy_array, np.array(data, dtype=np.float32))

    def test_from_numpy(self):
        """Test converting from NumPy to a tensor."""
        manager = BackendManager()
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # For NumPy backend, from_numpy should return the same array
        tensor = manager.from_numpy(data)
        self.assertIsInstance(tensor, np.ndarray)
        np.testing.assert_array_equal(tensor, data)

    def test_calculate_weights(self):
        """Test calculating weighted values."""
        manager = BackendManager()
        tensors = [
            manager.create_tensor([1.0, 2.0, 3.0]),
            manager.create_tensor([4.0, 5.0, 6.0]),
            manager.create_tensor([7.0, 8.0, 9.0]),
        ]

        # Test with default factors (all 1.0)
        result = manager.calculate_weights(tensors)
        expected = np.array([12.0, 15.0, 18.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        # Test with custom factors
        factors = [0.5, 1.0, 2.0]
        result = manager.calculate_weights(tensors, factors)
        expected = np.array(
            [
                0.5 * 1.0 + 1.0 * 4.0 + 2.0 * 7.0,
                0.5 * 2.0 + 1.0 * 5.0 + 2.0 * 8.0,
                0.5 * 3.0 + 1.0 * 6.0 + 2.0 * 9.0,
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_almost_equal(result, expected)

        # Test with empty list
        result = manager.calculate_weights([])
        self.assertEqual(len(result), 0)

        # Test with mismatched tensors and factors
        with self.assertRaises(ValueError):
            manager.calculate_weights(tensors, [1.0, 2.0])  # Missing one factor

    def test_apply_softmax(self):
        """Test applying softmax function."""
        manager = BackendManager()
        tensor = manager.create_tensor([1.0, 2.0, 3.0])

        # Test with default temperature
        result = manager.apply_softmax(tensor)

        # Calculate expected softmax manually
        exp_values = np.exp(
            np.array([1.0, 2.0, 3.0]) - 3.0
        )  # Subtract max for stability
        expected = exp_values / np.sum(exp_values)

        np.testing.assert_array_almost_equal(result, expected)

        # Check that result sums to 1
        self.assertAlmostEqual(np.sum(result), 1.0)

        # Test with different temperature
        result = manager.apply_softmax(tensor, temperature=2.0)

        # Calculate expected softmax manually with temperature
        exp_values = np.exp((np.array([1.0, 2.0, 3.0]) - 3.0) / 2.0)
        expected = exp_values / np.sum(exp_values)

        np.testing.assert_array_almost_equal(result, expected)

        # Check that result sums to 1
        self.assertAlmostEqual(np.sum(result), 1.0)

    def test_get_backend_name(self):
        """Test getting the backend name."""
        manager = BackendManager()
        self.assertEqual(manager.get_backend_name(), "numpy")


class TestNumpyBackend(unittest.TestCase):
    """Test cases for the NumpyBackend class."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = NumpyBackend()

    def test_interface_implementation(self):
        """Test that NumpyBackend implements BackendInterface."""
        self.assertIsInstance(self.backend, BackendInterface)

    def test_create_tensor(self):
        """Test creating a tensor."""
        data = [1.0, 2.0, 3.0]
        tensor = self.backend.create_tensor(data)

        self.assertIsInstance(tensor, np.ndarray)
        np.testing.assert_array_equal(tensor, np.array(data, dtype=np.float32))

    def test_to_numpy(self):
        """Test converting a tensor to NumPy."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # For NumPy backend, to_numpy should return the same array
        result = self.backend.to_numpy(data)
        self.assertIs(result, data)  # Should be the same object

    def test_from_numpy(self):
        """Test converting from NumPy to a tensor."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # For NumPy backend, from_numpy should return the same array
        result = self.backend.from_numpy(data)
        self.assertIs(result, data)  # Should be the same object

    def test_calculate_weights(self):
        """Test calculating weighted values."""
        tensors = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
            np.array([7.0, 8.0, 9.0], dtype=np.float32),
        ]

        # Test with default factors (all 1.0)
        result = self.backend.calculate_weights(tensors)
        expected = np.array([12.0, 15.0, 18.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

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

    def test_apply_softmax(self):
        """Test applying softmax function."""
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

    def test_get_backend_name(self):
        """Test getting the backend name."""
        self.assertEqual(self.backend.get_backend_name(), "numpy")


if __name__ == "__main__":
    unittest.main()
