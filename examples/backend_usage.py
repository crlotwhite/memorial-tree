"""
Example demonstrating the use of the BackendManager.

This example shows how to create and use the BackendManager with the NumPy backend.
"""

import sys
import os
import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.memorial_tree import BackendManager


def main():
    """Demonstrate BackendManager usage."""
    print("Memorial Tree - BackendManager Example")
    print("=====================================")

    # Create a BackendManager with the default NumPy backend
    manager = BackendManager()
    print(f"Using backend: {manager.get_backend_name()}")

    # Create some tensors
    data1 = [1.0, 2.0, 3.0]
    data2 = [4.0, 5.0, 6.0]

    tensor1 = manager.create_tensor(data1)
    tensor2 = manager.create_tensor(data2)

    print(f"Tensor 1: {tensor1}")
    print(f"Tensor 2: {tensor2}")

    # Calculate weighted values
    factors = [0.5, 1.5]
    weighted = manager.calculate_weights([tensor1, tensor2], factors)
    print(f"Weighted result: {weighted}")

    # Apply softmax
    logits = manager.create_tensor([2.0, 1.0, 0.5])
    probabilities = manager.apply_softmax(logits)
    print(f"Logits: {logits}")
    print(f"Probabilities after softmax: {probabilities}")
    print(f"Sum of probabilities: {np.sum(probabilities)}")

    # Try different temperature
    probabilities_t2 = manager.apply_softmax(logits, temperature=2.0)
    print(f"Probabilities with temperature=2.0: {probabilities_t2}")
    print(f"Sum of probabilities: {np.sum(probabilities_t2)}")

    print("\nBackendManager successfully demonstrated!")


if __name__ == "__main__":
    main()
