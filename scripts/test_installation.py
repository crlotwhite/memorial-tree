#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to test the installation of memorial-tree from PyPI or TestPyPI.
This script creates a virtual environment, installs the package, and tests importing it.

Usage:
    python scripts/test_installation.py [--test-pypi]
"""

import argparse
import os
import subprocess
import sys
import tempfile
import venv


def run_command(command, description=None):
    """Run a shell command and print its output."""
    if description:
        print(f"\n{description}...")

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    return result.stdout.strip()


def test_installation(use_test_pypi=False):
    """Test installing the package from PyPI or TestPyPI."""
    source = "TestPyPI" if use_test_pypi else "PyPI"
    print(f"\nTesting installation from {source}...")

    # Create a temporary directory for the virtual environment
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a virtual environment
        venv_path = os.path.join(temp_dir, "venv")
        venv.create(venv_path, with_pip=True)

        # Determine the path to the Python executable in the virtual environment
        if os.name == "nt":  # Windows
            python_path = os.path.join(venv_path, "Scripts", "python.exe")
        else:  # Unix/Linux/Mac
            python_path = os.path.join(venv_path, "bin", "python")

        # Install the package
        if use_test_pypi:
            install_cmd = f"{python_path} -m pip install --index-url https://test.pypi.org/simple/ memorial-tree"
        else:
            install_cmd = f"{python_path} -m pip install memorial-tree"

        print(f"Running: {install_cmd}")
        run_command(install_cmd)

        # Test importing the package
        import_cmd = f"{python_path} -c \"import memorial_tree; print('Package version:', memorial_tree.__version__)\""
        output = run_command(import_cmd)
        print(f"Import test result: {output}")

        # Test basic functionality
        test_cmd = f"{python_path} -c \"from memorial_tree import MemorialTree; tree = MemorialTree(); print('Created MemorialTree instance successfully')\""
        output = run_command(test_cmd)
        print(f"Functionality test result: {output}")

    print(f"\nSuccessfully installed and tested memorial-tree from {source}!")


def main():
    parser = argparse.ArgumentParser(description="Test memorial-tree installation")
    parser.add_argument("--test-pypi", action="store_true", help="Test installation from TestPyPI")
    args = parser.parse_args()

    test_installation(use_test_pypi=args.test_pypi)


if __name__ == "__main__":
    main()