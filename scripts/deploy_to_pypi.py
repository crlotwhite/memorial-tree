#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to deploy the memorial-tree package to PyPI.
This script handles:
1. Building the package
2. Testing the package on TestPyPI
3. Deploying to the official PyPI

Usage:
    python scripts/deploy_to_pypi.py [--test-only]
"""

import argparse
import os
import subprocess
import sys
import time


def run_command(command, description=None):
    """Run a shell command and print its output."""
    if description:
        print(f"\n{description}...")

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    return result.stdout.strip()


def clean_build_directories():
    """Clean up build directories."""
    print("\nCleaning build directories...")
    directories = ["build", "dist", "*.egg-info"]
    for directory in directories:
        run_command(f"rm -rf {directory}")


def build_package():
    """Build the package."""
    print("\nBuilding package...")
    run_command("python3 -m pip install --upgrade pip build twine")
    run_command("python3 -m build")


def check_package():
    """Check the package with twine."""
    print("\nChecking package...")
    result = run_command("twine check dist/*")
    print(result)


def upload_to_test_pypi():
    """Upload the package to TestPyPI."""
    print("\nUploading to TestPyPI...")
    run_command("twine upload --repository-url https://test.pypi.org/legacy/ dist/*")
    print("\nPackage uploaded to TestPyPI!")
    print("You can install it with:")
    print("pip install --index-url https://test.pypi.org/simple/ memorial-tree")


def test_installation_from_test_pypi():
    """Test installing the package from TestPyPI."""
    print("\nTesting installation from TestPyPI...")

    # Create a temporary virtual environment
    run_command("python3 -m venv test_env")

    # Activate the virtual environment and install the package
    if os.name == "nt":  # Windows
        activate_cmd = "test_env\\Scripts\\activate"
    else:  # Unix/Linux/Mac
        activate_cmd = "source test_env/bin/activate"

    # Wait for TestPyPI to process the upload
    print("Waiting for TestPyPI to process the upload (30 seconds)...")
    time.sleep(30)

    # Install the package from TestPyPI
    install_cmd = f"{activate_cmd} && pip install --index-url https://test.pypi.org/simple/ memorial-tree"
    run_command(install_cmd)

    # Test importing the package
    import_test = f"{activate_cmd} && python3 -c \"import memorial_tree; print(memorial_tree.__version__)\""
    version = run_command(import_test)
    print(f"Successfully installed and imported memorial-tree version {version}")

    # Clean up the virtual environment
    run_command("rm -rf test_env")


def upload_to_pypi():
    """Upload the package to PyPI."""
    print("\nUploading to PyPI...")
    run_command("twine upload dist/*")
    print("\nPackage uploaded to PyPI!")
    print("You can install it with:")
    print("pip install memorial-tree")


def main():
    parser = argparse.ArgumentParser(description="Deploy memorial-tree to PyPI")
    parser.add_argument("--test-only", action="store_true", help="Only deploy to TestPyPI")
    args = parser.parse_args()

    clean_build_directories()
    build_package()
    check_package()

    upload_to_test_pypi()
    test_installation_from_test_pypi()

    if not args.test_only:
        confirmation = input("\nDo you want to upload to the official PyPI? (y/n): ")
        if confirmation.lower() == "y":
            upload_to_pypi()
        else:
            print("Skipping upload to official PyPI.")

    print("\nDeployment process completed!")


if __name__ == "__main__":
    main()