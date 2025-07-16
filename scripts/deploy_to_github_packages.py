#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to deploy the memorial-tree package to GitHub Packages.
This script handles:
1. Building the package
2. Configuring GitHub authentication
3. Uploading to GitHub Packages

Usage:
    python scripts/deploy_to_github_packages.py [--version VERSION]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple, Set, Callable, TypeVar, Generic, cast, Literal, NoReturn, overload, Final, ClassVar, Protocol, runtime_checkable, NamedTuple, TypedDict, Awaitable, AsyncIterator, AsyncGenerator, Coroutine, AsyncContextManager, ContextManager, Generator, Iterator, Iterable, Mapping, MutableMapping, Sequence, MutableSequence, AbstractSet, MutableSet, Container, Collection, Sized, Reversible, Type, Any, AnyStr, Text, IO, TextIO, BinaryIO, Pattern, Match, SupportsInt, SupportsFloat, SupportsComplex, SupportsBytes, SupportsAbs, SupportsRound, SupportsIndex, SupportsNext, Hashable, Sized, ByteString, SupportsFloat, SupportsInt, SupportsBytes, SupportsAbs, SupportsRound


def run_command(command: str, description: Optional[str] = None) -> str:
    """Run a shell command and print its output.

    Args:
        command: The command to run
        description: Optional description to print before running the command

    Returns:
        The command output as a string

    Raises:
        SystemExit: If the command fails
    """
    if description:
        print(f"\n{description}...")

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    return result.stdout.strip()


def clean_build_directories() -> None:
    """Clean up build directories."""
    print("\nCleaning build directories...")
    directories = ["build", "dist", "*.egg-info"]
    for directory in directories:
        run_command(f"rm -rf {directory}")


def build_package() -> None:
    """Build the package."""
    print("\nBuilding package...")
    run_command("python -m pip install --upgrade pip build twine")
    run_command("python -m build")


def check_package() -> None:
    """Check the package with twine."""
    print("\nChecking package...")
    result = run_command("twine check dist/*")
    print(result)


def configure_github_auth() -> None:
    """Configure GitHub authentication for package upload."""
    print("\nConfiguring GitHub authentication...")

    # Check if GitHub username and token are available in environment variables
    github_username = os.environ.get("GITHUB_USERNAME")
    github_token = os.environ.get("GITHUB_TOKEN")

    if not github_username or not github_token:
        print("GitHub username and/or token not found in environment variables.")
        print("Please set GITHUB_USERNAME and GITHUB_TOKEN environment variables.")
        print("Alternatively, you can configure ~/.pypirc manually.")

        # Ask if user wants to configure manually
        configure_manually = input("Do you want to configure authentication manually? (y/n): ")
        if configure_manually.lower() != "y":
            sys.exit(1)

        github_username = input("Enter your GitHub username: ")
        github_token = input("Enter your GitHub personal access token: ")

    # Create or update ~/.pypirc file
    pypirc_path = Path.home() / ".pypirc"

    pypirc_content = f"""[distutils]
index-servers =
    github

[github]
repository = https://github.com/memorialtree/memorial-tree/
username = {github_username}
password = {github_token}
"""

    with open(pypirc_path, "w") as f:
        f.write(pypirc_content)

    # Set permissions to user read/write only
    os.chmod(pypirc_path, 0o600)

    print(f"GitHub authentication configured in {pypirc_path}")


def upload_to_github_packages() -> None:
    """Upload the package to GitHub Packages."""
    print("\nUploading to GitHub Packages...")
    run_command("twine upload --repository github dist/*")
    print("\nPackage uploaded to GitHub Packages!")

    # Get the package version
    version = run_command("python -c \"import setuptools_scm; print(setuptools_scm.get_version())\"")

    print("\nYou can install it with:")
    print(f"pip install --index-url https://github.com/memorialtree/memorial-tree/raw/main/dist/ memorial-tree=={version}")
    print("\nOr add to your requirements.txt:")
    print(f"memorial-tree @ https://github.com/memorialtree/memorial-tree/releases/download/v{version}/memorial_tree-{version}-py3-none-any.whl")


def main() -> None:
    """Main function to deploy to GitHub Packages."""
    parser = argparse.ArgumentParser(description="Deploy memorial-tree to GitHub Packages")
    parser.add_argument("--version", help="Version to deploy (optional)")
    args = parser.parse_args()

    if args.version:
        # Set version environment variable for setuptools_scm
        os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = args.version

    clean_build_directories()
    build_package()
    check_package()
    configure_github_auth()

    # Confirm before uploading
    confirmation = input("\nDo you want to upload to GitHub Packages? (y/n): ")
    if confirmation.lower() == "y":
        upload_to_github_packages()
    else:
        print("Skipping upload to GitHub Packages.")

    print("\nDeployment process completed!")


if __name__ == "__main__":
    main()