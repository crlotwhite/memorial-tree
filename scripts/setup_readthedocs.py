#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to help set up Read the Docs for the memorial-tree project.
This script provides instructions and checks for Read the Docs configuration.

Usage:
    python3 scripts/setup_readthedocs.py
"""

import os
import sys
import webbrowser
from pathlib import Path


def check_readthedocs_config():
    """Check if the Read the Docs configuration file exists."""
    config_path = Path(".readthedocs.yml")
    if not config_path.exists():
        print("Error: .readthedocs.yml file not found!")
        print("Please create the file with the following content:")
        print("""
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.10"

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs

sphinx:
  configuration: docs/source/conf.py
  fail_on_warning: false

formats:
  - pdf
  - epub
        """)
        return False

    print("✅ .readthedocs.yml file found.")
    return True


def check_sphinx_config():
    """Check if the Sphinx configuration file exists."""
    config_path = Path("docs/source/conf.py")
    if not config_path.exists():
        print("Error: docs/source/conf.py file not found!")
        print("Please make sure the Sphinx documentation is properly set up.")
        return False

    print("✅ Sphinx configuration file found.")
    return True


def check_github_workflow():
    """Check if the GitHub workflow for documentation exists."""
    workflow_path = Path(".github/workflows/docs.yml")
    if not workflow_path.exists():
        print("Error: .github/workflows/docs.yml file not found!")
        print("Please create a GitHub workflow for documentation deployment.")
        return False

    print("✅ GitHub workflow for documentation found.")
    return True


def provide_readthedocs_instructions():
    """Provide instructions for setting up Read the Docs."""
    print("\n=== Read the Docs Setup Instructions ===\n")
    print("1. Go to https://readthedocs.org/ and sign in with your GitHub account.")
    print("2. Click on 'Import a Project' and select the memorial-tree repository.")
    print("3. Configure the project with the following settings:")
    print("   - Name: memorial-tree")
    print("   - Repository URL: https://github.com/yourusername/memorial-tree")
    print("   - Repository type: Git")
    print("   - Default branch: main")
    print("   - Language: English")
    print("4. Click 'Next' and then 'Build' to start the documentation build.")
    print("5. Once the build is complete, your documentation will be available at:")
    print("   https://memorial-tree.readthedocs.io/\n")

    print("6. To set up a webhook for automatic builds, go to your project's admin page:")
    print("   https://readthedocs.org/dashboard/memorial-tree/webhooks/")
    print("   and copy the webhook URL.")
    print("7. Add this URL to your GitHub repository's webhooks:")
    print("   https://github.com/yourusername/memorial-tree/settings/hooks/new")
    print("   with the following settings:")
    print("   - Payload URL: (the webhook URL from Read the Docs)")
    print("   - Content type: application/json")
    print("   - Secret: (leave empty)")
    print("   - Events: Just the push event")
    print("8. Click 'Add webhook' to save.\n")

    print("Would you like to open Read the Docs now? (y/n)")
    response = input().strip().lower()
    if response == 'y':
        webbrowser.open("https://readthedocs.org/")


def main():
    """Main function to check Read the Docs setup."""
    print("Checking Read the Docs configuration...\n")

    all_checks_passed = True
    all_checks_passed &= check_readthedocs_config()
    all_checks_passed &= check_sphinx_config()
    all_checks_passed &= check_github_workflow()

    if all_checks_passed:
        print("\nAll configuration files for Read the Docs are in place!")
    else:
        print("\nSome configuration files are missing. Please fix the issues above.")

    provide_readthedocs_instructions()


if __name__ == "__main__":
    main()