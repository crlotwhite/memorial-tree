#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="memorial-tree",
    version="0.1.0",
    description="A Python package for modeling human thought processes and decision-making using tree data structures",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Memorial Tree Team",
    author_email="info@memorialtree.org",
    url="https://github.com/memorialtree/memorial-tree",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
    ],
    extras_require={
        "pytorch": ["torch>=1.9.0"],
        "tensorflow": ["tensorflow>=2.6.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "mypy>=0.812",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.2",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="cognitive-modeling, decision-making, computational-psychiatry, tree-structure",
    license="MIT",
)
