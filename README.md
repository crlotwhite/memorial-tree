# Memorial Tree

[![PyPI version](https://badge.fury.io/py/memorial-tree.svg)](https://badge.fury.io/py/memorial-tree)
[![Tests](https://github.com/crlotwhite/memorial-tree/actions/workflows/test.yml/badge.svg)](https://github.com/crlotwhite/memorial-tree/actions/workflows/test.yml)
[![Lint](https://github.com/crlotwhite/memorial-tree/actions/workflows/lint.yml/badge.svg)](https://github.com/crlotwhite/memorial-tree/actions/workflows/lint.yml)
[![Documentation Status](https://readthedocs.io/en/latest/?badge=latest)](https://memorial-tree.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Read this in [한국어 (Korean)](README.ko.md)*

Memorial Tree is a Python package for modeling human thought processes and decision-making using tree data structures. The package aims to provide tools for computational psychiatry research by representing both conscious choices and unconscious influences (ghost nodes) in human cognitive processes.

## Features

- Tree-based modeling of human thought processes and decision-making
- Support for "ghost nodes" representing unconscious influences on decision-making
- Multiple backend support (NumPy, PyTorch, TensorFlow) for integration with existing ML workflows
- Specialized models for mental health conditions (ADHD, Depression, Anxiety)
- Visualization tools for analyzing thought patterns and decision paths

## Installation

### From PyPI (Coming Soon)

```bash
# Basic installation
pip install memorial-tree

# With PyTorch support
pip install memorial-tree[pytorch]

# With TensorFlow support
pip install memorial-tree[tensorflow]

# For development
pip install -e ".[dev,docs]"
```

### From GitHub Packages

You can install the package directly from GitHub Packages:

```bash
# Set up authentication (one-time setup)
export GITHUB_USERNAME=your-github-username
export GITHUB_TOKEN=your-personal-access-token

# Install the latest version
pip install --index-url https://github.com/crlotwhite/memorial-tree/raw/main/dist/ memorial-tree

# Or specify a version
pip install --index-url https://github.com/crlotwhite/memorial-tree/raw/main/dist/ memorial-tree==0.1.0
```

For more detailed instructions on using GitHub Packages, see [GitHub Packages Guide](docs/github_packages.md).

## Basic Usage

```python
from memorial_tree import MemorialTree

# Create a new thought tree
tree = MemorialTree()

# Add thoughts to the tree
root_id = tree.add_thought(parent_id=None, content="Should I go for a walk?")
yes_id = tree.add_thought(parent_id=root_id, content="Yes, I'll go for a walk", weight=0.7)
no_id = tree.add_thought(parent_id=root_id, content="No, I'll stay home", weight=0.3)

# Add a ghost node (unconscious influence)
tree.add_ghost_node(content="Walking makes me anxious", influence=0.4)

# Make a decision
decision = tree.make_choice(root_id)
print(f"Decision: {decision.content}")

# Visualize the tree
tree.visualize()
```

## Advanced Features

### Mental Health Models

Memorial Tree includes models for different mental health conditions:

```python
from memorial_tree.models import ADHDModel, DepressionModel, AnxietyModel

# Create a tree with ADHD model
adhd_tree = MemorialTree(model=ADHDModel())

# Create a tree with Depression model
depression_tree = MemorialTree(model=DepressionModel())

# Create a tree with Anxiety model
anxiety_tree = MemorialTree(model=AnxietyModel())
```

### Multiple Backends

Memorial Tree supports multiple numerical computation backends:

```python
# Using NumPy backend (default)
tree = MemorialTree(backend='numpy')

# Using PyTorch backend
tree = MemorialTree(backend='pytorch')

# Using TensorFlow backend
tree = MemorialTree(backend='tensorflow')
```

## Examples

Check out the [examples](examples/) directory for more detailed examples:

- [Basic Usage](examples/basic_usage.py)
- [Advanced Features](examples/advanced_features_example.py)
- [ADHD Model](examples/adhd_model_example.py)
- [Depression Model](examples/depression_model_example.py)
- [Anxiety Model](examples/anxiety_model_example.py)
- [Model Comparison](examples/model_comparison_example.py)
- [Tree Visualization](examples/tree_visualization_example.py)
- [Path Analysis](examples/path_analysis_example.py)

## Documentation

For full documentation, visit [memorialtree.readthedocs.io](https://memorialtree.readthedocs.io).

To build the documentation locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build the documentation
cd docs
make html
# View docs in browser at docs/build/html/index.html
```

The documentation is automatically built and deployed to Read the Docs when changes are pushed to the main branch or a new release is created.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details. If you have questions or suggestions, feel free to create an issue or contact me directly.

## Development

### CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment:

- **Tests**: Automatically runs tests on multiple Python versions for every push and pull request
- **Linting**: Checks code quality using flake8, black, and mypy
- **Publishing**: Automatically publishes new releases to PyPI when a new release is created
- **Documentation**: Automatically builds and deploys documentation to Read the Docs

To run the checks locally:

```bash
# Run tests
pytest --cov=memorial_tree tests/

# Check code formatting
black --check src tests examples

# Run linting
flake8 src tests examples

# Run type checking
mypy src tests examples
```

### Package Deployment

The package is automatically deployed to PyPI when a new release is created on GitHub. The deployment process includes:

1. Building the package
2. Testing the package on TestPyPI
3. Deploying to the official PyPI

You can also manually deploy the package using the provided script:

```bash
# Deploy to TestPyPI only
python scripts/deploy_to_pypi.py --test-only

# Deploy to TestPyPI and then to PyPI (after confirmation)
python scripts/deploy_to_pypi.py
```

To test the installation from PyPI or TestPyPI:

```bash
# Test installation from PyPI
python scripts/test_installation.py

# Test installation from TestPyPI
python scripts/test_installation.py --test-pypi
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Memorial Tree in your research, please cite:

```bibtex
@software{memorial_tree,
  author = {Noel Kim (crlotwhite)},
  title = {Memorial Tree: A Python Package for Modeling Human Thought Processes},
  year = {2025},
  url = {https://github.com/crlotwhite/memorial-tree},
  email = {crlotwhite@gmail.com}
}
```