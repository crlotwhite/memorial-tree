# Technical Stack & Development Guidelines

## Tech Stack

- **Language**: Python 3.8+
- **Core Dependencies**:
  - NumPy: For numerical computations and array operations
  - PyTorch: For tensor operations and gradient-based optimization
  - TensorFlow/Keras: For neural network integration
  - Matplotlib/NetworkX: For visualization of tree structures
  - Pytest: For testing framework

## Development Environment

- **Package Structure**: Standard Python package structure with `src/memorial_tree` as the main module
- **Version Control**: Git with feature branch workflow
- **Documentation**: Sphinx or MkDocs with auto-generated API docs
- **Type Hints**: Use Python type annotations throughout the codebase
- **Code Style**: Follow PEP 8 guidelines with Black formatter

## Build & Test Commands

### Installation

```bash
# Development installation
pip install -e .

# Install with all dependencies
pip install -e ".[dev,test,docs]"
```

### Testing

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=memorial_tree

# Run specific test category
pytest tests/test_core.py
```

### Documentation

```bash
# Generate documentation
cd docs
make html

# Serve documentation locally
python -m http.server -d docs/build/html
```

### Package Building & Distribution

```bash
# Build package
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

## CI/CD Pipeline

- GitHub Actions for automated testing on multiple Python versions
- Code quality checks with flake8, black, and mypy
- Automated documentation builds
- PyPI deployment on release tags