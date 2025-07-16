# Contributing to Memorial Tree

Thank you for your interest in contributing to Memorial Tree! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our code of conduct:

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/memorial-tree.git
   cd memorial-tree
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev,test,docs]"
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Code Style

We follow PEP 8 guidelines for Python code:

- Use the Black formatter to automatically format your code:
  ```bash
  black src tests examples
  ```
- Use type hints throughout the codebase
- Write docstrings in Google style format:
  ```python
  def function_name(param1: type, param2: type) -> return_type:
      """Short description of the function.

      Longer description if needed.

      Args:
          param1: Description of param1
          param2: Description of param2

      Returns:
          Description of return value

      Raises:
          ExceptionType: When and why this exception is raised
      """
  ```

### Testing

All new code should include tests:

1. Write tests for your new feature or bug fix in the `tests/` directory
2. Ensure all tests pass:
   ```bash
   pytest
   ```
3. Check code coverage:
   ```bash
   pytest --cov=memorial_tree
   ```

### CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment:

1. **Automated Tests**: Tests run automatically on every push and pull request
   - Tests run on multiple Python versions (3.8, 3.9, 3.10, 3.11)
   - Code coverage is reported to Codecov

2. **Code Quality Checks**: Linting and formatting checks run automatically
   - flake8 for code linting
   - black for code formatting
   - mypy for type checking

3. **Deployment**: New releases are automatically published to PyPI
   - Triggered when a new GitHub release is created
   - Can also be manually triggered with workflow_dispatch

To ensure your contribution passes CI checks, run these checks locally before submitting:

```bash
# Format code
black src tests examples

# Run linting
flake8 src tests examples

# Run type checking
mypy src tests examples

# Run tests with coverage
pytest --cov=memorial_tree
```

### Documentation

Update documentation for any changes:

1. Update docstrings for modified functions/classes
2. Update or add examples if needed
3. Build documentation locally to verify:
   ```bash
   cd docs
   make html
   # View docs in browser at docs/build/html/index.html
   ```

## Pull Request Process

1. Update the README.md or documentation with details of changes if appropriate
2. Update the version number in relevant files following semantic versioning
3. Make sure all tests pass and code coverage is maintained
4. Submit a pull request to the main repository
5. Address any feedback from code reviews

## Feature Requests and Bug Reports

- Use GitHub Issues to report bugs or request features
- For bug reports, include:
  - Description of the bug
  - Steps to reproduce
  - Expected behavior
  - Screenshots if applicable
  - Environment information (OS, Python version, package versions)
- For feature requests, include:
  - Clear description of the feature
  - Rationale for adding the feature
  - Example use cases

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning:

- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes

## License

By contributing to Memorial Tree, you agree that your contributions will be licensed under the project's MIT License.