Contributing
===========

We welcome contributions to Memorial Tree! This document provides guidelines and instructions for contributing.

Setting Up Development Environment
--------------------------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

       git clone https://github.com/yourusername/memorial-tree.git
       cd memorial-tree

3. Install development dependencies:

   .. code-block:: bash

       pip install -e ".[dev,test,docs]"

4. Create a branch for your changes:

   .. code-block:: bash

       git checkout -b feature/your-feature-name

Code Style
---------

We follow PEP 8 guidelines for Python code. Please ensure your code adheres to these standards.

- Use the Black formatter to automatically format your code:

  .. code-block:: bash

      black src tests examples

- Use type hints throughout the codebase
- Write docstrings in Google style format

Testing
------

All new code should include tests:

1. Write tests for your new feature or bug fix
2. Ensure all tests pass:

   .. code-block:: bash

       pytest

3. Check code coverage:

   .. code-block:: bash

       pytest --cov=memorial_tree

Documentation
------------

Update documentation for any changes:

1. Update docstrings for modified functions/classes
2. Update or add examples if needed
3. Build documentation locally to verify:

   .. code-block:: bash

       cd docs
       make html
       # View docs in browser at docs/build/html/index.html

Pull Request Process
------------------

1. Update the README.md or documentation with details of changes if appropriate
2. Update the version number in relevant files following semantic versioning
3. Submit a pull request to the main repository
4. Address any feedback from code reviews

Code of Conduct
-------------

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members