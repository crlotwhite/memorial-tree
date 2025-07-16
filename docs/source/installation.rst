Installation
============

Memorial Tree can be installed from PyPI:

.. code-block:: bash

    # Basic installation
    pip install memorial-tree

    # With PyTorch support
    pip install memorial-tree[pytorch]

    # With TensorFlow support
    pip install memorial-tree[tensorflow]

    # For development
    pip install -e ".[dev,docs]"

Requirements
-----------

Memorial Tree requires Python 3.8 or higher and has the following dependencies:

* numpy
* matplotlib
* networkx

Optional dependencies:

* pytorch
* tensorflow
* sphinx (for documentation)
* pytest (for testing)

Development Installation
-----------------------

To install Memorial Tree for development:

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/yourusername/memorial-tree.git
       cd memorial-tree

2. Install in development mode:

   .. code-block:: bash

       pip install -e ".[dev,docs,test]"

3. Run tests to verify the installation:

   .. code-block:: bash

       pytest