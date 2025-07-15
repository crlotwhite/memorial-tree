# Memorial Tree

Memorial Tree is a Python package for modeling human thought processes and decision-making using tree data structures. The package aims to provide tools for computational psychiatry research by representing both conscious choices and unconscious influences (ghost nodes) in human cognitive processes.

## Features

- Tree-based modeling of human thought processes and decision-making
- Support for "ghost nodes" representing unconscious influences on decision-making
- Multiple backend support (NumPy, PyTorch, TensorFlow) for integration with existing ML workflows
- Specialized models for mental health conditions (ADHD, Depression, Anxiety)
- Visualization tools for analyzing thought patterns and decision paths

## Installation

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

## Documentation

For full documentation, visit [memorialtree.readthedocs.io](https://memorialtree.readthedocs.io).

## License

This project is licensed under the MIT License - see the LICENSE file for details.