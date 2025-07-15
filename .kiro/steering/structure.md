# Project Structure & Organization

## Directory Structure

```
memorial_tree/
├── src/
│   └── memorial_tree/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── thought_node.py
│       │   ├── ghost_node.py
│       │   └── memorial_tree.py
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── backend_manager.py
│       │   ├── numpy_backend.py
│       │   ├── pytorch_backend.py
│       │   └── tensorflow_backend.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── adhd_model.py
│       │   ├── depression_model.py
│       │   └── anxiety_model.py
│       └── visualization/
│           ├── __init__.py
│           ├── tree_visualizer.py
│           └── path_analyzer.py
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_backends.py
│   ├── test_models.py
│   └── test_visualization.py
├── examples/
│   ├── basic_usage.py
│   ├── advanced_features.py
│   └── mental_health_models.py
├── docs/
│   ├── conf.py
│   ├── index.rst
│   └── api/
├── setup.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Module Organization

### Core Module

The `core` module contains the fundamental data structures and algorithms:
- `thought_node.py`: Base class for representing thoughts and decisions
- `ghost_node.py`: Extension of ThoughtNode for unconscious influences
- `memorial_tree.py`: Main tree structure for modeling thought processes

### Backends Module

The `backends` module provides abstraction for different numerical computation libraries:
- `backend_manager.py`: Factory and manager for different backends
- `numpy_backend.py`: NumPy implementation
- `pytorch_backend.py`: PyTorch implementation
- `tensorflow_backend.py`: TensorFlow/Keras implementation

### Models Module

The `models` module contains specialized models for different mental health conditions:
- `adhd_model.py`: Model for attention deficit and hyperactivity
- `depression_model.py`: Model for depressive thought patterns
- `anxiety_model.py`: Model for anxiety-related decision processes

### Visualization Module

The `visualization` module provides tools for analyzing and visualizing thought trees:
- `tree_visualizer.py`: Tools for rendering tree structures
- `path_analyzer.py`: Analysis of decision paths and influences

## Naming Conventions

- **Classes**: CamelCase (e.g., `ThoughtNode`, `MemorialTree`)
- **Functions/Methods**: snake_case (e.g., `add_thought`, `calculate_influence`)
- **Variables**: snake_case (e.g., `node_id`, `influence_weight`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_WEIGHT`, `MAX_DEPTH`)
- **Private members**: Prefix with underscore (e.g., `_initialize_backend`)

## Import Conventions

- Avoid wildcard imports (`from x import *`)
- Group imports in the following order:
  1. Standard library imports
  2. Third-party library imports
  3. Local application imports
- Use absolute imports within the package