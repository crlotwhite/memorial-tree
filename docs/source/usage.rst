Usage
=====

Basic Usage
----------

Creating a simple thought tree:

.. code-block:: python

    from memorial_tree import MemorialTree

    # Create a new thought tree
    tree = MemorialTree()

    # Add thoughts to the tree
    root_id = tree.add_thought(parent_id=None, content="Should I go for a walk?")
    yes_id = tree.add_thought(parent_id=root_id, content="Yes, I'll go for a walk", weight=0.7)
    no_id = tree.add_thought(parent_id=root_id, content="No, I'll stay home", weight=0.3)

    # Make a decision
    decision = tree.make_choice(root_id)
    print(f"Decision: {decision.content}")

    # Visualize the tree
    tree.visualize()

Working with Ghost Nodes
-----------------------

Ghost nodes represent unconscious influences on decision-making:

.. code-block:: python

    # Add a ghost node (unconscious influence)
    tree.add_ghost_node(content="Walking makes me anxious", influence=0.4)

    # Make a decision (now influenced by the ghost node)
    decision = tree.make_choice(root_id)
    print(f"Decision: {decision.content}")

Using Different Backends
----------------------

Memorial Tree supports multiple numerical computation backends:

.. code-block:: python

    # Using NumPy backend (default)
    tree = MemorialTree(backend='numpy')

    # Using PyTorch backend
    tree = MemorialTree(backend='pytorch')

    # Using TensorFlow backend
    tree = MemorialTree(backend='tensorflow')

    # Switching backends at runtime
    tree.switch_backend('pytorch')

Mental Health Models
------------------

Memorial Tree includes models for different mental health conditions:

.. code-block:: python

    from memorial_tree.models import ADHDModel, DepressionModel, AnxietyModel

    # Create a tree with ADHD model
    adhd_tree = MemorialTree(model=ADHDModel())

    # Create a tree with Depression model
    depression_tree = MemorialTree(model=DepressionModel())

    # Create a tree with Anxiety model
    anxiety_tree = MemorialTree(model=AnxietyModel())

    # Compare decision patterns
    adhd_decision = adhd_tree.make_choice(root_id)
    depression_decision = depression_tree.make_choice(root_id)
    anxiety_decision = anxiety_tree.make_choice(root_id)

Visualization
-----------

Memorial Tree provides tools for visualizing thought trees:

.. code-block:: python

    # Basic visualization
    tree.visualize()

    # Save visualization to file
    tree.visualize(save_path='my_tree.png')

    # Customize visualization
    tree.visualize(
        highlight_path=True,
        show_weights=True,
        show_ghost_influences=True,
        layout='spring'
    )

    # Analyze decision paths
    from memorial_tree.visualization import PathAnalyzer

    analyzer = PathAnalyzer(tree)
    analyzer.show_path_distribution()
    analyzer.show_ghost_influence_heatmap()