Installation
============

.. toctree::

Installation with mamba
-----------------------

``cinnabar`` is available through *conda-forge* and can be installed with mamba (or conda):

.. code-block:: bash

    $ mamba install -c conda-forge cinnabar


Developer Installation
----------------------

If you're a developer, you will likely want to create a local editable installation.

1. clone the repository:

.. code-block:: bash

    $ git clone https://github.com/OpenFreeEnergy/cinnabar.git
    $ cd cinnabar


2. create and activate a new environment:

.. code-block:: bash

    $ mamba create -f devtools/conda-envs/env.yml
    $ mamba activate cinnabar

3. build an editable installation:

.. code-block:: bash

    $ python -m pip install --no-deps -e .