Installation
============

Requirements
------------

- Python 3.9 or newer.
- Project dependencies installed from ``pyproject.toml``.

Install the package
-------------------

.. code-block:: bash

   pip install -e .

Install docs dependencies
-------------------------

.. code-block:: bash

   pip install -e .[docs]

Optional CCF backend
--------------------

The CCF backend can use a Python extension first and fall back to a C++ implementation if needed.

If you need the optional wrapper build:

.. code-block:: bash

   cd doppleriann/physics/ccf_resources
   python setup_fit_CCF_PPP.py build_ext --inplace
