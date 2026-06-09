Project Structure
=================

This repository is organized around a small set of top-level directories:

- ``doppleriann/`` contains the library code.
- ``data_generators/`` contains scripts that build intermediate datasets and shell products.
- ``experiments/`` contains training, evaluation, and inference scripts.
- ``notebooks/`` contains analysis scripts and plotting utilities.
- ``data/`` stores smaller generated artifacts used by the workflow.
- ``large_data/`` stores larger intermediate arrays and HDF5 products.


.. figure:: /img/structure.png

Supporting files at the repository root include ``README.md``, ``pyproject.toml``, and ``LICENSE``.

The docs are built from ``docs_sphinx/source/``.
