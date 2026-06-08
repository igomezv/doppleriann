Introduction
============

DopplerIANN is a framework for Doppler-shift inference with artificial neural networks.

It combines physical modeling, data preparation, shell generation, and neural-network-based inference for stellar spectroscopy analysis.

Main components
---------------

- ``doppleriann.physics`` for CCFs, shell-level injections, and signal analysis.
- ``doppleriann.data`` for loading, preprocessing, and scaling datasets.
- ``doppleriann.networks`` for CNNs, VAEs, MLPs, and KANs.
- ``doppleriann.utils`` for logging and shared helpers.
