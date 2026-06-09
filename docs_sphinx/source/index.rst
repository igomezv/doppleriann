doppleriann
===========

``doppleriann`` is a Python package for modeling Doppler shifts in
high-resolution stellar spectra using physically motivated spectral-shell
representations and deep learning.

The package was developed to support radial-velocity exoplanet searches in the
presence of stellar variability. It provides tools to construct flux- and
temperature-based shell representations, compute cross-correlation functions
(CCFs), inject synthetic planetary Doppler signals, train convolutional neural
networks, and evaluate planetary recoverability through time-series and
periodogram analyses.

This documentation site is still under construction, so some sections may be
incomplete or updated over time.

As experimental features ``doppleriann`` also includes other neural networks architectures to 
train them with spectra or shells.

Overview
--------

``doppleriann`` is designed around a modular workflow:

1. Load and preprocess high-resolution spectra.
2. Select relevant spectral regions using masks.
3. Compute CCF-based radial velocities and activity indicators.
4. Inject controlled planetary Doppler shifts.
5. Build flux- or temperature-based spectral-shell representations.
6. Train neural-network models to predict radial velocity and Doppler shift.
7. Evaluate recovered signals using hold-out tests, cross-validation, and
   periodogram-based detection criteria.

The framework is particularly aimed at applications where stellar activity
dominates the radial-velocity signal and where low-amplitude planetary signals
must be recovered from real spectroscopic data.

Main features
-------------

* Flux-based and temperature-based spectral-shell representations.
* Weighted and masked shell construction.
* CCF calculation and radial-velocity extraction.
* Synthetic planetary signal injection.
* HARPS-N solar-data preprocessing utilities.
* CNN architectures for radial-velocity and Doppler-shift prediction.
* Hold-out and cross-validation training strategies.
* Monte Carlo dropout inference for predictive dispersion estimates.
* Scripts for hyperparameter optimization, model training, and evaluation.
* Reproducible experiment folders for the analyses presented in the paper.

Source code
-----------

The source code is available on GitHub:

* `GitHub repository <https://github.com/igomezv/doppleriann.git>`_
* `Download archive <https://github.com/igomezv/doppleriann/archive/refs/heads/master.zip>`_

To clone the repository:

.. code-block:: bash

   git clone https://github.com/igomezv/doppleriann.git
   cd doppleriann

For setup instructions, see :doc:`installation`.

Documentation contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   structure
   pipeline
   data_generators
   shell_data
   usage
   tutorials
   references
   api

Changelog
---------

Project changes and release notes will be listed here as the package evolves.

Citation
--------

If you use ``doppleriann`` in scientific work, please cite the associated paper:

   Gómez-Vargas et al., *Modeling Doppler Shifts in Radial-Velocity Data with
   Deep Learning toward Earth-mass Exoplanet Detection*.

License
-------

``doppleriann`` is distributed under the license included in the repository.
