Pipeline
========

Overview
--------

DopplerIANN provides:

- Physical modeling for CCF computation, shell-based Doppler injection, and periodogram analysis.
- Data handling for 3D preprocessing and HDF5 dataset utilities.
- Neural architectures including CNNs, VAEs, MLPs, and KANs.
- Exploration utilities for signal recovery, shell extraction, and uncertainty estimation.

Canonical pipeline
------------------

The reproducible code path used for the paper experiments is:

``HARPS-N -> flux spectra -> temperature spectra -> planetary injections + CCF RV -> shell HDF5 -> HO/CV5 experiments``

Reference experiment folders:

- ``experiments/cnnShell_HO/``
- ``experiments/cnnShell_CV/``

Experiment scripts
------------------

Hold-out scripts:

- ``experiments/cnnShell_HO/cnnShellTemp.py``
- ``experiments/cnnShell_HO/cnnShellFlux.py``
- ``experiments/cnnShell_HO/cnnShellDetection.py``
- ``experiments/cnnShell_hyperparameter_opt/cnnShellOptuna.py``

CV5 scripts:

- ``experiments/cnnShell_CV/cv5fold_cnn.py``
- ``experiments/cnnShell_CV/cv_cnn_predict.py``
- ``experiments/cnnShell_CV/cv_cnn_detection.py``
- ``experiments/cnnShell_CV/join_chunks.py``

Notebooks and analysis scripts
------------------------------

``notebooks/`` currently contains runnable Python analysis scripts:

- ``notebooks/ccf_calculator.py``
- ``notebooks/shells_plots_on_the_fly.py``
