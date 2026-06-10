Background
============

``doppleriann`` is a framework for Doppler-shift inference with neural networks.

It combines spectroscopy preprocessing, shell generation, CCF-based observables, planetary-signal injection, and model training and inference for
stellar-activity and exoplanet studies. The package is designed around the idea that physically motivated spectral representations can provide a compact and
informative description of high-resolution stellar spectra, making them suitable for deep-learning models applied to radial-velocity data.


Scientific motivation
---------------------

The radial-velocity method searches for the small Doppler shifts induced by orbiting planets. For Earth-mass planets around solar-type stars, these signals
can be comparable to, or smaller than, the variability produced by stellar activity. This makes it difficult to separate planetary signals from activity,
instrumental effects, and other sources of correlated variability.

``doppleriann`` provides tools to explore this problem using real stellar spectra, controlled synthetic Doppler-shift injections, and neural-network
models trained on reduced spectral representations. The framework is intended for experiments in which the user wants to test how well Doppler signals can be
recovered from spectroscopic data after transforming the spectra into more compact, physically motivated inputs.

With the library the results obtained in Gómez-Vargas, et al (2026) can be reproduced using 10-years of HARPS-N solar data. However, the library can be used with different data and also different DL approaches. 

Spectral-shell representations
------------------------------

A central component of ``doppleriann`` is the use of spectral shells. Instead of training directly on full high-resolution spectra, the spectra are projected
onto low-dimensional shell representations. These shells encode how spectral information is distributed as a function of quantities such as normalized flux,
line-formation temperature, and their velocity gradients. The repository includes tools for working with both flux-based and temperature-based shells. Flux shells provide a compact representation of the spectral information commonly used in Doppler measurements, while temperature-based shells are designed to include information related to the formation depth of spectral lines. Weighted or masked shell representations can also be used to emphasize spectral regions with higher Doppler information content and reduce the contribution of poorly informative or noisy regions. 

Deep-learning models
--------------------

``doppleriann`` includes neural-network tools for learning the relation between shell representations and Doppler-shift observables. In the main framework,
convolutional neural networks are trained to predict both a CCF-derived radial velocity and an injected Doppler-shift signal. The radial-velocity output
provides a reference connected to standard spectroscopic measurements, while the Doppler-shift output is used to study the recovery of controlled planetary
signals.

The repository also contains scripts associated with model training, hyperparameter optimization, and evaluation. These tools are intended to make it
easier to compare different shell representations, training configurations, and validation strategies.

Training, validation, and signal recovery
-----------------------------------------

The framework supports experiments based on hold-out testing and cross-validation. In the hold-out approach, a subset of spectra is kept unseen
during training and used only for evaluation. In the cross-validation approach, the dataset is split into folds so that different subsets are used as unseen
evaluation data in different training runs.

After inference, the predicted Doppler-shift time series can be analyzed with periodogram methods to test whether the injected planetary signal is recovered.
This makes the package useful for injection-recovery experiments, comparison of flux- and temperature-based representations, and tests of model robustness.

Hyperparameter optimization
---------------------------

The repository includes scripts and utilities associated with hyperparameter optimization. These components are useful for exploring neural-network
configurations such as learning rate, batch size, convolutional layers, dense layers, and related training parameters.

In the associated workflow, hyperparameter optimization is used to select stable model configurations before running the main hold-out or cross-validation
experiments. This helps make comparisons between different spectral representations more systematic and reduces the dependence on manually selected
architectures.

Uncertainty diagnostics of the ANN
------------------------------------------------

``doppleriann`` also includes tools for estimating prediction variability using Monte Carlo dropout. In this approach, dropout remains active during inference and multiple stochastic forward passes are performed for the same input. The mean of these predictions can be used as the final prediction, while the standard deviation across stochastic passes provides a measure of model
dispersion.

These Monte Carlo dropout values should be interpreted only as diagnostic uncertainty estimates. They are not calibrated predictive intervals and should
not be treated as complete statistical error bars on the recovered radial velocities or Doppler shifts. Their main purpose is to diagnose model behavior,
compare different configurations, identify unstable predictions, and assess the relative dispersion of predictions produced by different shell representations or training strategies.

Core modules
------------

- ``doppleriann.data`` for loading, preprocessing, and scaling datasets.
- ``doppleriann.physics`` for CCFs, shell-level injections, and signal analysis.
- ``doppleriann.networks`` for neural-network models, training utilities, and
  Monte Carlo dropout inference.
- ``doppleriann.utils`` for logging and shared helpers.

Next steps
----------

- Read `installation <installation.html>`_ to set up the environment.
- Review `pipeline <pipeline.html>`_ for the end-to-end workflow.
- See `usage <usage.html>`_ for a compact walkthrough.
