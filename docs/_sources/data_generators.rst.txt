Data Generators
===============

This guide documents the scripts in ``data_generators/``, their inputs and outputs, and the recommended execution order.

Recommended order
-----------------

1. ``data_generators/load_harpsn_data.py``
2. ``data_generators/temp_and_kitcat_gen.py``
3. ``data_generators/ccf_data_generator.py`` (optional)
4. ``data_generators/test_shell_gen_fixed.py``

Run all commands from the repository root.

``load_harpsn_data.py``
-----------------------

Purpose:

- Load HARPS-N products and create baseline flux arrays plus observation-time metadata.

Main required inputs:

- HARPS-N workspace folder defined in the script.
- ``WORKSPACE/Analyse_summary.csv``
- ``WORKSPACE/Analyse_material.p``
- ``WORKSPACE/CONTINUUM/Continuum_matching_mad.npy``
- ``CORRECTION_MAP/map_matching_activity.npy``
- Per-observation ``RASSINE_Stacked_spectrum_*.p`` files referenced in the catalog.

Outputs written to ``data/``:

- ``spectra_orig.npy``
- ``spectra_active.npy``
- ``time_df.csv``

``temp_and_kitcat_gen.py``
--------------------------

Purpose:

- Convert flux spectra to temperature-space spectra.
- Apply KITCAT mask filtering to produce reduced flux and temperature arrays used by shell generation.

Main required inputs:

- ``data/spectra_orig.npy``
- ``data/spectra_active.npy``
- ``data/time_df.csv``
- ``data/wavelengths.txt``
- ``data/spectra_orig_err.npy``
- ``data/T1o2_spec.csv``
- ``data/mask_kitcat_NEW_kitcat_CCF_mask_Sun.npz``

Outputs written to ``data/``:

- ``temp_or.npy``
- ``temp_act.npy``
- ``waves_kitcat.txt``

Outputs written to ``large_data/``:

- ``spectra_kitcat_or.npy``
- ``spectra_kitcat_act.npy``
- ``temp_kitcat_or.npy``
- ``temp_kitcat_act.npy``

``ccf_data_generator.py``
-------------------------

Purpose:

- Compute CCF-derived observables such as ``rv``, ``rv_err``, ``fwhm``, ``fwhm_err``, and ``bis``.

Main required inputs:

- ``data/time_df.csv``
- ``data/wavelengths.txt``
- ``data/spectra_orig.npy``
- ``data/spectra_active.npy``

Outputs written to ``data/``:

- ``astro_data_orig.csv``
- ``astro_data_active.csv``

``test_shell_gen_fixed.py``
---------------------------

Purpose:

- Generate shell datasets with controlled planetary injections over a grid of periods and amplitudes.
- Save one shell directory per array index in ``data/shells/<idx>/``.

Main required inputs:

- ``data/time_df.csv``
- ``data/waves_kitcat.txt``
- ``large_data/spectra_kitcat_or_err.npy``
- ``large_data/temp_kitcat_or_err.npy``
- ``large_data/spectra_kitcat_act.npy``
- ``large_data/temp_kitcat_act.npy``

Outputs written to ``data/shells/<idx>/``:

- ``flux_PI*_P*_act.h5``
- ``temp_PI*_P*_act.h5``
- ``injection_phases.txt``
