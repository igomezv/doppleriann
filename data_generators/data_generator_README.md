# Data Generator Scripts (DopplerIANN)

This guide documents the scripts in `data_generators/`, their inputs/outputs, and the recommended execution order.

## Recommended Order

1. `data_generators/load_harpsn_data.py`
2. `data_generators/temp_and_kitcat_gen.py`
3. `data_generators/ccf_data_generator.py` (optional)
4. `data_generators/test_shell_gen_fixed.py`

Run all commands from the repository root.

## 1) `load_harpsn_data.py`

Purpose:
- Load HARPS-N products and create baseline flux arrays (original and activity-corrected) plus observation-time metadata.

Main required inputs:
- HARPS-N workspace folder defined in the script (`data_dir`).
- `WORKSPACE/Analyse_summary.csv`
- `WORKSPACE/Analyse_material.p`
- `WORKSPACE/CONTINUUM/Continuum_matching_mad.npy`
- `CORRECTION_MAP/map_matching_activity.npy`
- Per-observation `RASSINE_Stacked_spectrum_*.p` files referenced in the catalog.

Outputs written to `data/`:
- `spectra_orig.npy`
- `spectra_active.npy`
- `time_df.csv`

Run:

```bash
python data_generators/load_harpsn_data.py
```

Notes:
- The script currently uses a hard-coded `data_dir`; update it for your machine before running.
- Downstream scripts use `data/wavelengths.txt` and a `date` column in `data/time_df.csv`.
  If your generated files do not include them, use the repository-provided versions or generate them before the next step.

## 2) `temp_and_kitcat_gen.py`

Purpose:
- Convert flux spectra to temperature-space spectra.
- Apply KITCAT mask filtering to produce reduced flux/temperature arrays used by shell generation.

Main required inputs:
- `data/spectra_orig.npy`
- `data/spectra_active.npy`
- `data/time_df.csv` (expects `date` column)
- `data/wavelengths.txt`
- `data/spectra_orig_err.npy`
- `data/T1o2_spec.csv`
- `data/mask_kitcat_NEW_kitcat_CCF_mask_Sun.npz`

Outputs written to `data/`:
- `temp_or.npy`, `temp_act.npy`
- `spectra_kitcat_or.npy`, `spectra_kitcat_act.npy`
- `temp_kitcat_or.npy`, `temp_kitcat_act.npy`
- `waves_kitcat.txt`

Run:

```bash
python data_generators/temp_and_kitcat_gen.py
```

Notes:
- The script has a `device_hpc` flag and defines `large_data_dir`; verify these paths for your setup.

## 3) `ccf_data_generator.py` (optional)

Purpose:
- Compute CCF-derived observables (`rv`, `rv_err`, `fwhm`, `fwhm_err`, `bis`) for original and activity spectra.

Main required inputs:
- `data/time_df.csv`
- `data/wavelengths.txt`
- `data/spectra_orig.npy`
- `data/spectra_active.npy`

Outputs written to `data/`:
- `astro_data_orig.csv`
- `astro_data_active.csv`

Run:

```bash
python data_generators/ccf_data_generator.py
```

Notes:
- Uses the CCF backend described in `README.md` (wrapper first, C++ fallback).

## 4) `test_shell_gen_fixed.py`

Purpose:
- Generate shell datasets with controlled planetary injections over a grid of periods and amplitudes.
- Save one shell directory per array index (`data/shells/<idx>/`).

Main required inputs:
- `data/time_df.csv` (expects `date` column)
- `data/waves_kitcat.txt`
- From `large_data_dir` (`$HOME/data` when `device_hpc=True`):
  - `spectra_kitcat_or_err.npy`
  - `temp_kitcat_or_err.npy`
  - `spectra_kitcat_act.npy`
  - `temp_kitcat_act.npy`

Outputs written to `data/shells/<idx>/`:
- `flux_PI*_P*_act.h5`
- `temp_PI*_P*_act.h5`
- `injection_phases.txt`

Run (single index locally):

```bash
python data_generators/test_shell_gen_fixed.py 0
```

Run with SLURM array (recommended for full sweep):

```bash
sbatch runShellGen.sh
```

Notes:
- The script skips combinations already present in the output directory.
- `runShellGen.sh` launches array jobs `0..9`.

## Troubleshooting

- Missing `date` in `data/time_df.csv`:
  downstream scripts call `pd.DatetimeIndex(time_df.date)`, so make sure `date` exists.
- Missing `data/wavelengths.txt`:
  required by temperature/CCF generation.
- Missing GSL or compiler:
  needed only if CCF wrapper compilation fails and C++ fallback is used.
