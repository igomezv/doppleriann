# Doppler-shift Inference with Artificial Neural Networks (DopplerIANN)

---

## Overview

`doppleriann` is a Python package for modeling Doppler shifts in high-resolution stellar spectra using physically motivated spectral-shell representations and deep learning.

`doppleriann` provides:

- **Physical modeling** — CCF computation, shell-based spectral representations, planetary injection, and periodogram analysis.
- **Data handling** — processing, filtering, and masking of stellar spectra.
- **Neural architectures** — CNNs, VAEs, MLPs, and KANs.
- **Deep-learning features** — regularization and hyperparameter optimization.
- **Monte Carlo Dropout** — Monte Carlo dropout is implemented as a heuristic uncertainty-quantification method. It provides a measure of prediction variability across stochastic forward passes, not a calibrated predictive uncertainty.
- **Exploration utilities** — signal recovery, shell extraction, and uncertainty estimation.

## Repository Scope

This repository contains the `doppleriann` Python package and the reproducible code path used for the paper experiments. The core package provides utilities for spectral-shell generation, CCF-based observables, planetary injection, and neural-network experiments.

The full paper-reproduction pipeline requires HARPS-N-derived data products, shell datasets, and trained model artifacts. Some of these files are stored under `large_data/`, `data/shells/`, or experiment-specific model directories and may be managed separately through Git LFS or regenerated locally.

The paper experiments are based primarily on:

- `experiments/cnnShell_HO/` for hold-out (HO) experiments.
- `experiments/cnnShell_CV/` for 5-fold cross-validation (CV5) experiments.

The canonical pipeline represented in this repository is:

```text
HARPS-N -> flux spectra -> temperature spectra -> planetary injections + CCF RV -> shell HDF5 -> HO/CV5 experiments
```

For details on the Optuna-based hyperparameter optimization, including the genetic-algorithm search used for the CNN shell models, see:

```text
experiments/cnnShell_hyperparameter_opt/cnnShellOptuna.py
```

## Installation

Create and activate a dedicated Conda environment:

```bash
conda create --name doppleriann python=3.11
conda activate doppleriann
python -m pip install --upgrade pip setuptools
pip install -e ".[all]"
```

The `[all]` extra is intended to install the dependencies required for data generation, neural-network training, hyperparameter optimization, and analysis scripts. If you only need the core package, check whether a lighter installation with `pip install -e .` is sufficient for your use case.

### Quick verification

After installation, verify that the package imports correctly:

```bash
python -c "import doppleriann; print('doppleriann imported successfully')"
```

You can also run the CCF comparison script from the repository root:

```bash
python notebooks/ccf_calculator.py
```

## CCF Backend

CCF-derived observables are computed through:

```text
doppleriann/physics/CCFcalculator.py
```

By default, the package uses `wrapper=True`. On first use, it attempts to compile and load the Python C extension from:

```text
doppleriann/physics/ccf_resources/fit_CCF.c
```

A manual wrapper build can be performed with:

```bash
cd doppleriann/physics/ccf_resources
python setup_fit_CCF_PPP.py build_ext --inplace
```

If the Python C extension cannot be compiled or loaded, the code attempts to use the legacy C++ path (`wrapper=False`) based on:

```text
BIS_FIT2.cpp
```

This fallback requires a working `g++` compiler and GSL (`gsl-config`, version 2.6 or newer). Therefore, users who need CCF-derived observables should ensure that either the Python C extension builds successfully or the C++/GSL toolchain is installed.

In practice, a full C++/GSL setup is optional unless you explicitly run with `wrapper=False` or the Python wrapper compilation fails.

## Canonical Pipeline

Detailed instructions for the data-generation scripts are available in:

```text
data_generators/data_generator_README.md
```

The following pipeline describes the code path used for the paper experiments. It assumes that the required metadata files and HARPS-N-derived inputs are available under `data/` and `large_data/`.

### Step A. HARPS-N to flux arrays

- Script: `data_generators/load_harpsn_data.py`
- Main outputs written to `data/`:
  - `spectra_orig.npy`
  - `spectra_active.npy`
  - `time_df.csv`

### Step B. Flux to temperature spectra and KITCAT filtering

- Script: `data_generators/temp_and_kitcat_gen.py`
- Uses:
  - `data/T1o2_spec.csv`
  - `data/mask_kitcat_NEW_kitcat_CCF_mask_Sun.npz`

- Main outputs written to `data/`:
  - `temp_or.npy`
  - `temp_act.npy`
  - `waves_kitcat.txt`

- Main outputs written to `large_data/`:
  - `spectra_kitcat_or.npy`
  - `spectra_kitcat_act.npy`
  - `temp_kitcat_or.npy`
  - `temp_kitcat_act.npy`

If these files are already present in `large_data/`, you can skip Step B and proceed directly to Step C for shell generation.

### Step C. Shell generation with injections and CCF-derived RVs

- Script: `data_generators/test_shell_gen_fixed.py`

- Outputs written as HDF5 files under `data/shells/<idx>/`:
  - `flux_PI*_P*_act.h5`
  - `temp_PI*_P*_act.h5`
  - `injection_phases.txt`

- Uses `large_data/` for the filtered spectra and error arrays:
  - `spectra_kitcat_or_err.npy`
  - `temp_kitcat_or_err.npy`
  - `spectra_kitcat_act.npy`
  - `temp_kitcat_act.npy`

If the Step B `large_data/` outputs already exist, you can run Step C directly without regenerating them.

> **Note:** Step C expects error-array files such as `spectra_kitcat_or_err.npy` and `temp_kitcat_or_err.npy`. If these are not produced by Step B in your local workflow, make sure they are available through the expected data release, Git LFS artifacts, or a separate preprocessing step.

The `generate_data` function in:

```text
doppleriann/data/shell_generation.py
```

performs planetary injection, computes CCF-based observables, and writes the shell datasets.

### Step D. HO/CV5 experiments

The repository includes scripts for hold-out (HO) experiments and 5-fold cross-validation (CV5) experiments.

#### Hold-out scripts

- `experiments/cnnShell_HO/cnnShellTemp.py`  
  Loads pretrained models or trains from scratch for temperature shells.

- `experiments/cnnShell_HO/cnnShellFlux.py`  
  Loads pretrained models or trains from scratch for flux shells.

- `experiments/cnnShell_HO/cnnShellDetection.py`  
  Builds HO detection maps. This script is intended for runs with several shell realizations using different random phases.

#### CV5 scripts

- `experiments/cnnShell_CV/cv5fold_cnn.py`  
  Trains the CV5 CNN models.

- `experiments/cnnShell_CV/cv_cnn_predict.py`  
  Runs prediction with CV5 models. This script supports pretrained models and can be tested on a single shell realization.

For CV5 detection maps, several shell realizations are required:

- `experiments/cnnShell_CV/cv_cnn_detection.py`  
  Runs detection over period chunks that can be launched in parallel.

- `experiments/cnnShell_CV/join_chunks.py`  
  Merges chunk outputs into final CV matrices.

Detection-map results are not intended for a single shell realization.

### Step E. Analysis scripts

The `notebooks/` directory currently contains runnable Python analysis scripts rather than Jupyter notebooks:

- `notebooks/ccf_calculator.py`  
  Provides a quick comparison of CCF outputs using the wrapper and C++ paths on a mock spectrum.

- `notebooks/shells_plots_on_the_fly.py`  
  Generates shell representations on the fly for illustration. It loads the master spectra from `data/` and the full spectra/error arrays from `large_data/`, including:
  - `spectra_kitcat_act.npy`
  - `temp_kitcat_act.npy`
  - `spectra_kitcat_act_err.npy`
  - `temp_kitcat_or_err.npy`

Run these scripts from the repository root, for example:

```bash
python notebooks/ccf_calculator.py
```

or:

```bash
python notebooks/shells_plots_on_the_fly.py
```

## Required Data and Artifacts

The current pipeline expects both lightweight metadata files under `data/` and large spectral arrays under `large_data/`. Some files are generated by the scripts in `data_generators/`; others must already be available before running the relevant pipeline step.

### Metadata files required by the current scripts

- `data/time_df.csv`
- `data/wavelengths.txt`
- `data/T1o2_spec.csv`
- `data/mask_kitcat_NEW_kitcat_CCF_mask_Sun.npz`
- `data/waves_kitcat.txt`

### Files required to reproduce the paper HO results

To reproduce the paper hold-out results exactly, the original split files are required:

- `data/random_idx_train.npy`
- `data/random_idx_test.npy`

If these files are not available, you can build and save your own train/test split, but the numerical results may not exactly match the paper.

### Model and reuse artifacts required for pretrained CV5 inference

Pretrained CV5 inference requires:

- `experiments/cnnShell_CV/models/models/*.h5`
- `experiments/cnnShell_CV/models/models/*.pkl`
- `experiments/cnnShell_CV/outputs/*_fold*_test_idx.txt`

### Summary of key files for existing experiments

| Path or pattern | Required for | Produced by / source | Notes |
|---|---|---|---|
| `data/time_df.csv` | Step A onward | `load_harpsn_data.py` or provided data | Observation metadata |
| `data/wavelengths.txt` | Data processing | Provided data | Wavelength grid |
| `data/T1o2_spec.csv` | Step B | Provided data | Temperature-conversion input |
| `data/mask_kitcat_NEW_kitcat_CCF_mask_Sun.npz` | Step B | Provided data | KITCAT/CCF mask |
| `data/waves_kitcat.txt` | Step B onward | `temp_and_kitcat_gen.py` | KITCAT-filtered wavelength grid |
| `large_data/*.npy` | Steps B/C and analysis scripts | Step B, Git LFS, or provided data | Large spectral arrays |
| `data/shells/<idx>/*.h5` | HO/CV5 experiments | Step C | Shell datasets |
| `data/random_idx_train.npy` | Exact HO reproduction | Provided or locally generated | Original training split |
| `data/random_idx_test.npy` | Exact HO reproduction | Provided or locally generated | Original test split |
| `experiments/cnnShell_CV/models/models/*.h5` | Pretrained CV5 inference | Trained locally or provided artifact | Neural-network model weights |
| `experiments/cnnShell_CV/models/models/*.pkl` | Pretrained CV5 inference | Trained locally or provided artifact | Scalers or preprocessing artifacts |
| `experiments/cnnShell_CV/outputs/*_fold*_test_idx.txt` | Pretrained CV5 inference | CV5 training outputs or provided artifact | Fold-specific test indices |

## Notes on Large Files

- Shell datasets are stored as `.h5` files under `data/shells/`.
- Large `.npy` artifacts are stored under `large_data/` and may be managed with Git LFS.
- Trained models are stored as `.h5` files, usually with `.pkl` scalers or preprocessing artifacts.
- Keep the `.h5` and `.pkl` files required for reproduction.

If `large_data/` was cloned as Git LFS pointer files rather than actual `.npy` arrays, fetch the real artifacts with:

```bash
git lfs install
git lfs pull
```

Use `git lfs track "large_data/*.npy"` only when adding new large `.npy` files to the repository.

## Computational Notes

The HO and CV5 training scripts can be computationally intensive. GPU acceleration is recommended for training CNN models. Detection-map generation is designed to be parallelized over period chunks, especially for CV5 detection runs.

Exact paper reproduction requires the original train/test split files and pretrained CV5 artifacts. If these files are absent, the scripts can still be used to generate new splits and train new models, but the numerical results may differ from the paper.

## Citation

If you use `doppleriann` or the experiment pipeline in academic work, please cite the associated paper.

```bibtex
@article{doppleriann,
  title   = {Modeling Doppler Shifts in Radial-Velocity Data with Deep Learning toward Earth-mass Exoplanet Detection.},
  author  = {Gómez-Vargas, I., Dumusque, X., Zhao, Y., Al Moulla, K., \& Cretignier, M.},
  journal = {TBD},
  year    = {TBD}
}
```

