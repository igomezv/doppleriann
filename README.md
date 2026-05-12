# Doppler-shift Inference with Artificial Neural Networks (DopplerIANN)

---

## Overview

**DopplerIANN** provides:

- **Physical modeling** — CCF computation, shell-based Doppler injection, and periodogram analysis.  
- **Data handling** — scalable 3D preprocessing and HDF5 dataset utilities.  
- **Neural architectures** — CNNs, VAEs, MLPs, and KANs.  
- **Exploration utilities** — signal recovery, shell extraction, and uncertainty estimation.


## Installation

```bash
conda create --name doppleriann python=3.11
conda activate doppleriann
python -m pip install --upgrade pip setuptools
pip install -e ".[all]"
```

## CCF Backend (Optional C++ Requirement)

- CCF-derived observables are computed through `doppleriann/physics/CCFcalculator.py`.
- Default mode is `wrapper=True`: on first use, the package tries to compile/load the Python C extension from `doppleriann/physics/ccf_resources/fit_CCF.c`.
- Manual wrapper build (optional):

```bash
cd doppleriann/physics/ccf_resources
python setup_fit_CCF_PPP.py build_ext --inplace
```

- If the wrapper cannot be compiled or loaded, the code falls back to C++ mode (`wrapper=False`) using `BIS_FIT2.cpp`.
- C++ mode requires `g++` and GSL (`gsl-config`, version 2.6 or newer).
- In practice, a full C++/GSL setup is optional unless you explicitly run with `wrapper=False` or wrapper compilation fails.


-----

This repository contains the reproducible code path used for the paper experiments based on:

- `experiments/cnnShell_HO/`
- `experiments/cnnShell_CV/` (CV5)

The pipeline represented here is:

`HARPS-N -> flux spectra -> temperature spectra -> planetary injections + CCF RV -> shell HDF5 -> HO/CV5 experiments`


## Canonical Pipeline

Detailed generator-script instructions are available in `data_generators/data_generator_README.md`.

### Step A. HARPS-N to flux arrays

- Script: `data_generators/load_harpsn_data.py`
- Main outputs written to `data/`:
  - `spectra_orig.npy`
  - `spectra_active.npy`
  - `time_df.csv`

### Step B. Flux to temperature + KITCAT filtering

- Script: `data_generators/temp_and_kitcat_gen.py`
- Uses `data/T1o2_spec.csv` and `data/mask_kitcat_NEW_kitcat_CCF_mask_Sun.npz`
- Main outputs written to `data/`:
  - `temp_or.npy`, `temp_act.npy`
  - `waves_kitcat.txt`
- Main outputs written to `large_data/`:
  - `spectra_kitcat_or.npy`, `spectra_kitcat_act.npy`
  - `temp_kitcat_or.npy`, `temp_kitcat_act.npy`

- If these files are already present in `large_data/`, you can skip Step B and proceed directly to Step C for shell generation.

### Step C. Shell generation with injections and CCF-derived RV

- Script: `data_generators/test_shell_gen_fixed.py`

- Outputs in `data/shells/<idx>/` as HDF5 files:
  - `flux_PI*_P*_act.h5`
  - `temp_PI*_P*_act.h5`
  - `injection_phases.txt`
- Uses `large_data/` for the filtered spectra and error arrays:
  - `spectra_kitcat_or_err.npy`
  - `temp_kitcat_or_err.npy`
  - `spectra_kitcat_act.npy`
  - `temp_kitcat_act.npy`

- If the Step B `large_data/` outputs already exist, you can run Step C directly without regenerating them.

`generate_data` (in `doppleriann/data/shell_generation.py`) performs planetary injection, computes CCF-based observables, and writes shell datasets.

### Step D. HO/CV5 experiments

- Hold-out scripts:
  - `experiments/cnnShell_HO/cnnShellTemp.py`: can load pretrained models or train from scratch for temperature shells.
  - `experiments/cnnShell_HO/cnnShellFlux.py`: can load pretrained models or train from scratch for flux shells.
  - `experiments/cnnShell_HO/cnnShellDetection.py`: builds HO detection maps and is intended for runs with several shell realizations (different random phases).

- CV5 scripts:
  - Train: `experiments/cnnShell_CV/cv5fold_cnn.py`.
  - Predict: `experiments/cnnShell_CV/cv_cnn_predict.py` (supports pretrained models and can be tested on a single shell realization).
  - For detection maps (several shell realizations are needed):
    - Detection: `experiments/cnnShell_CV/cv_cnn_detection.py` runs period chunks that can be launched in parallel.
    - Chunk merge: `experiments/cnnShell_CV/join_chunks.py` merges chunk outputs into final CV matrices.
    - Detection-map results are not intended for a single shell realization.

Optional SLURM launchers:

- `runCVcnn.sh`
- `runCVpred.sh`
- `runCVDet.sh`
- `runCVchunk.sh`

### Step E. Notebooks and analysis scripts

`notebooks/` currently contains runnable Python analysis scripts (not `.ipynb` files):

- `notebooks/ccf_calculator.py`: quick comparison of CCF outputs with wrapper/C++ paths on a mock spectrum.
- `notebooks/shells_plots_on_the_fly.py` generates shell representations on the fly for illustration by loading the master spectra from `data/` and the full spectra/error arrays from `large_data/` (`spectra_kitcat_act.npy`, `temp_kitcat_act.npy`, `spectra_kitcat_act_err.npy`, `temp_kitcat_or_err.npy`).

Run from repository root, for example:

```bash
python notebooks/ccf_calculator.py
```
or 

```bash
python notebooks/shells_plots_on_the_fly.py
```



## Files Required by the Pipeline 

The following metadata files are required by the current scripts:

- `data/time_df.csv`
- `data/wavelengths.txt`
- `data/T1o2_spec.csv`
- `data/mask_kitcat_NEW_kitcat_CCF_mask_Sun.npz`
- `data/waves_kitcat.txt`

To reproduce paper HO results (otherwise, you can build and save your own split):

- `data/random_idx_train.npy`
- `data/random_idx_test.npy`

Model/reuse artifacts required for pretrained CV5 inference:

- `experiments/cnnShell_CV/models/models/*.h5`
- `experiments/cnnShell_CV/models/models/*.pkl`
- `experiments/cnnShell_CV/outputs/*_fold*_test_idx.txt`

## Notes on Large Files

- Shell datasets are stored as `.h5` under `data/shells/`.
- Large `.npy` artifacts are stored under `large_data/` and managed with Git LFS.
- If `large_data/` was cloned as Git LFS pointer files, run the following to fetch the actual `.npy` artifacts:

  ```bash
  git lfs track "large_data/*.npy"
  git lfs pull
  ```
- Trained models are also `.h5` (plus `.pkl` scalers).
- Keep `.h5`/`.pkl` files needed for reproduction.
