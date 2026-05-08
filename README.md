# DopplerIANN 

This repository contains the reproducible code path used for the paper experiments based on:

- `experiments/cnnShell_HO/`
- `experiments/cnnShell_CV/` (CV5)

The pipeline represented here is:

`HARPS-N -> flux spectra -> temperature spectra -> planetary injections + CCF RV -> shell HDF5 -> HO/CV5 experiments`

## 1) Environment

```bash
conda create --name doppleriann python=3.11
conda activate doppleriann
python -m pip install --upgrade pip setuptools
pip install -e ".[all]"
```

## 2) CCF Backend (Optional C++ Requirement)

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

## 3) Canonical Pipeline

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
  - `spectra_kitcat_or.npy`, `spectra_kitcat_act.npy`
  - `temp_kitcat_or.npy`, `temp_kitcat_act.npy`
  - `waves_kitcat.txt`

### Step C. Shell generation with injections and CCF-derived RV

- Script: `data_generators/test_shell_gen_fixed.py`
- Launcher (SLURM array 0..9): `runShellGen.sh`
- Outputs in `data/shells/<idx>/` as HDF5 files:
  - `flux_PI*_P*_act.h5`
  - `temp_PI*_P*_act.h5`
  - `injection_phases.txt`

`generate_data` (in `doppleriann/data/shell_generation.py`) performs planetary injection, computes CCF-based observables, and writes shell datasets.

### Step D. HO/CV5 experiments

- Hold-out scripts:
  - `experiments/cnnShell_HO/cnnShellTemp.py`
  - `experiments/cnnShell_HO/cnnShellFlux.py`
  - `experiments/cnnShell_HO/cnnShellDetection.py`

- CV5 scripts:
  - Train: `experiments/cnnShell_CV/cv5fold_cnn.py`
  - Predict: `experiments/cnnShell_CV/cv_cnn_predict.py`
  - Detection: `experiments/cnnShell_CV/cv_cnn_detection.py`
  - Chunk merge: `experiments/cnnShell_CV/join_chunks.py`

Optional SLURM launchers:

- `runCVcnn.sh`
- `runCVpred.sh`
- `runCVDet.sh`
- `runCVchunk.sh`

## 4) Files Required by the Pipeline (Do Not Delete)

The following metadata files are required by the current scripts:

- `data/time_df.csv`
- `data/random_idx_train.npy`
- `data/random_idx_test.npy`
- `data/waves_kitcat.txt`
- `data/wavelengths.txt`
- `data/T1o2_spec.csv`
- `data/mask_kitcat_NEW_kitcat_CCF_mask_Sun.npz`

Model/reuse artifacts required for pretrained CV5 inference:

- `experiments/cnnShell_CV/models/models/*.h5`
- `experiments/cnnShell_CV/models/models/*.pkl`
- `experiments/cnnShell_CV/outputs/*_fold*_test_idx.txt`

## 5) Notes on Large Files

- Shell datasets are stored as `.h5` under `data/shells/`.
- Trained models are also `.h5` (plus `.pkl` scalers).
- Keep `.h5`/`.pkl` files needed for reproduction; figure outputs are intentionally excluded.

## 6) Tests

```bash
pytest -v
```
