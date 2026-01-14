# Doppler-shift Inference with Artificial Neural Networks (DopplerIANN)

> A modular framework for stellar spectroscopy analysis and deep learning using physical-based shell representation.

---

## Overview

**DopplerIANN** provides a complete pipeline for astrophysical signal modeling, including:

- **Physical modeling** — CCF computation, shell-based Doppler injection, and periodogram analysis.  
- **Data handling** — scalable 3D preprocessing and HDF5 dataset utilities.  
- **Neural architectures** — CNNs, VAEs, MLPs, and KANs.  
- **Exploration utilities** — signal recovery, shell extraction, and uncertainty estimation.

Developed by  
**Isidro Gómez-Vargas (2025)**  
_University of Geneva_

---

## Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/DoppplerIANN.git
cd DopplerIANN
```

## Setting Up the Environment

1. **Create a Conda Environment**

   It is recommended to use `micromamba` or `conda`.  
   Run the following command to create a new environment:

   ```
   conda create --name doppleriann python=3.11
   ```

2. **Activate the Environment**

   ```
   conda activate doppleriann
   ```

3. **Install the Module in Editable Mode**

   Navigate to the root directory of DopplerIANN and run:

   ```
   python -m pip install --upgrade pip setuptools
   pip install -e ".[all]"
   ```

   This installs DopplerIANN in editable mode, allowing code changes to be reflected immediately.


## Data

Data files can be downloaded or generated locally, depending on your setup.  
Make sure paths inside your scripts point to the correct directories (for example: `data/`, `models/`, or `outputs/`).

---

## C++ Radial Velocity Calculation

DopplerIANN supports calculating radial velocities using a C++ implementation of the CCF.

The CCF computation is automatically handled by the `CCFcalculator` class in Python.  
The package dynamically selects between:
- **Python wrapper mode** (`wrapper=True`) using the precompiled shared library `fit_CCF.so`.
- **C++ executable mode** (`wrapper=False`) using the internal binary `BIS_FIT2`.

### Option 1 — Use the Python Wrapper (by default)

If a standard C compiler (e.g., GCC) is available, DopplerIANN will automatically attempt to build the **Python C wrapper** (`fit_CCF.c`).

You can also build it manually:

```bash
cd doppleriann/physics/ccf_resources
python setup_fit_CCF_PPP.py build
```

This will produce `fit_CCF.cpython-xxx-x86_64-linux-gnu.so` in the same directory, which the module will load automatically.

### Option 2 — Fallback to C++ Mode

If the Python wrapper cannot be compiled or loaded, **DopplerIANN automatically falls back to the native C++ implementation**. In this case, you must have the **GNU Scientific Library (GSL, version 2.6 or later)** installed on your system. Installation instructions are available here: [https://www.linuxfromscratch.org/blfs/view/cvs/general/gsl.html](https://www.linuxfromscratch.org/blfs/view/cvs/general/gsl.html).

When `wrapper=False`, the code will use:
- `BIS_FIT2.cpp` → compiled automatically into `BIS_FIT2`
- `G2_mask.txt` → built-in line mask file
- Internal handling for binary output (`ccf_parameter.bin`, etc.)

No manual setup is required — the C++ binary will compile and run on first use.  
If compilation fails, the system logs a warning suggesting switching to the C++ fallback mode.

---

## Project Structure

```
doppleriann/        # Core Python package
├── data/           # Data handling utilities
├── physics/        # Physical modeling and signal processing
│   └── ccf_resources/   # C++ source, mask, and compiled libraries
├── networks/       # Neural network architectures
├── utils/          # Logging and helpers

└── data_generators/  # Generations of shell data given stellar spectra
└── explorations/     # Research experiments and analysis scripts
└── notebooks/        # Short notebooks scripts, case of use
└── tests/            # Lightweight functional tests

```

## Quick Start

Once DopplerIANN is installed, you can quickly test a model build and training run:

```python
import numpy as np
from doppleriann.networks import ShellCNN1D
from doppleriann.data import MaskedStandardScaler3D, load_shell_astro_datah5

# Load and scale data
X, y = np.random.rand(100, 32, 64), np.random.rand(100, 2)
scaler = MaskedStandardScaler3D().fit(X)
X_scaled = scaler.transform(X)

# Build and train a CNN
model_arch = ShellCNN1D(input_shape=(32, 64), n_outputs=2)
model = model_arch.model_tf()
model.compile(optimizer="adam", loss="mse")
model.fit(X_scaled, y, epochs=5, batch_size=8)
```

---

## Testing

To verify that DopplerIANN installs and builds correctly:

```bash
pytest -v
```

Run a single test (e.g., network build):

```bash
pytest tests/test_networks_build.py -v
```

---

## Citation

If you use DopplerIANN in your research, please cite:

I. Gómez-Vargas, X. Dumusque (2025). *DopplerIANN: Doppler-shift Inference with Artificial Neural Networks*
