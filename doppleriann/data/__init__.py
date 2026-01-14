"""
DopplerIANN Data Module
----------------------------------------------------------
2025
by Isidro Gomez-Vargas
----------------------------------------------------------
This package provides tools for loading, preprocessing, and scaling
shell-based and astrophysical datasets used in DopplerIANN.
"""

# --- Data loading functions ---
from .data_loader import (
    load_shell_astro_datah5,
    load_shell_astro_datah5_random,
    load_and_process_file,
    load_hdf5_data,
)

# --- Scaler utilities ---
from .scalers import (
    MinMaxScaler3D,
    StandardScaler3D,
    MaskedStandardScaler3D,
)

from .shell_generation import generate_hidden_signals

__all__ = [
    # Data loading
    "load_shell_astro_datah5",
    "load_shell_astro_datah5_random",
    "load_and_process_file",
    "load_hdf5_data",

    # Scalers
    "MinMaxScaler3D",
    "StandardScaler3D",
    "MaskedStandardScaler3D",

    # Shell generation
    "generate_hidden_signals",
]
