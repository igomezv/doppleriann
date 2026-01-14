"""
DopplerIANN Physics Module
----------------------------------------------------------
2025
by Isidro Gomez-Vargas
----------------------------------------------------------
This package contains physical modeling tools for DopplerIANN,
including CCF computation, shell-level manipulations,
spectral transformations, and time-series signal analysis.
"""

# --- Cross-Correlation Function (CCF) calculations ---
from .CCFcalculator import CCFcalculator, astro_data

# --- Shell-level physical utilities ---
from .shell_utils import (
    inject_planet_at_shell_level,
    remove_planet_signal_shell_level,
    extract_shape_shell,
)

# --- Spectral transformations and Doppler injections ---
from .spec_transform import SpectrumData, interp_temp_given_flux

# --- Signal analysis tools ---
from .signal_utils import (
    sinusoidal_model,
    long_term_remover,
    remove_strongest_signals,
    recover_phase_offset,
    periodogram,
    generate_periodogram_test,
    prior_transform,
    circ_dist_cycles,
)

__all__ = [
    # --- CCF calculations ---
    "CCFcalculator",
    "astro_data",

    # --- Shell utilities ---
    "inject_planet_at_shell_level",
    "remove_planet_signal_shell_level",
    "extract_shape_shell",

    # --- Spectral transforms ---
    "SpectrumData",
    "interp_temp_given_flux",

    # --- Signal analysis ---
    "sinusoidal_model",
    "long_term_remover",
    "remove_strongest_signals",
    "recover_phase_offset",
    "periodogram",
    "generate_periodogram_test",
    "prior_transform",
]
