"""
-------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
-------------------------------------------------------

Collection of tools to manipulate shell representations of stellar spectra.
Includes methods for planetary signal injection, removal, and extraction of
Doppler-orthogonal components (shape shells).
"""

import numpy as np
import pandas as pd
from ..utils.logger_config import logger


def inject_planet_at_shell_level(shell_stats, dates, doppler_shift_amplitude, period_days, 
                                 reference_date, c, randomize_phase=True):
    """
    Injects a planetary signal at the shell level, considering the orbital period and observation dates.

    Parameters
    ----------
    shell_stats : dict
        Dictionary containing shell data and gradients. Must include:
        - 'shell_df': 2D DataFrame of the base shell.
        - 'mean_grad_map': 2D array of mean gradients per bin.
        - 'mean_wave_map': 2D array of mean wavelengths per bin.
    dates : ndarray
        Observation times (Julian dates) corresponding to each shell.
    doppler_shift_amplitude : float
        Semi-amplitude of the injected Doppler shift in m/s.
    period_days : float
        Orbital period of the planetary signal in days.
    reference_date : float
        Reference Julian date for phase calculation.
    c : float
        Speed of light in m/s (typically 299792458).
    randomize_phase : bool, optional
        Whether to introduce a random phase offset (default is True).

    Returns
    -------
    injected_shells : list of pd.DataFrame
        List of shells with injected planetary signals, one per observation date.
    phase_values : ndarray
        Phase values used for each injection, in radians.
    """

    mean_grad = shell_stats['mean_grad_map']
    mean_wave = shell_stats['mean_wave_map']
    base_shell_df = shell_stats['shell_df']

    # Calculate phase for each observation
    phase_values = 2 * np.pi * (dates - reference_date) / period_days

    if randomize_phase:
        phase_offset = np.random.uniform(0, 2 * np.pi)
        phase_values = (phase_values + phase_offset) % (2 * np.pi)

    # Calculate Doppler shifts for each observation date
    doppler_shifts = doppler_shift_amplitude * np.sin(phase_values)

    # Inject planetary signal for each date
    injected_shells = []
    for shift in doppler_shifts:
        delta_shell_injection = (mean_grad * mean_wave / c) * shift
        injected_shell = base_shell_df + delta_shell_injection
        injected_shells.append(injected_shell.copy())

    return injected_shells, phase_values


def remove_planet_signal_shell_level(shell_stats, dates, doppler_shift_amplitude, 
                                     period_days, reference_date, c):
    """
    Removes a previously injected planetary signal from shell-level data.

    Parameters
    ----------
    shell_stats : dict
        Dictionary containing shell-level quantities:
        - 'shell_df': 2D DataFrame of the injected shell.
        - 'mean_grad_map': 2D array of mean gradients per bin.
        - 'mean_wave_map': 2D array of mean wavelengths per bin.
    dates : ndarray
        Observation times (Julian dates) corresponding to each shell.
    doppler_shift_amplitude : float
        Semi-amplitude of the injected Doppler shift in m/s.
    period_days : float
        Orbital period of the planetary signal in days.
    reference_date : float
        Reference Julian date used in the injection.
    c : float
        Speed of light in m/s (typically 299792458).

    Returns
    -------
    cleaned_shells : list of pd.DataFrame
        List of shells with the injected planetary signal removed, one per observation date.
    """
    mean_grad_map = shell_stats['mean_grad_map']
    mean_wave_map = shell_stats['mean_wave_map']
    injected_shell_df = shell_stats['shell_df']

    cleaned_shells = []

    # Precompute constant per-bin correction factor
    shell_correction_factor = mean_grad_map * mean_wave_map / c

    # Compute modulation across all observation dates
    phase_values = 2 * np.pi * (dates - reference_date) / period_days
    modulated_shifts = doppler_shift_amplitude * np.sin(phase_values)

    for shift in modulated_shifts:
        correction = shell_correction_factor * shift
        cleaned_shell = injected_shell_df - correction
        cleaned_shells.append(cleaned_shell.copy())

    return cleaned_shells


def extract_shape_shell(shell_obs, shell_dop):
    """
    Extracts the "shape shell" component by projecting an observed shell onto a
    Doppler shell and subtracting the Doppler-aligned component.

    This isolates the residual (activity-related) part of the shell representation,
    orthogonal to the Doppler-induced deformation.

    Parameters
    ----------
    shell_obs : ndarray or pd.DataFrame
        Shell data for the observed spectrum.
    shell_dop : ndarray or pd.DataFrame
        Shell data for the Doppler-shifted (reference) spectrum.

    Returns
    -------
    shell_shape_df : pd.DataFrame
        2D shell representation of the shape shell, representing Doppler-orthogonal features.
    """

    shell_obs = shell_obs
    shell_dop = shell_dop

    v_obs = shell_obs.flatten()
    v_dop = shell_dop.flatten()

    # projection scalar: <obs, dop> / <dop, dop>
    dot_dop_dop = np.dot(v_dop, v_dop)
    if dot_dop_dop == 0:
        raise ValueError("doppler shell norm is zero, cannot project.")

    projection_scalar = np.dot(v_obs, v_dop) / dot_dop_dop

    v_proj = projection_scalar * v_dop
    v_shape = v_obs - v_proj

    shell_shape_df = pd.DataFrame(v_shape)

    return shell_shape_df

