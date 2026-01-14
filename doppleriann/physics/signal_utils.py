#!/usr/bin/env python
"""
-------------------------------------------------------
DopplerIANN Physics — Signal Utilities
-------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
-------------------------------------------------------

Collection of signal-processing tools for analyzing radial
velocity (RV) time series, including long-term trend removal,
periodogram analysis, and phase recovery methods.

Functions include:
- Sinusoidal modeling
- Polynomial detrending
- Periodogram generation (GLS)
- Iterative removal of strongest periodic signals
- Phase offset recovery and diagnostic plots
-------------------------------------------------------
"""

import os
import numpy as np
from PyAstronomy.pyTiming import pyPeriod
import matplotlib.pyplot as plt
from scipy.stats import circmean
from ..utils.logger_config import logger


def sinusoidal_model(x, A, T, phi, C):
    """
    Sinusoidal model function: A * sin(2πx/T + φ) + C

    Parameters
    ----------
    x : array_like
        Input time or phase values.
    A : float
        Amplitude of the sine wave.
    T : float
        Period of the signal.
    phi : float
        Phase offset (radians).
    C : float
        Constant offset.

    Returns
    -------
    ndarray
        Evaluated sinusoidal function values.
    """
    return A * np.sin(2 * np.pi / T * x + phi) + C


def long_term_remover(times, time_series, degree=3):
    """
    Removes long-term trends in a time series via polynomial fitting.

    Parameters
    ----------
    times : array_like
        Observation times.
    time_series : array_like
        Measured values (e.g., RVs).
    degree : int, optional
        Degree of the polynomial trend to remove (default: 3).
        If 0, no detrending is applied.

    Returns
    -------
    ndarray
        Detrended time series (original minus fitted trend).
    """
    if degree == 0:
        return time_series

    # Fit polynomial trend
    poly_coeff = np.polyfit(times, time_series, degree)
    poly_trend = np.polyval(poly_coeff, times)

    # Subtract from original
    detrended_time_series = time_series - poly_trend
    return detrended_time_series


def remove_strongest_signals(rvs, time, err, n_remove):
    """
    Iteratively remove the strongest periodic signals from an RV time series
    using the Generalized Lomb-Scargle (GLS) periodogram.

    Parameters
    ----------
    rvs : array_like
        Radial velocity values.
    time : array_like
        Observation times (days).
    err : array_like
        Measurement uncertainties.
    n_remove : int
        Number of strongest signals to iteratively remove.

    Returns
    -------
    ndarray
        Residuals after signal removal.
    """
    residuals = rvs.copy()

    for i in range(n_remove):
        try:
            clp = pyPeriod.Gls((time, residuals, err), norm="ZK")
            power = clp.power
            freq = clp.freq
            best_freq = freq[np.argmax(power)]
            best_period = 1.0 / best_freq

            print(f"[{i+1}] GLS peak before removal: period = {best_period:.4f} days, power = {np.max(power):.4f}")

            # Fit best sinusoid and subtract
            model_fit = clp.sinmod(time)
            residuals -= model_fit

        except Exception as e:
            print(f"[{i+1}] GLS subtraction failed: {e}")
            break

    return residuals


def recover_phase_offset(dates, phase_values, period_days, reference_date=None):
    """
    Recover the relative phase offset between injected and observed signals.

    Parameters
    ----------
    dates : array_like
        Observation dates.
    phase_values : array_like
        Injected or recovered phase values (radians).
    period_days : float
        Signal period in days.
    reference_date : float, optional
        Reference date for phase zero. Defaults to the first date.

    Returns
    -------
    float
        Circular mean phase offset in fractional phase units [0, 1].
    """
    if reference_date is None:
        reference_date = dates[0]

    # Compute fractional phase from dates
    raw_phase_frac = ((dates - reference_date) / period_days) % 1

    # Convert injected phase to fractional phase
    phase_values_frac = (phase_values / (2 * np.pi)) % 1

    # Compute difference and take circular mean
    delta_phase_frac = (phase_values_frac - raw_phase_frac) % 1
    phase_offset_frac = circmean(delta_phase_frac, high=1.0, low=0.0)

    return phase_offset_frac

def circ_dist_cycles(a, b):
    d = (a - b) % 1.0
    return min(d, 1.0 - d)

def periodogram(rvs, time, err=None, fap=0.1, min_period=5, max_period=2000):
    """
    Compute a Generalized Lomb-Scargle (GLS) periodogram for radial velocity data.

    Parameters
    ----------
    rvs : array_like
        Radial velocity measurements.
    time : array_like
        Observation times (days).
    err : array_like, optional
        Measurement uncertainties. Defaults to None.
    fap : float, optional
        False alarm probability threshold (default: 0.1).
    min_period : float, optional
        Minimum period to search (days).
    max_period : float, optional
        Maximum period to search (days).

    Returns
    -------
    tuple
        (clp, plevels)
        - clp : pyPeriod.Gls object containing periodogram results.
        - plevels : float, false-alarm probability threshold.
    """
    # Convert to frequency bounds
    minFreq = 1.0 / max_period
    maxFreq = 1.0 / min_period

    # Compute GLS periodogram
    clp = pyPeriod.Gls((time, rvs, err), norm="ZK", fbeg=minFreq, fend=maxFreq)
    clp.info()

    plevels = clp.powerLevel(fap)
    return clp, plevels


def generate_periodogram_test(**kwargs):
    """
    Generate and optionally plot GLS periodograms for real vs predicted data.

    Parameters
    ----------
    **kwargs : dict
        real_rv : array_like
            Original radial velocities.
        pred_rv : array_like
            Predicted radial velocities.
        pred_ds : array_like
            Predicted Doppler shifts or derived signal.
        dates : array_like
            Observation dates.
        prefix_name : str, optional
            Prefix for output plot filenames.
        shell_type_str : str, optional
            Optional tag for labeling.
        spec_type : str, optional
            Spectrum type (default: 'solar').
        ds_size : float, optional
            Doppler shift amplitude for annotation.
        period : float, optional
            Injected period (days).
        min_period, max_period : float, optional
            Period range for GLS computation.
        fap : float, optional
            False alarm probability (default: 0.1).
        plot : bool, optional
            If True, plot periodograms.
        savefig : bool, optional
            If True, save plots to /img directory.

    Returns
    -------
    dict
        Dictionary containing:
        - fig : matplotlib Figure (if plot=True)
        - clp_rv_real : GLS for real RVs
        - clp_rv_pred : GLS for predicted RVs
        - clp_ds_pred : GLS for predicted DS
    """
    real_rv = kwargs.get('real_rv')
    pred_rv = kwargs.get('pred_rv')
    pred_ds = kwargs.get('pred_ds')
    dates = kwargs.get('dates')
    prefix_name = kwargs.get('prefix_name', 'periodogram_test')
    shell_type_str = kwargs.get('shell_type_str', None)
    spec_type = kwargs.get('spec_type', 'solar')
    ds_size = kwargs.get('ds_size', None)
    period = kwargs.get('period', None)
    min_period = kwargs.get('min_period', 5)
    max_period = kwargs.get('max_period', 800)
    fap = kwargs.get('fap', 0.1)
    plot = kwargs.get('plot', False)
    output_dir = kwargs.get('output_dir', 'img')
    savefig = kwargs.get('savefig', False)

    # Load the data to test with different period
    fapLevels = np.array([fap])

    # Compute GLS periodograms
    clp, plevels = periodogram(real_rv, dates, err=None, fap=fap, min_period=min_period, max_period=max_period)
    clppred, plevelspred = periodogram(pred_rv, dates, err=None, fap=fap, min_period=min_period, max_period=max_period)
    clpDSTest, pDSlevelsTest = periodogram(pred_ds, dates, err=None, fap=fap, min_period=min_period, max_period=max_period)

    fig = None
    ds_size = None if ds_size == 0.0 else ds_size
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        scale_factor = 1.5
        base_fontsize = 10

        # --- First subplot: RVs ---
        axs[0].plot(1. / clp.freq, clp.power, 'r.-', label='RV CCF', alpha=0.5, linewidth=4.5)
        axs[0].plot(1. / clppred.freq, clppred.power, 'b.-', label='RV neural prediction', alpha=0.5, linewidth=4.5)
        axs[0].plot([1/min(clp.freq), 1/max(clp.freq)], [plevels]*2, '--', label="FAP = %4.1f%%" % (fapLevels*100))
        if ds_size is not None and period is not None:
            axs[0].set_title(f'Injected signal: period {period} days | DS: {ds_size} m/s',
                             fontsize=scale_factor * base_fontsize)
        else:
            axs[0].set_title('Without injected signal',
                             fontsize=scale_factor * base_fontsize)
        axs[0].set_xlabel("Period (days)", fontsize=scale_factor * base_fontsize)
        axs[0].set_ylabel("Power", fontsize=scale_factor * base_fontsize)
        axs[0].set_xscale('log')
        axs[0].legend(prop={'size': scale_factor * base_fontsize})

        # --- Second subplot: DS predictions ---
        axs[1].plot(1. / clpDSTest.freq, clpDSTest.power, 'b.-', label='DS neural prediction')
        axs[1].plot([1 / min(clpDSTest.freq), 1 / max(clpDSTest.freq)], [pDSlevelsTest] * 2, '--',
                    label="FAP = %4.1f%%" % (fapLevels * 100))
        axs[1].set_xlabel("Period (days)", fontsize=scale_factor * base_fontsize)
        axs[1].set_ylabel("Power", fontsize=scale_factor * base_fontsize)
        axs[1].set_xscale('log')
        axs[1].legend(loc='center', prop={'size': scale_factor * base_fontsize})

        plt.tight_layout()

        if savefig:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{prefix_name}_{spec_type}_combined_periodogram.png")
            plt.savefig(out_path, dpi=300)
            print(f"[INFO] Periodogram saved to: {out_path}")

    return {'fig': fig, 'clp_rv_real': clp, 'clp_rv_pred': clppred, 'clp_ds_pred': clpDSTest}


def prior_transform(u):
    """
    Transform unit-cube parameters into physical model parameters for sinusoidal fitting.

    Parameters
    ----------
    u : array_like of shape (4,)
        Uniform random variables in [0, 1].

    Returns
    -------
    tuple
        (A, T, phi, C)
        - A : float, amplitude (0-50)
        - T : float, period (1-200)
        - phi : float, phase (-pi to pi)
        - C : float, constant offset (-10 to 10)
    """
    A = u[0] * 50
    T = 1 + u[1] * 200
    phi = -np.pi + u[2] * 2 * np.pi
    C = -10 + u[3] * 20
    return A, T, phi, C
