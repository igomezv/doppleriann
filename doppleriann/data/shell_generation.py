import os
import numpy as np
import h5py
from doppleriann.physics import SpectrumData, astro_data
from doppleriann.utils.logger_config import logger

def generate_hidden_signals(
    waves_obs,
    dates,
    spec_vals_flux=None,
    spec_vals_temp=None,
    spec_type='act',
    ds_amplitude=0.1,
    period_range=(5, 100),
    save_to=None,
):
    """
    Injects a hidden planetary signal into provided flux and/or temperature spectra time series.

    Parameters
    ----------
    waves_obs : np.ndarray
        Wavelength array used to build SpectrumData.
    dates : np.ndarray
        Observation Julian dates.
    spec_vals_flux, spec_vals_temp : np.ndarray, optional
        Flux and temperature spectra arrays (2D). At least one must be provided.
    spec_type : str
        Spectrum type tag (e.g., 'act').
    ds_amplitude : float, optional
        Doppler semi-amplitude (m/s).
    period_range : tuple, optional
        Range of orbital periods to choose randomly from.
    save_to : str or None, optional
        Directory path to save results. If None, results are only returned.

    Returns
    -------
    dict
        {
          "inj_flux": ndarray or None,
          "inj_temp": ndarray or None,
          "flux_master": ndarray or None,
          "temp_master": ndarray or None,
          "params": {"phase_offset": float, "ds": float, "period": float}
        }
    """
    if spec_vals_flux is None and spec_vals_temp is None:
        raise ValueError("At least one of spec_vals_flux or spec_vals_temp must be provided.")

    spec = SpectrumData(wavelengths=waves_obs)

    # Random orbital parameters
    phase_offset = np.random.uniform(0, 2 * np.pi)
    period_days = np.random.uniform(*period_range)

    params = dict(phase_offset=phase_offset, ds=ds_amplitude, period=period_days)
    logger.info("Hidden injection params: %s", params)

    results = {"inj_flux": None, "inj_temp": None, "flux_master": None, "temp_master": None, "params": params}

    # ---- Flux injection ----
    if spec_vals_flux is not None:
        inj_flux, _, _ = spec.planet_inj(
            spec_vals_flux, dates=dates,
            doppler_shift_amplitude=ds_amplitude,
            period_days=period_days,
            phase_offset=phase_offset
        )
        flux_master = np.mean(spec_vals_flux, axis=0)
        results.update({"inj_flux": inj_flux, "flux_master": flux_master})
        logger.info("Hidden signal injected into flux spectra.")

    # ---- Temperature injection ----
    if spec_vals_temp is not None:
        inj_temp, _, _ = spec.planet_inj(
            spec_vals_temp, dates=dates,
            doppler_shift_amplitude=ds_amplitude,
            period_days=period_days,
            phase_offset=phase_offset
        )
        temp_master = np.mean(spec_vals_temp, axis=0)
        results.update({"inj_temp": inj_temp, "temp_master": temp_master})
        logger.info("Hidden signal injected into temperature spectra.")

    # ---- Optional saving ----
    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        np.save(os.path.join(save_to, f"flux_hidden_{spec_type}.npy"), results["inj_flux"])
        np.save(os.path.join(save_to, f"temp_hidden_{spec_type}.npy"), results["inj_temp"])
        np.save(os.path.join(save_to, f"flux_master_hidden_{spec_type}.npy"), results["flux_master"])
        np.save(os.path.join(save_to, f"temp_master_hidden_{spec_type}.npy"), results["temp_master"])
        with open(os.path.join(save_to, f"hidden_injection_params_{spec_type}.txt"), "w") as f:
            for k, v in params.items():
                f.write(f"{k} = {v}\n")
        logger.info("Saved hidden-injected spectra and parameters to %s", save_to)
    
    return results


def generate_data(
    num_it,
    spec_flux,
    spec_temp,
    spec_flux_master,
    spec_temp_master,
    flux_err_val,
    temp_err_val,
    waves_obs,
    dates,
    data_dir,
    doppler_shift_range=(0.1, 2.0),
    n_reso=9,
    spec_type='act',
    random_shifts=True,
    period=None,
    doppler_shift=None,
    phase_offset=None,
):
    """
    Generate synthetic datasets (flux + temperature) with random or controlled
    Doppler injections and their corresponding shell diagrams.

    Parameters
    ----------
    num_it : int
        Iteration index (for file naming).
    spec_flux, spec_temp : np.ndarray
        Input spectra arrays (flux and temperature).
    spec_flux_master, spec_temp_master : np.ndarray
        Corresponding master (reference) spectra.
    flux_err_val, temp_err_val : np.ndarray
        Per-pixel uncertainties for flux and temperature.
    waves_obs : np.ndarray
        Observed wavelengths array.
    dates : np.ndarray
        Julian dates for each observation.
    data_dir : str
        Directory where generated .h5 files will be stored.
    doppler_shift_range : tuple(float, float)
        Min and max Doppler shift amplitudes [m/s].
    n_reso : int
        Shell resolution.
    spec_type : str
        Spectrum type identifier (e.g., 'act').
    random_shifts : bool, default=True
        If True, apply random Doppler shifts per spectrum (randomized amplitudes and phases).
        If False, inject a single coherent sinusoidal planetary signal.
    period : float, optional
        Orbital period in days (required if random_shifts=False).
    doppler_shift : float, optional
        Semi-amplitude [m/s] (required if random_shifts=False).
    phase_offset : float, optional
        Initial orbital phase (optional, random if not provided).
    """

    # Sanity check for controlled injections
    if not random_shifts:
        if period is None or doppler_shift is None:
            raise ValueError("When random_shifts=False, 'period' and 'doppler_shift' must be provided.")
        if phase_offset is None:
            phase_offset = np.random.uniform(0, 2 * np.pi)

        logger.info(f"[Controlled injection] period={period:.2f} days, K={doppler_shift:.3f} m/s, phase={phase_offset:.2f}")

    # Iterate over both spectrum types
    for is_temp, spec_vals, spec_master, spec_err in [
        (True, spec_temp, spec_temp_master, temp_err_val),
        (False, spec_flux, spec_flux_master, flux_err_val),
    ]:
        label = "temperature" if is_temp else "flux"

        # Build output name
        if random_shifts:
            root_name = f"{'temp' if is_temp else 'flux'}_PI_random_{spec_type}"
            file_name = os.path.join(data_dir, f"{root_name}_{num_it}.h5")
        else:
            root_name = f"{'temp' if is_temp else 'flux'}_PI{doppler_shift}_P{period}_{spec_type}"
            file_name = os.path.join(data_dir, f"{root_name}.h5")

        # Skip existing
        if os.path.exists(file_name):
            logger.info(f"Skipping existing file: {file_name}")
            continue

        logger.info(f"[{num_it}] Generating {label} data → {file_name}")

        spec = SpectrumData(wavelengths=waves_obs)

        # --- 1. Doppler injection ---
        if random_shifts:
            inj_spec, ds, phases = spec.inject_random_doppler_shifts(
                spec_vals, doppler_shift_range=doppler_shift_range
            )
        else:
            inj_spec, ds, phases = spec.planet_inj(
                spec_vals,
                dates=dates,
                doppler_shift_amplitude=doppler_shift,
                period_days=period,
                phase_offset=phase_offset,
            )

        # --- 2. Compute astrodata ---
        astrodata = astro_data(inj_spec, wave_cols=waves_obs, wrapper=True)
        astrodata = np.column_stack((astrodata, ds, phases))
        # astro_columns = ["rv", "rv_err", "fwhm", "fwhm_err", "bis", "ds", "phase"]

        # --- 3. Compute shell decomposition ---
        n_spec = len(spec_vals)
        shells = np.zeros((n_spec, n_reso, n_reso))
        density = np.zeros_like(shells)
        gradshells = np.zeros_like(shells)
        waveshells = np.zeros_like(shells)
        phase_offsets = np.zeros(n_spec)

        for i, frame in enumerate(inj_spec):
            frame_norm = frame / np.nanmax(frame)
            shell_stats = spec.shell_diagram(frame_norm, spec_master, spec_err[i, :], n_reso)
            shells[i] = shell_stats["shell_df"].values
            density[i] = shell_stats["density_map"]
            gradshells[i] = shell_stats["mean_grad_map"]
            waveshells[i] = shell_stats["mean_wave_map"]

        # --- 4. Save to HDF5 ---
        with h5py.File(file_name, "w") as hf:
            hf.create_dataset("astrodata", data=astrodata)
            hf.create_dataset("shells", data=shells)
            hf.create_dataset("density", data=density)
            hf.create_dataset("grad", data=gradshells)
            hf.create_dataset("waves", data=waveshells)
            hf.create_dataset("phase_offsets", data=phase_offsets)

        logger.info(f"Saved dataset: {file_name}")

        # --- 5. Reference dataset (no Doppler injection) ---
        if num_it == 0 and random_shifts:
            file_name_ref = os.path.join(data_dir, f"{root_name}_ref.h5")
            logger.info(f"Generating reference dataset: {file_name_ref}")

            astrodata_ref = astro_data(spec_vals, wave_cols=waves_obs)
            ds_ref = np.zeros(n_spec)
            phase_ref = np.zeros(n_spec)
            astrodata_ref = np.column_stack((astrodata_ref, ds_ref, phase_ref))

            shells_ref = np.zeros_like(shells)
            density_ref = np.zeros_like(shells)
            gradshells_ref = np.zeros_like(shells)
            waveshells_ref = np.zeros_like(shells)

            for i, frame_ref in enumerate(spec_vals):
                frame_norm = frame_ref / np.nanmax(frame_ref)
                shell_ref = spec.shell_diagram(frame_norm, spec_master, spec_err[i, :], n_reso)
                shells_ref[i] = shell_ref["shell_df"].values
                density_ref[i] = shell_ref["density_map"]
                gradshells_ref[i] = shell_ref["mean_grad_map"]
                waveshells_ref[i] = shell_ref["mean_wave_map"]

            with h5py.File(file_name_ref, "w") as hf:
                hf.create_dataset("astrodata", data=astrodata_ref)
                hf.create_dataset("shells", data=shells_ref)
                hf.create_dataset("density", data=density_ref)
                hf.create_dataset("grad", data=gradshells_ref)
                hf.create_dataset("waves", data=waveshells_ref)

            logger.info(f"Reference saved: {file_name_ref}")

    logger.info(f"Iteration {num_it} completed successfully.")