"""
----------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
----------------------------------------------------------

Methods to load and preprocess shell and astrophysical data
from HDF5 files used in Doppleriann. Supports merging datasets,
handling planetary signal injections, and generating shell-level inputs.
"""

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..physics.shell_utils import extract_shape_shell
from ..utils.logger_config import logger

# ==========================================================
# Global constants
# ==========================================================
ASTRO_COLUMNS = ["rv", "rv_err", "fwhm", "fwhm_err", "bis", "ds", "phase"]
# ==========================================================

def load_shell_astro_datah5(
    pis=[0.1, 1.0],
    periods=[20],
    use_mask=False,
    spec_type="act",
    use_residuals=False,
    use_temp=True,
    data_dir="data/shells/",
    dataset_name="default",
    selected_idx=None,
    remove_ds=False,
    remove_ds_val=5.0,
    remove_ds_period=100,
):
    """
    Load and merge multiple HDF5 shell datasets into full training arrays.

    Each file corresponds to a specific planetary injection setup, defined
    by a Doppler semi-amplitude (PI) and orbital period (P).

    Parameters
    ----------
    pis : list of float
        List of Doppler semi-amplitude values (m/s) to include. Examples: [0.1, 1.0].
    periods : list of float
        List of orbital periods (days) for which data is loaded.
    use_mask : bool, optional
        Whether to multiply shell data by the density matrix (default False).
    spec_type : str, optional
        Spectrum type identifier (act, or, sim) (default "act").
    use_residuals : bool, optional
        Subtract mean shell value to compute residuals (default False).
    use_temp : bool, optional
        Use temperature shells (True) or flux shells (False) (default True).
    data_dir : str, optional
        Directory containing the HDF5 files (default "data/shells/").
    dataset_name : str, optional
        Direct filename to load instead of looping through PI/period (default "default").
    selected_idx : list or ndarray, optional
        Specific indexes of spectra to select (default None).
    remove_ds : bool, optional
        Whether to remove the Doppler-shift component using a reference dataset (default False).
    remove_ds_val : float, optional
        Reference Doppler amplitude (m/s) for removal (default 5.0).
    remove_ds_period : float, optional
        Reference period (days) for removal dataset (default 100).

    Returns
    -------
    shell_data_full : ndarray
        Combined shell data array (N, H, W).
    astro_data_full : ndarray
        Corresponding astrophysical quantities.
    density_data_full : ndarray
        Density maps for each shell.
    grad_data_full : ndarray
        Gradient maps for each shell.
    wave_data_full : ndarray
        Wavelength maps for each shell.
    """
    str_shell_type = 'temp' if use_temp else 'flux'
    first = True  
    if dataset_name != 'default':
        shell_data_full, astro_data_full, density_data_full, grad_data_full, wave_data_full = load_and_process_file(dataset_name)
    else:
        for period in periods:
            for idx, ds_val in enumerate(pis):
                # Each file_name correspond to a different time series of shells with different planetary injections.
                logger.info(f"Loading file: {str_shell_type}_PI{ds_val}_P{period}_{spec_type}.h5")
                file_name = os.path.join(data_dir, f"{str_shell_type}_PI{ds_val}_P{period}_{spec_type}.h5")
                shell_data, astro_data, density, grad_data, wave_data = load_and_process_file(
                    file_name, use_residuals=use_residuals, use_mask=use_mask, selected_idx=selected_idx)
                if first:
                    shell_data_full = np.copy(shell_data)
                    density_data_full = np.copy(density)
                    wave_data_full = np.copy(wave_data)
                    grad_data_full = np.copy(grad_data)
                    astro_data_full = np.copy(astro_data)
                else:
                    shell_data_full = np.concatenate((shell_data_full, shell_data), axis=0)
                    density_data_full = np.concatenate((density_data_full, density), axis=0)
                    grad_data_full = np.concatenate((grad_data_full, grad_data), axis=0)
                    wave_data_full = np.concatenate((wave_data_full, wave_data), axis=0)
                    astro_data_full = np.concatenate((astro_data_full, astro_data), axis=0)
                if remove_ds:
                    file_name_dop = os.path.join(data_dir, f"{str_shell_type}_PI{remove_ds_val}_P{remove_ds_period}_{spec_type}.h5")
                    shell_data_dop, astro_data_dop, density_dop, grad_data_dop, wave_data_dop = load_and_process_file(
                        file_name_dop, use_residuals=use_residuals, use_mask=use_mask, selected_idx=selected_idx)
                    if first:
                        shells_dop = np.copy(shell_data_dop)
                        astrodatatest_dop = np.copy(astro_data_dop)
                        densitytest_dop = np.copy(density_dop)
                        gradtest_dop = np.copy(grad_data_dop)
                        wavestest_dop = np.copy(wave_data_dop)
                    else: 
                        shells_dop = np.concatenate((shells_dop, shell_data_dop), axis=0)
                        astrodatatest_dop = np.concatenate((astrodatatest_dop, astro_data_dop), axis=0)
                        densitytest_dop = np.concatenate((densitytest_dop, density_dop), axis=0)
                        gradtest_dop = np.concatenate((gradtest_dop, grad_data_dop), axis=0)
                        wavestest_dop = np.concatenate((wavestest_dop, wave_data_dop), axis=0)
                first = False
                
    if remove_ds:
        shape_shells = []
        for i in range(len(shell_data_full)):
            shape_shell = extract_shape_shell(shell_data_full[i], shells_dop[i])
            shape_shells.append(shape_shell)
        shell_data_full = np.array(shape_shells)

    return shell_data_full, astro_data_full, density_data_full, grad_data_full, wave_data_full


def load_shell_astro_datah5_random(
    use_mask=False,
    spec_type='act',
    use_residuals=False,
    use_temp=True,
    data_dir="data/shellsHiddenRandom/",
    selected_idx=None,
    remove_ds=False,
    remove_ds_val=5.0,
    remove_ds_period=100,
    min_ds=0.0,
    max_ds=np.inf,
    n_size= None,
):
    """
    Load a random subset of HDF5 shell datasets for stochastic training.

    Parameters
    ----------
    use_mask : bool, optional
        Whether to multiply shell data by the density matrix (default False).
    spec_type : str, optional
        Spectrum type identifier (act, or, sim) (default "act").
    use_residuals : bool, optional
        Subtract mean shell value to compute residuals (default False).
    use_temp : bool, optional
        Use temperature shells (True) or flux shells (False) (default True).
    data_dir : str, optional
        Directory containing the HDF5 files (default "data/shellsHiddenRandom/").
    selected_idx : list or ndarray, optional
        Specific indexes of spectra to select (default None).
    remove_ds : bool, optional
        Whether to remove the Doppler-shift component using a reference dataset (default False).
    remove_ds_val : float, optional
        Reference Doppler amplitude (m/s) for removal (default 5.0).
    remove_ds_period : float, optional
        Reference period (days) for removal dataset (default 100).
    min_ds : float, optional
        Minimum Doppler shift (m/s) for inclusion (default 0.0).
    max_ds : float, optional
        Maximum Doppler shift (m/s) for inclusion (default np.inf).
    n_size : int, optional
        Number of random files to load (default None).

    Returns
    -------
    shell_data_full : ndarray
        Combined shell data array (N, H, W).
    astro_data_full : ndarray
        Corresponding astrophysical quantities.
    density_data_full : ndarray
        Density maps for each shell.
    grad_data_full : ndarray
        Gradient maps for each shell.
    wave_data_full : ndarray
        Wavelength maps for each shell.
    """

    str_shell_type = 'temp' if use_temp else 'flux'
    pattern = f"{str_shell_type}_PI_random"
    file_list = sorted([f for f in os.listdir(data_dir) if f.startswith(pattern)])
    if isinstance(file_list, np.ndarray):
        file_list = file_list.tolist()
    if not file_list:
        raise RuntimeError("No matching .h5 files found in the specified directory.")
    
    logger.info(f"Selected files: {file_list}")

    first = True
    filtered_files = 0

    if remove_ds:
        file_name_dop = os.path.join(data_dir, f"{str_shell_type}_PI{remove_ds_val}_P{remove_ds_period}_{spec_type}.h5")
        shell_data_dop, astro_data_dop, density_dop, grad_data_dop, wave_data_dop = load_and_process_file(
            file_name_dop, use_residuals=use_residuals, use_mask=use_mask, selected_idx=selected_idx)
        shells_dop = shell_data_dop

    for file_name in file_list:
        full_path = os.path.join(data_dir, file_name)
        logger.info(f"Loading file: {file_name}")
        
        shell_data, astro_data, density, grad_data, wave_data = load_and_process_file(
            full_path, use_residuals=use_residuals, use_mask=use_mask, selected_idx=selected_idx
        )

        ds_abs = np.abs(astro_data[:, 5])
        logger.info(f"Before masking: min_ds: {np.min(ds_abs)}, max_ds: {np.max(ds_abs)}")
        

        # Create mask to keep only rows within [min_ds, max_ds]
        mask = (ds_abs >= min_ds) & (ds_abs <= max_ds)

        # If no valid rows, skip file entirely
        if not np.any(mask):
            logger.info(f"Skipping file {file_name}: no items within DS range [{min_ds}, {max_ds}]")
            continue

        # Apply mask to all related arrays
        astro_data = astro_data[mask]
        shell_data = shell_data[mask]
        density = density[mask]
        grad_data = grad_data[mask]
        wave_data = wave_data[mask]

        ds_abs = np.abs(astro_data[:, 5])
        logger.info(f"After masking: min_ds: {np.min(ds_abs)}, max_ds: {np.max(ds_abs)}")

        logger.info(f"Kept {np.sum(mask)} / {len(mask)} items in {file_name} after DS filtering")

        if first:
            shell_data_full = np.copy(shell_data)
            astro_data_full = np.copy(astro_data)
            density_data_full = np.copy(density)
            grad_data_full = np.copy(grad_data)
            wave_data_full = np.copy(wave_data)
            first = False
        else:
            shell_data_full = np.concatenate((shell_data_full, shell_data), axis=0)
            astro_data_full = np.concatenate((astro_data_full, astro_data), axis=0)
            density_data_full = np.concatenate((density_data_full, density), axis=0)
            grad_data_full = np.concatenate((grad_data_full, grad_data), axis=0)
            wave_data_full = np.concatenate((wave_data_full, wave_data), axis=0)
        filtered_files += 1
    logger.info(f"Total files loaded after filtering: {filtered_files} between ds {min_ds} and {max_ds}")
    if remove_ds:
        shape_shells = []
        for i in range(len(shell_data_full)):
            shape_shell = extract_shape_shell(shell_data_full[i], shells_dop[i])
            shape_shells.append(shape_shell)
        shell_data_full = np.array(shape_shells)

    if n_size is not None:
        if n_size >= shell_data_full.shape[0]:
            logger.warning(f"Requested n_size {n_size} exceeds available data {shell_data_full.shape[0]}. Using all available data.")
            n_size = shell_data_full.shape[0]    
        random_idx = np.random.permutation(shell_data_full.shape[0])[:n_size]
        shell_data_full = shell_data_full[random_idx]
        astro_data_full = astro_data_full[random_idx]
        density_data_full = density_data_full[random_idx]
        grad_data_full = grad_data_full[random_idx]
        wave_data_full = wave_data_full[random_idx]

    return shell_data_full, astro_data_full, density_data_full, grad_data_full, wave_data_full


def load_hdf5_data(file_name, key):
    """Load a specific dataset from an HDF5 file."""
    with h5py.File(file_name, "r") as hf:
        if key not in hf:
            raise KeyError(f"Dataset '{key}' not found in {file_name}")
        return hf[key][:]


def load_and_process_file(file_name, use_residuals=True, use_mask=False, selected_idx=None):
    """
    Load shell and astrophysical datasets from an HDF5 file.

    Compatible with both legacy (Pandas .to_hdf) and modern (h5py) DopplerIANN formats.
    """
    astro_data = None
    # astro_columns = None

    # --- Try Pandas first (old format) ---
    try:
        astro_data_df = pd.read_hdf(file_name, key="astrodata")
        astro_data = astro_data_df.to_numpy()
        # astro_columns = list(astro_data_df.columns)
        logger.debug(f"Loaded astrodata via Pandas from {file_name}")
    except (KeyError, OSError, ValueError, TypeError):
        # --- Fall back to h5py (new format) ---
        with h5py.File(file_name, "r") as hf:
            if "astrodata" not in hf:
                raise KeyError(f"No 'astrodata' dataset found in {file_name}")

            astro_data = hf["astrodata"][:]

    # --- Load other arrays ---
    shell_data = load_hdf5_data(file_name, "shells")
    density = load_hdf5_data(file_name, "density")
    grad_data = load_hdf5_data(file_name, "grad")
    wave_data = load_hdf5_data(file_name, "waves")

    logger.info(f"Loaded shell data shape: {shell_data.shape}")

    # --- Apply selection ---
    if selected_idx is not None:
        astro_data = astro_data[selected_idx]
        shell_data = shell_data[selected_idx]
        density = density[selected_idx]
        grad_data = grad_data[selected_idx]
        wave_data = wave_data[selected_idx]
        logger.info(f"Subset selection applied ({len(selected_idx)} samples)")

    # --- Normalize velocities ---
    astro_data[:, 0] -= np.mean(astro_data[:, 0])

    # --- Optional preprocessing ---
    if use_residuals:
        shell_data -= np.mean(shell_data)

    if use_mask:
        shell_data *= density

    return shell_data, astro_data, density, grad_data, wave_data