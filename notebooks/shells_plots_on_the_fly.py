import sys
import os
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

from doppleriann.physics import SpectrumData
from doppleriann.utils.logger_config import logger

# ============================================================
# LOAD DATA
# ============================================================
logger.info("Loading data...")

large_data_dir = "large_data"

n_reso = 9
doppler_shift = 20.0  # m/s

# ----------------------------------------------------------------------
# WAVES (length = 31066)
# ----------------------------------------------------------------------
wavelengths = np.loadtxt("data/waves_kitcat.txt")
spec_obj = SpectrumData(wavelengths)

# ----------------------------------------------------------------------
# DATES
# ----------------------------------------------------------------------
time_df = pd.read_csv("data/time_df.csv")
dates = pd.DatetimeIndex(time_df.date).to_julian_date().values

# ----------------------------------------------------------------------
# LOAD MASTER SPECTRA
# ----------------------------------------------------------------------
spec_type = "act"
flux_master_file = f"data/flux_kitcat_master_{spec_type}.npy"
temp_master_file = f"data/temp_kitcat_master_{spec_type}.npy"

flux_master_spec = np.load(flux_master_file)  # (31066,)
temp_master_spec = np.load(temp_master_file)  # (31066,)

# ----------------------------------------------------------------------
# LOAD FULL ERRORS (2D) AND COLLAPSE THEM TO MASTER ERR
# ----------------------------------------------------------------------
temp_err_val_file = f"{large_data_dir}/temp_kitcat_or_err.npy"
flux_err_val_file = f"{large_data_dir}/spectra_kitcat_act_err.npy"

try:
    flux_err_data = np.load(flux_err_val_file, allow_pickle=True)  # (2036, 31066)
    temp_err_data = np.load(temp_err_val_file, allow_pickle=True)  # (2036, 31066)
    logger.info("Loaded error arrays from disk.")
except (FileNotFoundError, ValueError, pickle.UnpicklingError, OSError):
    logger.warning(
        "Error files are missing or unreadable in large_data/. Using constant fallback spec_err = 1e-3."
    )
    n_wave = len(wavelengths)
    flux_err_data = None  # will be set after flux_spec_data is loaded
    temp_err_data = None  # will be set after temp_spec_data is loaded

# master_err_flux = np.sqrt(np.mean(flux_err_data**2, axis=0))  # (31066,)
# master_err_temp = np.sqrt(np.mean(temp_err_data**2, axis=0))  # (31066,)

# ----------------------------------------------------------------------
# LOAD FULL SPECTRAL TIME SERIES (needed for injection)
# ----------------------------------------------------------------------
full_flux_file = f"{large_data_dir}/spectra_kitcat_{spec_type}.npy"
full_temp_file = f"{large_data_dir}/temp_kitcat_{spec_type}.npy"

try:
    flux_spec_data = np.load(full_flux_file)  # (2036, 31066)
    temp_spec_data = np.load(full_temp_file)  # (2036, 31066)
    logger.info("Loaded full spectral time series from disk.")
except (FileNotFoundError, ValueError, pickle.UnpicklingError, OSError):
    logger.warning(
        "Full spectra are missing or unreadable in large_data/. Using master + small noise as fallback."
    )
    rng = np.random.default_rng(42)
    n_mock, n_wave = 10, len(wavelengths)
    flux_spec_data = flux_master_spec[np.newaxis, :] + rng.normal(0, 1e-3, (n_mock, n_wave))
    temp_spec_data = temp_master_spec[np.newaxis, :] + rng.normal(0, 1e-3, (n_mock, n_wave))

# Fallback constant errors if external drive was not available
if flux_err_data is None:
    n_spec, n_wave = flux_spec_data.shape
    flux_err_data = np.full((n_spec, n_wave), 1e-3)
    temp_err_data = np.full((n_spec, n_wave), 1e-3)

# ============================================================
# HELPER: build shells (unmasked + masked) for a single spectrum
# ============================================================


def make_shells_for_spec(spec, master_spec, spec_err):
    """
    Returns:
      shell_unmasked : (n_reso, n_reso)
      shell_masked   : (n_reso, n_reso)  (low-density cells zeroed)
      grad_centers   : (n_reso,)  bin centers along the gradient axis
      flux_centers   : (n_reso,)  bin centers along the flux axis
    """
    stats = spec_obj.shell_diagram(
        spec_input=spec,
        master_spec=master_spec,
        spec_err=spec_err,
        n_reso=n_reso,
        num_limits_factor=0.0,
    )
    shell_df = stats["shell_df"]
    shell = shell_df.values
    dens = stats["density_map"]

    # Bin centers come directly from the DataFrame axes
    grad_centers = np.array(shell_df.columns, dtype=float)  # X axis
    flux_centers = np.array(
        shell_df.index, dtype=float
    )  # Y axis (already flipped in shell_diagram)

    # Simple mask: zero cells with density below 10% of max
    dens_thr = 0.1 * np.max(dens) if np.max(dens) > 0 else 0.0
    shell_masked = shell.copy()
    shell_masked[dens < dens_thr] = 0.0

    return shell, shell_masked, grad_centers, flux_centers


# ============================================================
# PICK ONE EPOCH AND INJECT 50 m/s
# ============================================================
idx = 0  # choose first observation; can change if you want

# ----- FLUX -----
flux_unshift = flux_spec_data[idx]

inj_flux, shifts_flux, phases_flux = spec_obj.planet_inj(
    full_spec_data=flux_spec_data[idx : idx + 1],
    dates=dates[idx : idx + 1],
    doppler_shift_amplitude=doppler_shift,
    period_days=10.0,  # long period → nearly constant
    reference_date=dates[idx],
    phase_offset=np.pi / 2,  # sin=1 → shift = +K
)
flux_shift = inj_flux[0]

shell_flux_unshift, shell_flux_unshift_mask, grad_centers_flux, flux_centers_flux = (
    make_shells_for_spec(flux_unshift, flux_master_spec, flux_err_data[idx])
)
shell_flux_shift, shell_flux_shift_mask, _, _ = make_shells_for_spec(
    flux_shift, flux_master_spec, flux_err_data[idx]
)

# ----- TEMPERATURE -----
temp_unshift = temp_spec_data[idx]

inj_temp, shifts_temp, phases_temp = spec_obj.planet_inj(
    full_spec_data=temp_spec_data[idx : idx + 1],
    dates=dates[idx : idx + 1],
    doppler_shift_amplitude=doppler_shift,
    period_days=10.0,
    reference_date=dates[idx],
    phase_offset=np.pi / 2,
)
temp_shift = inj_temp[0]

shell_temp_unshift, shell_temp_unshift_mask, grad_centers_temp, flux_centers_temp = (
    make_shells_for_spec(temp_unshift, temp_master_spec, temp_err_data[idx])
)
shell_temp_shift, shell_temp_shift_mask, _, _ = make_shells_for_spec(
    temp_shift, temp_master_spec, temp_err_data[idx]
)

# ============================================================
# MIMIC ORIGINAL STRUCTURE: shells_* and astro_*
# ============================================================
# We pretend we have a "time axis" of length 1; 'frame' index is 0.

shells_flux = [
    np.expand_dims(shell_flux_unshift, axis=0),
    np.expand_dims(shell_flux_shift, axis=0),
    np.expand_dims(shell_flux_unshift_mask, axis=0),
    np.expand_dims(shell_flux_shift_mask, axis=0),
]

shells_temp = [
    np.expand_dims(shell_temp_unshift, axis=0),
    np.expand_dims(shell_temp_shift, axis=0),
    np.expand_dims(shell_temp_unshift_mask, axis=0),
    np.expand_dims(shell_temp_shift_mask, axis=0),
]

# astro_* just needs a column -2 with DS; we use a 3-column dummy:
astro_flux = [
    np.array([[dates[idx], 0.0, 0.0]]),
    np.array([[dates[idx], doppler_shift, 0.0]]),
    np.array([[dates[idx], 0.0, 0.0]]),
    np.array([[dates[idx], doppler_shift, 0.0]]),
]

astro_temp = [
    np.array([[dates[idx], 0.0, 0.0]]),
    np.array([[dates[idx], doppler_shift, 0.0]]),
    np.array([[dates[idx], 0.0, 0.0]]),
    np.array([[dates[idx], doppler_shift, 0.0]]),
]

# ============================================================
# FRAME SELECTION (same pattern as original)
# ============================================================
frame = np.argmax(astro_flux[1][:, -2])  # will be 0, but keeps API
print(f"Frame with max flux DS: {frame}, value: {astro_flux[1][frame, -2]}")

# ============================================================
# Compute vlims per group
# ============================================================
v_flux_unmasked = np.max([np.abs(shells_flux[0][frame]), np.abs(shells_flux[1][frame])]) * 0.8
v_flux_masked = np.max([np.abs(shells_flux[2][frame]), np.abs(shells_flux[3][frame])]) * 0.7

v_temp_unmasked = np.max([np.abs(shells_temp[0][frame]), np.abs(shells_temp[1][frame])]) * 0.8
v_temp_masked = np.max([np.abs(shells_temp[2][frame]), np.abs(shells_temp[3][frame])]) * 0.7

# ============================================================
# PLOTTING (your original code, unchanged)
# ============================================================
fig = plt.figure(figsize=(15, 7))
gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 0.24, 1, 1], wspace=0.2, hspace=0.2)
axes = np.empty((2, 4), dtype=object)

# Create axes (skip column 2 for spacing)
for row in range(2):
    for i in range(4):
        col = i if i < 2 else i + 1  # insert gap at column 2
        axes[row, i] = fig.add_subplot(gs[row, col])

cmap = "seismic"
titles = [
    "Shell without shift",
    "Shell with shift",
    "Masked shell without shift",
    "Masked shell with shift",
]


def plot_cell(
    ax, shell, ds, vlim, cmap, x_lim, y_lim=(0, 1), show_title=False, ylabel=None, xlabel=None
):
    im = ax.imshow(shell, cmap=cmap, vmin=-vlim, vmax=vlim, origin="lower", aspect="equal")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)

    n = shell.shape[0]
    tick_positions = list(range(n))

    y_labels = np.linspace(y_lim[0], y_lim[1], n)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f"{v:.1f}" for v in y_labels])

    x_labels = np.linspace(x_lim[0], x_lim[1], n)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{v:.0f}" for v in x_labels])

    if show_title:
        ax.set_title(f"{titles[i]}\nDS = {ds:.2f} m/s", fontsize=10)
    ax.tick_params(labelsize=9)
    return im


# Plot Flux (top row)
for i in range(4):
    vlim = v_flux_unmasked if i < 2 else v_flux_masked
    plot_cell(
        axes[0, i],
        shells_flux[i][frame],
        astro_flux[i][frame, -2],
        vlim,
        cmap,
        x_lim=(-15, 15),
        show_title=True,
        ylabel="Normalized Flux" if i == 0 else None,
        xlabel=r"$\partial F/\partial v$",
    )

# Plot Temp (bottom row)
for i in range(4):
    vlim = v_temp_unmasked if i < 2 else v_temp_masked
    plot_cell(
        axes[1, i],
        shells_temp[i][frame],
        astro_temp[i][frame, -2],
        vlim,
        cmap,
        x_lim=(-30, 30),
        show_title=False,
        ylabel="Normalized Temperature" if i == 0 else None,
        xlabel=r"$\partial T/\partial v$",
    )

# Colorbars
cb_flux_unmasked = plt.cm.ScalarMappable(
    norm=mcolors.Normalize(vmin=-v_flux_unmasked, vmax=v_flux_unmasked), cmap=cmap
)
cb_flux_masked = plt.cm.ScalarMappable(
    norm=mcolors.Normalize(vmin=-v_flux_masked, vmax=v_flux_masked), cmap=cmap
)
cb_temp_unmasked = plt.cm.ScalarMappable(
    norm=mcolors.Normalize(vmin=-v_temp_unmasked, vmax=v_temp_unmasked), cmap=cmap
)
cb_temp_masked = plt.cm.ScalarMappable(
    norm=mcolors.Normalize(vmin=-v_temp_masked, vmax=v_temp_masked), cmap=cmap
)

for cb, label in [
    (fig.colorbar(cb_flux_unmasked, ax=axes[0, :2], fraction=0.05, pad=0.02), r"$\Delta F$"),
    (fig.colorbar(cb_flux_masked, ax=axes[0, 2:], fraction=0.05, pad=0.02), r"$\Delta F$"),
    (fig.colorbar(cb_temp_unmasked, ax=axes[1, :2], fraction=0.05, pad=0.02), r"$\Delta T$"),
    (fig.colorbar(cb_temp_masked, ax=axes[1, 2:], fraction=0.05, pad=0.02), r"$\Delta T$"),
]:
    cb.ax.set_title(label, fontsize=11, pad=6, loc="center", rotation=0)

plt.tight_layout(rect=[0, 0, 0.92, 1])
plt.savefig("shells_separate_colorbars_spaced.png", dpi=500)
plt.savefig("shells_separate_colorbars_spaced.pdf")
plt.show()
