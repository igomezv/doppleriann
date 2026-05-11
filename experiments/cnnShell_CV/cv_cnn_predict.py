import sys
import pickle
import numpy as np
from pathlib import Path

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit

# --- DopplerIANN imports ---
from doppleriann.networks import ShellCNN1D
from doppleriann.data import MaskedStandardScaler3D, load_shell_astro_datah5
from doppleriann.physics import (
    recover_phase_offset,
    periodogram,
)
from doppleriann.utils.logger_config import logger


# ============================================================
# Paths & config
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
LARGE_DATA_DIR = PROJECT_ROOT / "large_data"
LOCAL_MODELS_DIR = SCRIPT_DIR / "models/models"
LOCAL_OUTPUTS_DIR = SCRIPT_DIR / "outputs"

# kept for compatibility, not used for data selection
nrun = int(sys.argv[1]) if len(sys.argv) > 1 else 0

shell_type_temp = True
use_residuals = True
use_density_shell_mask = True

spec_types = ["act"]
n_reso = 9

shell_type_str = "temp" if shell_type_temp else "flux"
str_spec_types = "_".join(spec_types)

prefix_name = f"cnnshellCV5_{n_reso}_{shell_type_str}_{str_spec_types}"
prefix_name += "_mask" if use_density_shell_mask else ""
prefix_name += "_res" if use_residuals else ""


# base_shells_dir = DATA_DIR / f"shells{n_reso}"
base_shells_dir = DATA_DIR / f"shells"
# Chosen (ds, P) to evaluate — change here as needed
ds_plot = 0.2
period_plot = 100

# ============================================================
# Load time axis
# ============================================================

time_df = pd.read_csv(DATA_DIR / "time_df.csv")
dates = time_df["jdb"].values
n_samples = len(dates)

# ============================================================
# Load fold test indices
# ============================================================

num_folds = 5
fold_test_indices = []
for fold_id in range(num_folds):
    path = LOCAL_OUTPUTS_DIR / f"{prefix_name}_fold{fold_id}_test_idx.txt"
    idx = np.loadtxt(path, dtype=int)
    fold_test_indices.append(idx)
    logger.info(f"Loaded fold {fold_id} test idx (len={len(idx)})")

# ============================================================
# Prepare per-shell prediction arrays (10 shells 0–9)
# ============================================================

n_shells = 10
pred_rv_all = [np.zeros(n_samples) for _ in range(n_shells)]
pred_ds_all = [np.zeros(n_samples) for _ in range(n_shells)]

# ============================================================
# Main inference: for each fold and each shell
# predict on that fold's test indices and stitch
# ============================================================

for fold_id in range(num_folds):
    logger.info(f"\n=== FOLD {fold_id} ===")

    # ------------------------------
    # Load scalers
    # ------------------------------
    with open(LOCAL_MODELS_DIR / f"{prefix_name}_fold{fold_id}_scalerx.pkl", "rb") as f:
        scalerx = pickle.load(f)
    with open(LOCAL_MODELS_DIR / f"{prefix_name}_fold{fold_id}_scalery.pkl", "rb") as f:
        scalery = pickle.load(f)

    # ------------------------------
    # Load model EXACTLY like HO
    # ------------------------------
    model_path = LOCAL_MODELS_DIR / f"{prefix_name}_fold{fold_id}.h5"
    logger.info(f"Loading model from {model_path}")

    dummy_wrapper = ShellCNN1D(
        input_shape=(1, 1),  # not used during load_model
        n_outputs=2,
        conv_layers=None,
        dense_layers=None,
        dropout=None,
        actfn=None,
    )
    cnn = dummy_wrapper.load_model(model_path)
    logger.info("Model loaded successfully with custom layers.")

    test_idx = fold_test_indices[fold_id]

    # ------------------------------
    # Predict for each shell (0–9) on this fold's test indices
    # ------------------------------
    for shell_id in range(n_shells):
        shells_dir = base_shells_dir / f"{shell_id}"

        params = dict(
            pis=[ds_plot],
            periods=[period_plot],
            use_temp=shell_type_temp,
            use_mask=use_density_shell_mask,
            use_residuals=use_residuals,
            data_dir=shells_dir,
            selected_idx=test_idx,
        )

        loader = [load_shell_astro_datah5(spec_type=s, **params) for s in spec_types]
        shell_list, _, _, _, _ = zip(*loader)
        shell_test = np.concatenate(shell_list, axis=0)

        x_test = scalerx.transform(shell_test)
        pred_scaled = cnn.predict(x_test, verbose=0)
        pred_phys = scalery.inverse_transform(pred_scaled)

        pred_rv_all[shell_id][test_idx] = pred_phys[:, 0]
        pred_ds_all[shell_id][test_idx] = pred_phys[:, 1]

# Stack predictions: shape (10, n_samples)
pred_rv_all_arr = np.stack(pred_rv_all, axis=0)
pred_ds_all_arr = np.stack(pred_ds_all, axis=0)

np.save(
    LOCAL_OUTPUTS_DIR / f"{prefix_name}_CV5_pred_rv_ds_shells_ds{ds_plot}_P{int(period_plot)}.npy",
    np.stack([pred_rv_all_arr, pred_ds_all_arr], axis=-1),
)
logger.info(
    "Stitched predictions per shell. Shapes: RV %s, DS %s",
    pred_rv_all_arr.shape,
    pred_ds_all_arr.shape,
)

# ============================================================
# Load RAW RV & DS for each shell directory (correct phases)
# + injected Keplerian curve
# ============================================================

raw_rv_all = np.zeros_like(pred_rv_all_arr)  # (10, n_samples)
raw_ds_all = np.zeros_like(pred_ds_all_arr)  # (10, n_samples)
inj_all = np.zeros_like(pred_ds_all_arr)  # injected Keplerian column (10, n_samples)

for shell_id in range(n_shells):
    shells_dir = base_shells_dir / f"{shell_id}"
    params_raw = dict(
        pis=[ds_plot],
        periods=[period_plot],
        use_temp=shell_type_temp,
        use_mask=use_density_shell_mask,
        use_residuals=use_residuals,
        data_dir=shells_dir,
        selected_idx=np.arange(n_samples),
    )
    loader_raw = [load_shell_astro_datah5(spec_type=s, **params_raw) for s in spec_types]
    _, astro_raw_list, _, _, _ = zip(*loader_raw)
    astro_raw = np.concatenate(astro_raw_list, axis=0)

    raw_rv_all[shell_id] = astro_raw[:, 0]
    raw_ds_all[shell_id] = astro_raw[:, -2]  # DS
    inj_all[shell_id] = astro_raw[:, -1]  # injected phase in radians

# Mean raw and mean injection across shells
raw_rv_mean = np.mean(raw_rv_all, axis=0)
raw_ds_mean = np.mean(raw_ds_all, axis=0)
# inj_mean    = np.mean(inj_all, axis=0)

# ============================================================
# Build HO-style diagnostics using the 10 shell predictions
# ============================================================

fap = 0.001
min_period = 5
max_period = 2000

n_datasets = n_shells

# Storage for GLS objects
periods_rv_all_gls = []
periods_ds_all_gls = []

# Raw RV periodogram (mean raw)
clp_raw_rv_mean, plevels_raw_rv_mean = periodogram(
    rvs=raw_rv_mean, time=dates, err=None, fap=fap, min_period=min_period, max_period=max_period
)

# Per-shell GLS for predictions
for i in range(n_datasets):
    clp_rv, _ = periodogram(
        rvs=pred_rv_all_arr[i],
        time=dates,
        err=None,
        fap=fap,
        min_period=min_period,
        max_period=max_period,
    )
    periods_rv_all_gls.append(clp_rv)

    clp_ds, plevels_ds_shell = periodogram(
        rvs=pred_ds_all_arr[i],
        time=dates,
        err=None,
        fap=fap,
        min_period=min_period,
        max_period=max_period,
    )
    periods_ds_all_gls.append(clp_ds)

# Mean prediction across 10 shells
pred_rv_mean = np.mean(pred_rv_all_arr, axis=0)
pred_ds_mean = np.mean(pred_ds_all_arr, axis=0)

# Periodogram of mean DS prediction (for FAP + phase)
clp_ds_mean, plevels_ds_mean = periodogram(
    rvs=pred_ds_mean, time=dates, err=None, fap=fap, min_period=min_period, max_period=max_period
)

# ============================================================
# RMS & MAD across shells (RV & DS)
# ============================================================

rv_rms_all = []
rv_mad_all = []
ds_rms_all = []
ds_mad_all = []
mad_factor_corr = 1.482602218505602

for i in range(n_datasets):
    r_rv = raw_rv_all[i] - pred_rv_all_arr[i]
    rv_rms_all.append(np.sqrt(np.mean(r_rv**2)))
    rv_mad_all.append(mad_factor_corr * np.median(np.abs(r_rv - np.median(r_rv))))

    r_ds = raw_ds_all[i] - pred_ds_all_arr[i]
    ds_rms_all.append(np.sqrt(np.mean(r_ds**2)))
    ds_mad_all.append(mad_factor_corr * np.median(np.abs(r_ds - np.median(r_ds))))

rv_rms_min, rv_rms_max = np.min(rv_rms_all), np.max(rv_rms_all)
rv_mad_min, rv_mad_max = np.min(rv_mad_all), np.max(rv_mad_all)
ds_rms_min, ds_rms_max = np.min(ds_rms_all), np.max(ds_rms_all)
ds_mad_min, ds_mad_max = np.min(ds_mad_all), np.max(ds_mad_all)

# ============================================================
# Phase folding for DS MEAN prediction
# ============================================================

shell_phase_id = 0
# Use one shell only (true injected phase reference)
inj_phase_rad = inj_all[shell_phase_id]  # radians
phi_inj = (inj_phase_rad / (2 * np.pi)) % 1.0  # cycles
pred_ds_single = pred_ds_all_arr[shell_phase_id]
inj_ds_single = raw_ds_all[shell_phase_id]

# Sort by injected phase
idx_sort = np.argsort(phi_inj)
phase_sorted = phi_inj[idx_sort]
pred_sorted = pred_ds_single[idx_sort]
inj_sorted = inj_ds_single[idx_sort]

# Phase binning: fixed 0.1 steps
bin_edges = np.arange(0.0, 1.0001, 0.1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
nbins = len(bin_centers)

binned_mean = np.zeros(nbins)
binned_std = np.zeros(nbins)  # scatter within each bin
binned_err = np.zeros(nbins)  # standard error of the mean

for i in range(nbins):
    m = (phase_sorted >= bin_edges[i]) & (phase_sorted < bin_edges[i + 1])
    n_bin = np.sum(m)

    if n_bin > 0:
        vals = pred_sorted[m]
        binned_mean[i] = np.mean(vals)

        if n_bin > 1:
            binned_std[i] = np.std(vals, ddof=1)
            binned_err[i] = binned_std[i] / np.sqrt(n_bin)
        else:
            binned_std[i] = np.nan
            binned_err[i] = np.nan
    else:
        binned_mean[i] = np.nan
        binned_std[i] = np.nan
        binned_err[i] = np.nan


# ------------------------------------------------------------
# (B) Injected truth: use the stored injected phase in astrodatatest2[:, -1]
#     This is the ONLY correct x-axis if you want to show the injected signal
#     without inventing an epoch/offset.
# ------------------------------------------------------------
phase_inj = (inj_phase_rad / (2 * np.pi)) % 1.0  # cycles in [0, 1)
inj_ds = inj_ds_single  # injected DS truth in m/s

idx_inj = np.argsort(phase_inj)
phase_inj_sorted = phase_inj[idx_inj]
inj_sorted = inj_ds[idx_inj]


# ------------------------------------------------------------
# (C) To smooth injected curve, we fit a sinusoid to the injected DS as a function of injected phase.
# ------------------------------------------------------------
def sine_truth_phase(ph, A, phi, C):
    return A * np.sin(2 * np.pi * ph + phi) + C


# robust-ish initial guesses
A0_t = 0.5 * (np.nanmax(inj_ds) - np.nanmin(inj_ds))
C0_t = np.nanmean(inj_ds)
phi0_t = 0.0

popt_t, _ = curve_fit(
    sine_truth_phase, phase_inj_sorted, inj_sorted, p0=[A0_t, phi0_t, C0_t], maxfev=20000
)
A_t, phi_t, C_t = popt_t

phase_truth_dense = np.linspace(0, 1, 500)
inj_dense = sine_truth_phase(phase_truth_dense, A_t, phi_t, C_t)

# ============================================================
# BIG FIGURE: 2 columns × 3 rows (like HO)
# ============================================================

fig = plt.figure(figsize=(16, 18))
gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.0, 1.0], hspace=0.35, wspace=0.25)

ax_rvP = fig.add_subplot(gs[0, 0])
ax_dsP = fig.add_subplot(gs[0, 1])
ax_rvTS = fig.add_subplot(gs[1, 0])
ax_dsPH = fig.add_subplot(gs[1, 1])
ax_rvRes = fig.add_subplot(gs[2, 0])
ax_dsRes = fig.add_subplot(gs[2, 1])

# ---------------------------------------------------------
# (1) RV Periodogram — 10 shells
# ---------------------------------------------------------

colors_rv = cm.Oranges(np.linspace(0.3, 1, n_datasets))

for clp_rv_shell, col in zip(periods_rv_all_gls, colors_rv):
    ax_rvP.plot(1.0 / clp_rv_shell.freq, clp_rv_shell.power, color=col, alpha=0.5)

info_text = (
    f"Injected period: {period_plot:.0f} d\nInjected DS: {ds_plot} m/s\nFAP: {100 * fap:.1f}%"
)

ax_rvP.text(
    0.05,
    0.95,
    info_text,
    transform=ax_rvP.transAxes,
    fontsize=16,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="white"),
)

# RAW RV (mean) in green
ax_rvP.plot(1.0 / clp_raw_rv_mean.freq, clp_raw_rv_mean.power, color="green", lw=2, alpha=0.7)
ax_rvP.axhline(plevels_raw_rv_mean, color="gray", ls="--", lw=2)
ax_rvP.set_xscale("log")
ax_rvP.set_title("RV", fontsize=20)
ax_rvP.set_xlabel("Period (days)", fontsize=16)
ax_rvP.set_ylabel("Power", fontsize=16)
ax_rvP.tick_params(axis="both", labelsize=14)

legend_handles = [
    Line2D([0], [0], color="tab:green", lw=2, label="CCF RV"),
    Line2D([0], [0], color="tab:orange", lw=2, label="RV Predictions"),
]
ax_rvP.legend(handles=legend_handles, fontsize=14, loc="upper right")

# ---------------------------------------------------------
# (2) DS Periodogram + inset zoom (10 shells)
# ---------------------------------------------------------

colors_ds = cm.Oranges(np.linspace(0.3, 1, n_datasets))

for clp_ds_shell, col in zip(periods_ds_all_gls, colors_ds):
    ax_dsP.plot(1.0 / clp_ds_shell.freq, clp_ds_shell.power, color=col, alpha=0.6)

ax_dsP.axhline(plevels_ds_mean, color="gray", ls="--", lw=2)
ax_dsP.set_xscale("log")
ax_dsP.set_title("DS", fontsize=20)
ax_dsP.set_xlabel("Period (days)", fontsize=16)
ax_dsP.set_ylabel("Power", fontsize=16)
ax_dsP.tick_params(axis="both", labelsize=14)

# Inset zoom around injected period — placed flush to top frame (~1 mm below)
# [left, bottom, width, height] in axes-fraction coordinates
axins = ax_dsP.inset_axes([0.12, 0.58, 0.35, 0.40])

for clp_ds_shell, col in zip(periods_ds_all_gls, colors_ds):
    axins.plot(1.0 / clp_ds_shell.freq, clp_ds_shell.power, color=col, alpha=0.6)

zoom_width = 0.12 * period_plot
pmin_zoom = period_plot - zoom_width
pmax_zoom = period_plot + zoom_width

freqs_zoom = 1.0 / periods_ds_all_gls[0].freq
mask_zoom = (freqs_zoom > pmin_zoom) & (freqs_zoom < pmax_zoom)
ymax_zoom = max(np.max(clp_ds_shell.power[mask_zoom]) for clp_ds_shell in periods_ds_all_gls)

axins.set_xlim(pmin_zoom, pmax_zoom)
axins.set_ylim(0, ymax_zoom * 1.15)
axins.axhline(plevels_ds_mean, color="gray", ls="--", lw=1)

# Y-axis: display as integers representing values ×10⁻²
import matplotlib.ticker as mticker

axins.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}"))
axins.tick_params(axis="y", labelsize=10)
axins.tick_params(axis="x", labelsize=10)

# ---------------------------------------------------------
# (3) RV Time Series — raw vs predictions
# ---------------------------------------------------------

ax_rvTS.scatter(dates, raw_rv_mean, s=20, alpha=0.7, label="Mean Raw RV", color="tab:green")

for y_shell in pred_rv_all_arr:
    ax_rvTS.scatter(dates, y_shell, s=8, alpha=0.15, color="tab:orange")

ax_rvTS.scatter(
    dates, pred_rv_mean, s=20, alpha=0.7, label="Predicted RV (mean)", color="tab:orange"
)

ax_rvTS.set_xlabel("Time [BJD]", fontsize=16)
ax_rvTS.set_ylabel("RV [m/s]", fontsize=16)
ax_rvTS.tick_params(axis="both", labelsize=14)
ax_rvTS.legend(fontsize=14)

# ---------------------------------------------------------
# (4) RV Residuals (mean)
# ---------------------------------------------------------

rv_residuals_mean = raw_rv_mean - pred_rv_mean
ax_rvRes.scatter(dates, rv_residuals_mean, s=15, color="tab:blue", alpha=0.8)

ax_rvRes.axhline(0, color="k")
ax_rvRes.set_xlabel("Time [BJD]", fontsize=16)
ax_rvRes.set_ylabel("Residual RV [m/s]", fontsize=16)
ax_rvRes.tick_params(axis="both", labelsize=14)

ax_rvRes.legend(
    [
        f" RMS = [{rv_rms_min:.2f} - {rv_rms_max:.2f}] m/s \n"
        f" MAD* = [{rv_mad_min:.2f} - {rv_mad_max:.2f}] m/s"
    ],
    fontsize=13,
    loc="upper right",
)

# ---------------------------------------------------------
# (5) DS Phase-Folded, Binned, Injected vs Predicted
# ---------------------------------------------------------

# Scatter predictions (optional)
ax_dsPH.scatter(phase_sorted, pred_sorted, s=8, alpha=0.25, color="tab:blue")

# Injected truth (gray curve)
ax_dsPH.plot(phase_sorted, inj_sorted, color="red", lw=2, alpha=0.9, label="Injected signal")
# Binned prediction (black dots)
ax_dsPH.errorbar(
    bin_centers,
    binned_mean,
    yerr=binned_err,
    fmt="o",
    color="k",
    capsize=3,
    label="Binned prediction",
)

ax_dsRes.set_ylim(-2, 2)
ax_dsPH.set_ylim(-0.6, 0.6)
ax_dsPH.set_xlabel("Phase", fontsize=16)
ax_dsPH.set_ylabel("DS [m/s]", fontsize=16)
ax_dsPH.tick_params(axis="both", labelsize=14)
ax_dsPH.legend(fontsize=14)

# ---------------------------------------------------------
# (6) DS Residuals (mean)
# ---------------------------------------------------------

ds_residuals_mean = raw_ds_mean - pred_ds_mean
ax_dsRes.scatter(dates, ds_residuals_mean, s=10, alpha=0.7, color="tab:blue")

ax_dsRes.axhline(0, color="k", lw=1)
ax_dsRes.set_xlabel("Time [BJD]", fontsize=16)
ax_dsRes.set_ylabel("Residual DS [m/s]", fontsize=16)
ax_dsRes.tick_params(axis="both", labelsize=14)
ax_dsRes.set_ylim(-2, 2)

ax_dsRes.legend(
    [
        f"RMS = [{ds_rms_min:.2f} – {ds_rms_max:.2f}] m/s\n"
        f"MAD* = [{ds_mad_min:.2f} – {ds_mad_max:.2f}] m/s"
    ],
    fontsize=13,
    loc="upper right",
)

# ---------------------------------------------------------
plt.savefig(LOCAL_OUTPUTS_DIR / f"{prefix_name}_CV_summary_ds{ds_plot}_P{int(period_plot)}.pdf")
plt.show()
