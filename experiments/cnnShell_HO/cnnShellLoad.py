shell_type_temp = True  # True for temp, False for Flux shells

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD, AdamW
from tensorflow.keras.losses import Huber, MeanSquaredError

# --- DopplerIANN imports ---
from doppleriann.networks import ShellCNN1D
from doppleriann.data import (
    MaskedStandardScaler3D,
    load_shell_astro_datah5,
)
from doppleriann.physics import generate_periodogram_test, recover_phase_offset, periodogram
from doppleriann.utils.logger_config import logger
from pathlib import Path

# === Project paths ===
# Current script directory
SCRIPT_DIR = Path(__file__).resolve().parent
# Project root directory (one level above /experiments/)
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# Paths
DATA_DIR = PROJECT_ROOT / "data"
LARGE_DATA_DIR = PROJECT_ROOT / "large_data"
LOCAL_MODELS_DIR = SCRIPT_DIR / "models"
LOCAL_OUTPUTS_DIR = SCRIPT_DIR / "outputs"

# Ensure local model dir exists (optional)
LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"[INFO] Data directory: {DATA_DIR}")
logger.info(f"[INFO] Local models directory: {LOCAL_MODELS_DIR}")
logger.info(f"[INFO] Local outputs directory: {LOCAL_OUTPUTS_DIR}")


np.random.seed(42)

# INPUT: shell, output regression of RV and DS, with CNN
show_pred_plots = False
hpc_device = True
ds_size_test = 0.25
period_test = 350
n_reso = 9  # 9 or 15
large_datadir = LARGE_DATA_DIR
shells_dir = f"{DATA_DIR}/shells/0/"
# 1636 random indices for training  around 80% of the total
random_idx_train = np.load(f"{DATA_DIR}/random_idx_train.npy")
# 400 random indices for testing  around 20% of the total
# Testing with unseen elements.
random_idx_test = np.load(f"{DATA_DIR}/random_idx_test.npy")
# Testing with the first N spectra (including elements of the training set)
# random_idx_test = np.arange(2036)[:2000]
use_residuals = True
use_density_shell_mask = True
## Training set settings

planetary_injections = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
periods_train = [20, 40, 60, 80, 100]

shell_type_str = "temp" if shell_type_temp else "flux"
# spec_types: 'act', 'or'
spec_types = ["act"]

# CNN settings
actfn = "selu"
loss_fn = "mean_squared_error"
epochs = 1000
# Callbacks
patience = 40
es = EarlyStopping(
    monitor="val_loss", patience=patience, min_delta=1e-5, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=patience // 2, min_delta=1e-5, min_lr=1e-6
)
callbacks = [es, reduce_lr]
dropout_rate = 0.2

bs = 128
conv_layers = [(256, 5), (512, 5)]
dense_layers = [512]
learning_rate = 0.0002

optimizer = Adam(learning_rate=learning_rate)

str_spec_types = "_".join(spec_types)
prefix_name = f"cnnshell_{n_reso}_{shell_type_str}_{str_spec_types}"
prefix_name += "_mask" if use_density_shell_mask else ""
prefix_name += "_res" if use_residuals else ""

# Common parameters for loading data
data_params = dict(
    pis=planetary_injections,
    periods=periods_train,
    use_temp=shell_type_temp,
    use_mask=use_density_shell_mask,
    use_residuals=use_residuals,
    data_dir=shells_dir,
    selected_idx=random_idx_train,
)

# Load data for each spec_type using a list comprehension
data_loader = [load_shell_astro_datah5(spec_type=st, **data_params) for st in spec_types]
shell_data_list, astrodata_list, _, _, _ = zip(*data_loader)

# Concatenate data across spec_types
shell_data_x = np.concatenate(shell_data_list, axis=0)
astrodata = np.concatenate(astrodata_list, axis=0)

logger.info(f"shell data shape {np.shape(shell_data_x)}")

# Scale shell_data_x using a custom 3D scaler
scalerx = MaskedStandardScaler3D()
scalerx.fit(shell_data_x)
x = scalerx.transform(shell_data_x)

# Process astrodata to extract target variable y (using first and last columns)
y = astrodata[:, [0, -2]]
scalery = StandardScaler()
scalery.fit(y)
y = scalery.transform(y)

logger.info(f"X SIZE: {np.shape(x)} | y SIZE: {np.shape(y)}")

model = ShellCNN1D(
    input_shape=(x.shape[1], x.shape[2]),
    n_outputs=2,
    conv_layers=conv_layers,
    dense_layers=dense_layers,
    dropout=dropout_rate,
    actfn=actfn,
)

cnn = model.load_model(f"{LOCAL_MODELS_DIR}/{prefix_name}.h5")
cnn.summary()

# === Testing the model ===
spec_type = "act"
time_df = pd.read_csv(f"{DATA_DIR}/time_df.csv")
dates = time_df["jdb"].values
dates = dates[random_idx_test]

fap = 0.001
min_period = 5
max_period = 2000


# ============================================================
# 0. Setup
# ============================================================
n_datasets = 10
colors = cm.tab10(np.linspace(0, 1, n_datasets))

# Storage
periods_rv_all = []
periods_ds_all = []
pred_rv_all = []
pred_ds_all = []

# ============================================================
# 1. Load RAW RV (only one time, shell=0)
# ============================================================
shells_dir_test = f"{DATA_DIR}/shells/0/"
test_set2, astrodatatest2, _, _, _ = load_shell_astro_datah5(
    pis=[ds_size_test],
    periods=[period_test],
    spec_type=spec_type,
    use_temp=shell_type_temp,
    use_mask=use_density_shell_mask,
    use_residuals=use_residuals,
    selected_idx=random_idx_test,
    data_dir=shells_dir_test,
)

raw_rv = astrodatatest2[:, 0]
raw_ds = astrodatatest2[:, -2]
inj_phase = astrodatatest2[:, -1]  # radians


# Raw RV periodogram
clp_raw, plevels_raw = periodogram(
    rvs=raw_rv, time=dates, err=None, fap=fap, min_period=min_period, max_period=max_period
)

# ============================================================
# 2. Loop over 10 datasets → store predictions, periodograms
# ============================================================
for i in range(n_datasets):
    shells_dir_test = f"{DATA_DIR}/shells/{i}/"

    test_set2, astrodatatest2_tmp, _, _, _ = load_shell_astro_datah5(
        pis=[ds_size_test],
        periods=[period_test],
        spec_type=spec_type,
        use_temp=shell_type_temp,
        use_mask=use_density_shell_mask,
        use_residuals=use_residuals,
        selected_idx=random_idx_test,
        data_dir=shells_dir_test,
    )

    test_set2 = scalerx.transform(test_set2)

    pred_mcdo = model.mcdo_predict(test_set2, cnn, mc_dropout_num=100)
    pred = scalery.inverse_transform(pred_mcdo["mean"])

    pred_rv_all.append(pred[:, 0])
    pred_ds_all.append(pred[:, 1])

    # RV periodogram
    clp_rv, _ = periodogram(
        rvs=pred[:, 0], time=dates, err=None, fap=fap, min_period=min_period, max_period=max_period
    )
    periods_rv_all.append(clp_rv)

    # DS periodogram
    clp_ds, plevels_ds = periodogram(
        rvs=pred[:, 1], time=dates, err=None, fap=fap, min_period=min_period, max_period=max_period
    )
    periods_ds_all.append(clp_ds)

# ============================================================
# 3. Compute MEAN prediction across 10 shells
# ============================================================
pred_rv_mean = np.mean(pred_rv_all, axis=0)
pred_ds_mean = np.mean(pred_ds_all, axis=0)

np.save(
    LOCAL_OUTPUTS_DIR
    / f"{prefix_name}_pred_rv_ds_shells_ds{ds_size_test}_P{int(period_test)}.npy",
    np.stack([np.stack(pred_rv_all, axis=0), np.stack(pred_ds_all, axis=0)], axis=-1),
)

# ============================================================
# 4. Compute RMS & MAD ranges across shells (RV & DS)
# ============================================================
rv_rms_all = []
rv_mad_all = []
ds_rms_all = []
ds_mad_all = []
mad_factor_corr = 1.482602218505602

for i in range(n_datasets):
    r_rv = raw_rv - pred_rv_all[i]
    rv_rms_all.append(np.sqrt(np.mean(r_rv**2)))
    rv_mad_all.append(mad_factor_corr * np.median(np.abs(r_rv - np.median(r_rv))))

    r_ds = raw_ds - pred_ds_all[i]
    ds_rms_all.append(np.sqrt(np.mean(r_ds**2)))
    ds_mad_all.append(mad_factor_corr * np.median(np.abs(r_ds - np.median(r_ds))))

rv_rms_min, rv_rms_max = np.min(rv_rms_all), np.max(rv_rms_all)
rv_mad_min, rv_mad_max = np.min(rv_mad_all), np.max(rv_mad_all)
ds_rms_min, ds_rms_max = np.min(ds_rms_all), np.max(ds_rms_all)
ds_mad_min, ds_mad_max = np.min(ds_mad_all), np.max(ds_mad_all)

# ============================================================
# 5. Phase plot for DS (single dataset, use stored injected phase)
# ============================================================
k_dataset = 0
shells_dir_test = f"{DATA_DIR}/shells/{k_dataset}/"


test_set2single, astrodatatest2_single, _, _, _ = load_shell_astro_datah5(
    pis=[ds_size_test],
    periods=[period_test],
    spec_type=spec_type,
    use_temp=shell_type_temp,
    use_mask=use_density_shell_mask,
    use_residuals=use_residuals,
    selected_idx=random_idx_test,
    data_dir=shells_dir_test,
)

test_set2single = scalerx.transform(test_set2single)

pred_mcdo = model.mcdo_predict(test_set2single, cnn, mc_dropout_num=100)
pred = scalery.inverse_transform(pred_mcdo["mean"])
# Stored injected phase in radians (includes the random phase_offset used in injection)
inj_phase_rad = astrodatatest2_single[:, -1]  # radians in [0, 2pi)
phi_inj = (inj_phase_rad / (2 * np.pi)) % 1.0  # cycles in [0,1)

# Truth and prediction DS time series (same timestamps / same rows)
inj_ds = astrodatatest2_single[:, -2]  # injected DS truth in m/s
pred_ds = pred[:, 1]  # predicted DS in m/s (single dataset)

# Sort by injected phase for clean plotting
idx = np.argsort(phi_inj)
phi_sorted = phi_inj[idx]
inj_sorted = inj_ds[idx]
pred_sorted = pred_ds[idx]

# Bin predictions in injected-phase bins (optional, for black dots)
bin_edges = np.arange(0.0, 1.0001, 0.1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
nbins = len(bin_centers)

binned_mean = np.full(nbins, np.nan)
binned_std = np.full(nbins, np.nan)

for k in range(nbins):
    m = (phi_sorted >= bin_edges[k]) & (phi_sorted < bin_edges[k + 1])
    if np.any(m):
        binned_mean[k] = np.mean(pred_sorted[m])
        binned_std[k] = np.std(pred_sorted[m])

# ------------------------------------------------------------
# (B) Injected truth: use the stored injected phase in astrodatatest2[:, -1]
#     This is the ONLY correct x-axis if you want to show the injected signal
#     without inventing an epoch/offset.
# ------------------------------------------------------------
inj_phase_rad = astrodatatest2[:, -1]  # radians, already wrapped in [0, 2π)
phase_inj = (inj_phase_rad / (2 * np.pi)) % 1.0  # cycles in [0, 1)
inj_ds = astrodatatest2[:, -2]  # injected DS truth in m/s

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
# 6. BIG FIGURE: 2 columns × 3 rows  (IMPROVED LAYOUT)
# ============================================================

# New layout: taller first row
fig = plt.figure(figsize=(16, 18))
# fig.suptitle( f" Injected period: {period_test} d | Injected DS: {ds_size_test} m/s | --- FAP {100*fap:.1f}%", fontsize=22)
gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.0, 1.0], hspace=0.35, wspace=0.25)

ax_rvP = fig.add_subplot(gs[0, 0])
ax_dsP = fig.add_subplot(gs[0, 1])
ax_rvTS = fig.add_subplot(gs[1, 0])
ax_dsPH = fig.add_subplot(gs[1, 1])
ax_rvRes = fig.add_subplot(gs[2, 0])
ax_dsRes = fig.add_subplot(gs[2, 1])

# Distinct color sequences
colors_rv = cm.Oranges(np.linspace(0.3, 1, n_datasets))
colors_ds = cm.Oranges(np.linspace(0.3, 1, n_datasets))


# ---------------------------------------------------------
# (1) RV Periodogram
# ---------------------------------------------------------
for clp, col in zip(periods_rv_all, colors_rv):
    ax_rvP.plot(1.0 / clp.freq, clp.power, color=col, alpha=0.5)

info_text = (
    f"Injected period: {period_test} d\nInjected DS: {ds_size_test} m/s\nFAP: {100 * fap:.1f}%"
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
# RAW RV in GREEN (fixed color)
ax_rvP.plot(1.0 / clp_raw.freq, clp_raw.power, color="green", lw=2, alpha=0.5)
ax_rvP.axhline(plevels_raw, color="gray", ls="--", lw=2)
ax_rvP.set_xscale("log")
ax_rvP.set_title("RV", fontsize=20)
ax_rvP.set_xlabel("Period (days)", fontsize=16)
ax_rvP.set_ylabel("Power", fontsize=16)
ax_rvP.tick_params(axis="both", labelsize=14)
# Create proxy artists to control legend colors explicitly
legend_handles = [
    Line2D([0], [0], color="tab:orange", lw=2, label="RV Predictions"),
    Line2D([0], [0], color="tab:green", lw=2, label="Raw RV (CCF)"),
]
ax_rvP.legend(handles=legend_handles, fontsize=14, loc="upper right")


# ---------------------------------------------------------
# (2) DS Periodogram + Inset Zoom
# ---------------------------------------------------------
for clp, col in zip(periods_ds_all, colors_ds):
    ax_dsP.plot(1.0 / clp.freq, clp.power, color=col, alpha=0.6)

ax_dsP.axhline(plevels_ds, color="gray", ls="--", lw=2)

ax_dsP.set_xscale("log")
ax_dsP.set_title("DS", fontsize=20)
ax_dsP.set_xlabel("Period (days)", fontsize=16)
ax_dsP.set_ylabel("Power", fontsize=16)
ax_dsP.tick_params(axis="both", labelsize=14)
# ax_dsP.legend([f"FAP = {100*fap:.1f}%"], fontsize=14, loc="upper right")


# --- Inset zoom around the injected period ---

axins = inset_axes(ax_dsP, width="48%", height="40%", loc="upper left", borderpad=1)

for clp, col in zip(periods_ds_all, colors_ds):
    axins.plot(1.0 / clp.freq, clp.power, color=col, alpha=0.6)

zoom_width = 0.12 * period_test
pmin = period_test - zoom_width
pmax = period_test + zoom_width

freqs = 1.0 / periods_ds_all[0].freq
mask = (freqs > pmin) & (freqs < pmax)
ymax_zoom = max(np.max(clp.power[mask]) for clp in periods_ds_all)

axins.set_xlim(pmin, pmax)
axins.set_ylim(0, ymax_zoom * 1.15)
axins.axhline(plevels_ds, color="gray", ls="--", lw=1)
# axins.set_xticks([])
axins.set_yticks([])
# axins.set_title(f"{period_test} d zoom", fontsize=10)


# ---------------------------------------------------------
# (3) RV Time Series – raw vs mean prediction
# ---------------------------------------------------------
ax_rvTS.scatter(dates, raw_rv, s=20, alpha=0.7, label="Raw RV (CCF)", color="tab:green")
ax_rvTS.scatter(
    dates, pred_rv_mean, s=20, alpha=0.7, label="Predicted RV (mean)", color="tab:orange"
)

# ax_rvTS.set_title("RV Time Series", fontsize=20)
ax_rvTS.set_xlabel("Time [BJD]", fontsize=16)
ax_rvTS.set_ylabel("RV [m/s]", fontsize=16)
ax_rvTS.tick_params(axis="both", labelsize=14)
# ax_rvTS.legend(fontsize=14)


# ---------------------------------------------------------
# (4) RV Residuals
# ---------------------------------------------------------
rv_residuals_mean = raw_rv - pred_rv_mean
ax_rvRes.scatter(dates, rv_residuals_mean, s=15, color="tab:blue", alpha=0.8)

ax_rvRes.axhline(0, color="k")
# ax_rvRes.set_title("RV Residuals", fontsize=20)
ax_rvRes.set_xlabel("Time [BJD]", fontsize=16)
ax_rvRes.set_ylabel("Residual RV [m/s]", fontsize=16)
ax_rvRes.tick_params(axis="both", labelsize=14)

ax_rvRes.legend(
    [
        f" RMS = [{rv_rms_min:.2f} - {rv_rms_max:.2f}] m/s \n MAD* = [{rv_mad_min:.2f} - {rv_mad_max:.2f}] m/s"
    ],
    fontsize=13,
    loc="upper right",
)


# ---------------------------------------------------------
# (5) DS Phase-Folded, Binned, Injected vs Predicted
# ---------------------------------------------------------

# Scatter predictions (optional)
ax_dsPH.scatter(phi_sorted, pred_sorted, s=8, alpha=0.25, color="tab:blue")

# Injected truth (gray curve)
ax_dsPH.plot(phi_sorted, inj_sorted, color="red", lw=2, alpha=0.9, label="Injected signal")
# Binned prediction (black dots)
ax_dsPH.errorbar(
    bin_centers,
    binned_mean,
    yerr=binned_std,
    fmt="o",
    color="k",
    capsize=3,
    label="Binned prediction",
)


# ax_dsPH.set_title("DS Phase-Folded", fontsize=20)
# ax_dsRes.set_xlim(0, 1)
ax_dsRes.set_ylim(-2, 2)
ax_dsPH.set_ylim(-2, 2)
ax_dsPH.set_xlabel("Phase", fontsize=16)
ax_dsPH.set_ylabel("DS [m/s]", fontsize=16)
ax_dsPH.tick_params(axis="both", labelsize=14)
ax_dsPH.legend(fontsize=14)


# ---------------------------------------------------------
# (6) DS Residuals
# ---------------------------------------------------------
ds_residuals_mean = raw_ds - pred_ds_mean

ax_dsRes.scatter(dates, ds_residuals_mean, s=10, alpha=0.7, color="tab:blue")

ax_dsRes.axhline(0, color="k", lw=1)

# ax_dsRes.set_title("DS Residuals", fontsize=20)
ax_dsRes.set_xlabel("Time [BJD]", fontsize=16)
ax_dsRes.set_ylabel("Residual DS [m/s]", fontsize=16)
ax_dsRes.tick_params(axis="both", labelsize=14)

# Autoscale or keep your manual y-limits:
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
plt.savefig(
    f"{LOCAL_OUTPUTS_DIR}/{prefix_name}_test_ds{ds_size_test}_period{period_test}_summary.pdf"
)
plt.show()
