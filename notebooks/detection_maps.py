import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase


def detection_map(x_values, y_values, data, fname="Detection_Map"):
    fig, ax = plt.subplots(figsize=(10, 4))

    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            color = "blue" if data[i, j] == 1 else "red"
            ax.plot(x, y, "o", color=color)

    ax.set_xlabel("Period (Days)", fontsize=14, weight="bold")
    ax.set_ylabel("Amplitude (m/s)", fontsize=14, weight="bold")
    ax.set_xticks(x_values)
    ax.set_yticks(y_values)

    fig.tight_layout(rect=[0, 0, 0.9, 1])

    cmap = ListedColormap(["red", "blue"])
    norm = BoundaryNorm([0, 0.5, 1], cmap.N)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    cb = ColorbarBase(
        cbar_ax, cmap=cmap, norm=norm, boundaries=[0, 0.5, 1], ticks=[0.25, 0.75]
    )
    cb.ax.set_yticklabels(["0", "1"])
    cb.ax.tick_params(labelsize=20)
    cb.set_label("Detection", fontsize=15, weight="bold", rotation=270, labelpad=13)

    fig.savefig(f"img/{fname}_binary.pdf", bbox_inches="tight")


def dots_map(
    x_values,
    y_values,
    data,
    fname="Detection_Map",
    title="Detection/Recovery Matrix",
    cbarlabel="% Detection Probability",
    vmin=None,
    vmax=None,
    cmap="coolwarm_r",
    mask=None,
):
    X, Y = np.meshgrid(np.arange(len(x_values)), np.arange(len(y_values)))

    x_flat = X.flatten()
    y_flat = Y.flatten()
    val_flat = data.flatten()

    if mask is not None:
        mask = mask.flatten()
        assert mask.shape == val_flat.shape, "Mask and data shape mismatch"
        x_flat = x_flat[mask == 1]
        y_flat = y_flat[mask == 1]
        val_flat = val_flat[mask == 1]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        x_flat,
        y_flat,
        c=val_flat,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=100,
        marker="o",
        edgecolor="k",
    )

    plt.yticks(ticks=np.arange(len(y_values)), labels=y_values)
    plt.xticks(
        ticks=np.arange(len(x_values)), labels=x_values, rotation=45, ha="right"
    )

    plt.ylabel("Amplitude (m/s)", fontsize=18, weight="bold")
    plt.xlabel("Period (days)", fontsize=18, weight="bold")
    plt.title(title, fontsize=18, weight="bold", pad=2)

    cbar = plt.colorbar(sc)
    cbar.set_label(cbarlabel, fontsize=17, weight="bold", labelpad=16)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.label.set_rotation(270)

    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    plt.tick_params(axis="both", labelsize=16, width=1.2)

    plt.tight_layout()
    plt.savefig(f"img/{fname}_dots.pdf", bbox_inches="tight")


# ============================================================
# Switches
# ============================================================
use_temp = True   # True for temp, False for flux
use_ho = False      # True for HO, False for CV

shell = "temp" if use_temp else "flux"

# Directories
data_dir = "experiments/cnnShell_HO/outputs/" if use_ho else "experiments/cnnShell_CV/"

# Grid
ds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
periods = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]

# ============================================================
# Load data
# ============================================================
if use_ho:
    fname = "HO_Temp" if use_temp else "HO_Flux"
    title_str = "HO Temperature" if use_temp else "HO Flux"

    base = f"{data_dir}"

    tag = f"cnnshell_9_{shell}_act_mask_res"

    detections_binary = pd.read_csv(f"{base}detections_binary_{tag}.csv", usecols=[1,2,3,4,5,6,7,8]).values.T
    detections        = pd.read_csv(f"{base}detections_{tag}.csv",        usecols=[1,2,3,4,5,6,7,8]).values.T
    amplitudes        = pd.read_csv(f"{base}amplitudes_{tag}.csv",        usecols=[1,2,3,4,5,6,7,8]).values.T
    amplitudes_perc   = pd.read_csv(f"{base}amplitudes_perc_{tag}.csv",   usecols=[1,2,3,4,5,6,7,8]).values.T
    phases            = pd.read_csv(f"{base}phases_{tag}.csv",            usecols=[1,2,3,4,5,6,7,8]).values.T
    periods_diff      = pd.read_csv(f"{base}periods_{tag}.csv",           usecols=[1,2,3,4,5,6,7,8]).values.T

    residuals_rv      = pd.read_csv(f"{base}residuals_rv_{tag}.csv",      usecols=[1,2,3,4,5,6,7,8]).values.T
    residuals_ds      = pd.read_csv(f"{base}residuals_ds_{tag}.csv",      usecols=[1,2,3,4,5,6,7,8]).values.T
    residuals_med_rv  = pd.read_csv(f"{base}residuals_med_rv_{tag}.csv",  usecols=[1,2,3,4,5,6,7,8]).values.T
    residuals_med_ds  = pd.read_csv(f"{base}residuals_med_ds_{tag}.csv",  usecols=[1,2,3,4,5,6,7,8]).values.T
    variance_rv       = pd.read_csv(f"{base}variance_rv_{tag}.csv",       usecols=[1,2,3,4,5,6,7,8]).values.T
    variance_ds       = pd.read_csv(f"{base}variance_ds_{tag}.csv",       usecols=[1,2,3,4,5,6,7,8]).values.T

else:
    fname = "CV_Temp" if use_temp else "CV_Flux"
    title_str = "CV Temperature" if use_temp else "CV Flux"

    base = f"{data_dir}"
    # Joined outputs (your join_chunks creates these)
    detections        = pd.read_csv(f"{base}detection_matrix_CV_{shell}_detections_cnn.csv",        usecols=[1,2,3,4,5,6,7,8]).values.T
    detections_binary = pd.read_csv(f"{base}detection_matrix_CV_{shell}_detections_binary_cnn.csv", usecols=[1,2,3,4,5,6,7,8]).values.T

    amplitudes        = pd.read_csv(f"{base}detection_matrix_CV_{shell}_amplitudes_cnn.csv",        usecols=[1,2,3,4,5,6,7,8]).values.T
    phases            = pd.read_csv(f"{base}detection_matrix_CV_{shell}_phases_cnn.csv",            usecols=[1,2,3,4,5,6,7,8]).values.T
    periods_diff      = pd.read_csv(f"{base}detection_matrix_CV_{shell}_periods_cnn.csv",           usecols=[1,2,3,4,5,6,7,8]).values.T

    # NEW: joined uncertainty/residual maps
    variance_rv       = pd.read_csv(f"{base}detection_matrix_CV_{shell}_variance_rv_cnn.csv",       usecols=[1,2,3,4,5,6,7,8]).values.T
    variance_ds       = pd.read_csv(f"{base}detection_matrix_CV_{shell}_variance_ds_cnn.csv",       usecols=[1,2,3,4,5,6,7,8]).values.T
    residuals_rv      = pd.read_csv(f"{base}detection_matrix_CV_{shell}_residuals_rv_cnn.csv",      usecols=[1,2,3,4,5,6,7,8]).values.T
    residuals_ds      = pd.read_csv(f"{base}detection_matrix_CV_{shell}_residuals_ds_cnn.csv",      usecols=[1,2,3,4,5,6,7,8]).values.T
    residuals_med_rv  = pd.read_csv(f"{base}detection_matrix_CV_{shell}_residuals_med_rv_cnn.csv",  usecols=[1,2,3,4,5,6,7,8]).values.T
    residuals_med_ds  = pd.read_csv(f"{base}detection_matrix_CV_{shell}_residuals_med_ds_cnn.csv",  usecols=[1,2,3,4,5,6,7,8]).values.T

    # Optional: if you join amplitudes_perc for CV, load it; otherwise compute it
    try:
        amplitudes_perc = pd.read_csv(
            f"{base}detection_matrix_CV_{shell}_amplitudes_perc_cnn.csv",
            usecols=[1,2,3,4,5,6,7,8]
        ).values.T
    except FileNotFoundError:
        amplitudes_perc = 100 * amplitudes / np.array(ds)[:, np.newaxis]


# ============================================================
# Post-processing
# ============================================================
# For CV, periods_diff in days -> convert to percent of true period (as you do)
periods_diff = 100 * periods_diff / np.array(periods)[np.newaxis, :]

# If you prefer binary derived from detections rather than reading file:
# detections_binary = (detections >= 0.7).astype(int)

# ============================================================
# Plots
# ============================================================
dots_map(periods, ds, detections, fname=f"{fname}_prob",
         title=f"Detection {title_str}", cbarlabel="Probability detection",
         vmin=0, vmax=1)
plt.show()

dots_map(periods, ds, detections_binary, fname=f"{fname}_binary",
         title=f"Detection (Binary) {title_str}", cbarlabel="Detection",
         vmin=0, vmax=1)
plt.show()

# Amplitude (%)
dots_map(periods, ds, amplitudes_perc, fname=f"{fname}_ampl",
         title=f"Amplitude difference {title_str}",
         cbarlabel="Amplitude difference (%)",
         vmin=0, vmax=50, cmap="jet", mask=detections_binary)
plt.show()

# Phase
dots_map(periods, ds, phases, fname=f"{fname}_phases",
         title=f"Phase difference {title_str}",
         cbarlabel="Phase difference (cycles)",
         vmin=0, vmax=0.4, cmap="jet", mask=detections_binary)
plt.show()

# Period difference (%)
dots_map(periods, ds, periods_diff, fname=f"{fname}_periods",
         title=f"Periods difference {title_str}",
         cbarlabel="Period difference (%)",
         vmin=0, vmax=4, cmap="jet", mask=detections_binary)
plt.show()

# Residuals (mean abs)
dots_map(periods, ds, residuals_rv, fname=f"{fname}_residuals_rv",
         title=f"Residuals RV {title_str}", cbarlabel="Residuals RV",
         vmin=None, vmax=None, cmap="jet", mask=detections_binary)
plt.show()

dots_map(periods, ds, residuals_ds, fname=f"{fname}_residuals_ds",
         title=f"Residuals DS {title_str}", cbarlabel="Residuals DS",
         vmin=None, vmax=None, cmap="jet", mask=detections_binary)
plt.show()

# Uncertainty (MCDO std; label “Std” is more accurate than “Variance” unless you squared it)
dots_map(periods, ds, variance_rv, fname=f"{fname}_uncertainty_rv",
         title=f"Uncertainty RV {title_str}", cbarlabel="MCDO std RV",
         vmin=None, vmax=None, cmap="jet", mask=detections_binary)
plt.show()

dots_map(periods, ds, variance_ds, fname=f"{fname}_uncertainty_ds",
         title=f"Uncertainty DS {title_str}", cbarlabel="MCDO std DS",
         vmin=None, vmax=None, cmap="jet", mask=detections_binary)
plt.show()