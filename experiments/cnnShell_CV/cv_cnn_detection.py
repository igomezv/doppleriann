#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

# --------------------------------------------------------------
# DopplerIANN imports
# --------------------------------------------------------------
from doppleriann.networks import ShellCNN1D
from doppleriann.data import load_shell_astro_datah5
from doppleriann.physics import periodogram, recover_phase_offset, circ_dist_cycles
from doppleriann.utils.logger_config import logger

# --------------------------------------------------------------
# Project paths
# --------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
LARGE_DATA_DIR = PROJECT_ROOT / "large_data"
LOCAL_MODELS_DIR = SCRIPT_DIR / "models/models"
LOCAL_OUTPUTS_DIR = SCRIPT_DIR / "outputs"

LOCAL_MODELS_DIR.mkdir(exist_ok=True)
LOCAL_OUTPUTS_DIR.mkdir(exist_ok=True)

logger.info(f"DATA_DIR = {DATA_DIR}")
logger.info(f"MODELS   = {LOCAL_MODELS_DIR}")
logger.info(f"OUTPUTS  = {LOCAL_OUTPUTS_DIR}")

# --------------------------------------------------------------
# Read nrun = which chunk of periods
# --------------------------------------------------------------
nrun = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# --------------------------------------------------------------
# CV/Model Setup
# --------------------------------------------------------------
num_folds = 5               # fold0..fold4
n_shells = 10               # directories 0..9  (DIFFERENT PHASES!)

shell_type_temp = False
use_residuals = True
use_mask = True

spec_types = ["act"]
n_reso = 9

shell_type_str = "temp" if shell_type_temp else "flux"
str_spec_types = "_".join(spec_types)

prefix_name = f"cnnshellCV5_{n_reso}_{shell_type_str}_{str_spec_types}"
prefix_name += "_mask" if use_mask else ""
prefix_name += "_res" if use_residuals else ""

# base_shells_dir = DATA_DIR / f"shells{n_reso}"
base_shells_dir = DATA_DIR / f"shells"

# --------------------------------------------------------------
# Detection grid
# --------------------------------------------------------------
ds_size_test = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

all_periods = [
    10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    150, 200, 250, 300, 350, 400, 450, 500, 550
]

# Split periods into 19 chunks
n_splits = 19
chunk_size = int(np.ceil(len(all_periods) / n_splits))
start_idx = (nrun - 1) * chunk_size
end_idx = start_idx + chunk_size
period_test = all_periods[start_idx:end_idx]

logger.info(f"RUN {nrun}: Testing periods = {period_test}")

# --------------------------------------------------------------
# Load time axis
# --------------------------------------------------------------
time_df = pd.read_csv(DATA_DIR / "time_df.csv")
dates = time_df["jdb"].values
n_samples = len(dates)

# --------------------------------------------------------------
# Output DataFrames
# --------------------------------------------------------------
df_detections      = pd.DataFrame(0.0, index=period_test, columns=ds_size_test)
df_amplitudes      = pd.DataFrame(0.0, index=period_test, columns=ds_size_test)
df_amplitudes_perc = pd.DataFrame(0.0, index=period_test, columns=ds_size_test)
df_phases          = pd.DataFrame(0.0, index=period_test, columns=ds_size_test)
df_periods         = pd.DataFrame(0.0, index=period_test, columns=ds_size_test)
df_det_count       = pd.DataFrame(0.0, index=period_test, columns=ds_size_test)
df_detections_highest_peak = pd.DataFrame(0.0, index=period_test, columns=ds_size_test)

df_variance_rv       = pd.DataFrame(0.0, index=period_test, columns=ds_size_test)
df_variance_ds       = pd.DataFrame(0.0, index=period_test, columns=ds_size_test)

df_residuals_rv      = pd.DataFrame(0.0, index=period_test, columns=ds_size_test)
df_residuals_ds      = pd.DataFrame(0.0, index=period_test, columns=ds_size_test)

df_med_residuals_rv  = pd.DataFrame(0.0, index=period_test, columns=ds_size_test)
df_med_residuals_ds  = pd.DataFrame(0.0, index=period_test, columns=ds_size_test)

# --------------------------------------------------------------
# Load fold test indices (needed for OOF stitching)
# --------------------------------------------------------------
fold_test_indices = []
for fold_id in range(num_folds):
    idx_path = LOCAL_OUTPUTS_DIR / f"{prefix_name}_fold{fold_id}_test_idx.txt"
    idx = np.loadtxt(idx_path, dtype=int)
    fold_test_indices.append(idx)
    logger.info(f"Loaded fold {fold_id} test idx (len={len(idx)})")

# --------------------------------------------------------------
# Load CV fold scalers + models ONCE
# --------------------------------------------------------------
logger.info("Loading CV scalers + models...")

fold_scalers = []
fold_models = []

for fold_id in range(num_folds):
    sx_path = LOCAL_MODELS_DIR / f"{prefix_name}_fold{fold_id}_scalerx.pkl"
    sy_path = LOCAL_MODELS_DIR / f"{prefix_name}_fold{fold_id}_scalery.pkl"
    m_path  = LOCAL_MODELS_DIR / f"{prefix_name}_fold{fold_id}.h5"

    with open(sx_path, "rb") as f:
        scalerx = pickle.load(f)
    with open(sy_path, "rb") as f:
        scalery = pickle.load(f)

    dummy_wrapper = ShellCNN1D(
        input_shape=(1, 1),
        n_outputs=2,
        conv_layers=None,
        dense_layers=None,
        dropout=None,
        actfn=None,
    )
    cnn = dummy_wrapper.load_model(m_path)

    fold_scalers.append((scalerx, scalery))
    fold_models.append(cnn)

logger.info("All CV models + scalers loaded successfully.")

# --------------------------------------------------------------
# Helper: OOF prediction for ONE shell directory (single phase)
# --------------------------------------------------------------
def predict_oof_for_shell(shell_dir: Path, ds_i: float, period_j: float, mc_dropout_num: int = 100):
    """
    Returns OOF predictions (mean + std) for one shell directory:
        raw_rv, raw_ds, inj_phase_rad,
        pred_rv_oof_mean, pred_ds_oof_mean,
        pred_rv_oof_std,  pred_ds_oof_std
    All arrays are length n_samples.
    """
    params = dict(
        pis=[ds_i],
        periods=[period_j],
        use_temp=shell_type_temp,
        use_mask=use_mask,
        use_residuals=use_residuals,
        data_dir=shell_dir,
        selected_idx=np.arange(n_samples),
    )

    loader = [load_shell_astro_datah5(spec_type=s, **params) for s in spec_types]
    shell_list, astro_list, _, _, _ = zip(*loader)

    shell_x = np.concatenate(shell_list, axis=0)   # (n_samples, ...)
    astro_y = np.concatenate(astro_list, axis=0)   # (n_samples, ...)

    raw_rv = astro_y[:, 0]
    raw_ds = astro_y[:, -2]
    inj_phase_rad = astro_y[:, -1]

    # OOF stitched mean + std
    pred_rv_mean = np.full(n_samples, np.nan)
    pred_ds_mean = np.full(n_samples, np.nan)
    pred_rv_std  = np.full(n_samples, np.nan)
    pred_ds_std  = np.full(n_samples, np.nan)

    # dummy wrapper only for calling mcdo_predict (same pattern you used before)
    mcdo_wrapper = ShellCNN1D(
        input_shape=(1, 1),
        n_outputs=2,
        conv_layers=None,
        dense_layers=None,
        dropout=None,
        actfn=None,
    )

    for fold_id, ((scalerx, scalery), cnn) in enumerate(zip(fold_scalers, fold_models)):
        test_idx = fold_test_indices[fold_id]

        x_test = shell_x[test_idx]
        x_test_scaled = scalerx.transform(x_test)

        # ---- MC dropout prediction on this fold's test subset ----
        pred_mcdo = mcdo_wrapper.mcdo_predict(x_test_scaled, cnn, mc_dropout_num=mc_dropout_num)
        mean_scaled = pred_mcdo["mean"]  # (n_test, 2) in scaled y-space
        std_scaled  = pred_mcdo["std"]   # (n_test, 2) in scaled y-space

        mean_phys = scalery.inverse_transform(mean_scaled)

        # std: convert from scaled-y to physical units.
        # For StandardScaler: y_phys = mean + std_scaled * scalery.scale_
        std_phys = std_scaled * scalery.scale_

        pred_rv_mean[test_idx] = mean_phys[:, 0]
        pred_ds_mean[test_idx] = mean_phys[:, 1]
        pred_rv_std[test_idx]  = std_phys[:, 0]
        pred_ds_std[test_idx]  = std_phys[:, 1]

    # Safety checks
    for arr, name in [
        (pred_ds_mean, "pred_ds_mean"), (pred_ds_std, "pred_ds_std"),
        (pred_rv_mean, "pred_rv_mean"), (pred_rv_std, "pred_rv_std"),
    ]:
        if np.any(np.isnan(arr)):
            missing = np.where(np.isnan(arr))[0]
            raise RuntimeError(
                f"OOF stitching left {len(missing)} NaNs in {name}. "
                f"Your fold indices likely do not cover all samples."
            )

    return raw_rv, raw_ds, inj_phase_rad, pred_rv_mean, pred_ds_mean, pred_rv_std, pred_ds_std


# --------------------------------------------------------------
# Main loop
# --------------------------------------------------------------
fap = 0.001
min_period = 5
max_period = 2000

# IMPORTANT: your “n_evaluations” is the number of independent datasets.
# Here: the 10 shell directories are those datasets (different phases).
n_evaluations = n_shells

for shell_id in range(n_shells):
    logger.info(f"\n=== EVALUATION shell {shell_id}/{n_shells-1} ===")
    shell_dir = base_shells_dir / f"{shell_id}"

    for ds_i in ds_size_test:
        for period_j in period_test:

            logger.info(f"Shell {shell_id} | ds={ds_i} | P={period_j}")

            raw_rv, raw_ds, inj_phase_rad, pred_rv, pred_ds, pred_rv_std, pred_ds_std = predict_oof_for_shell(
                shell_dir=shell_dir,
                ds_i=ds_i,
                period_j=period_j, mc_dropout_num=100
            )

            # Residuals
            # res_rv = np.abs(raw_rv - pred_rv)
            # res_ds = np.abs(raw_ds - pred_ds)
            err_rv = raw_rv - pred_rv
            err_ds = raw_ds - pred_ds

            rmse_rv = np.sqrt(np.mean(err_rv**2))
            rmse_ds = np.sqrt(np.mean(err_ds**2))

            # ALWAYS accumulate these (regardless of detection success)
            df_variance_rv.loc[period_j, ds_i]      += np.median(pred_rv_std)
            df_variance_ds.loc[period_j, ds_i]      += np.median(pred_ds_std)
            df_med_residuals_rv.loc[period_j, ds_i] += np.median(err_rv)
            df_med_residuals_ds.loc[period_j, ds_i] += np.median(err_ds)
            df_residuals_rv.loc[period_j, ds_i]     += rmse_rv
            df_residuals_ds.loc[period_j, ds_i]     += rmse_ds

            # ---- Periodogram on predicted DS ----
            clp_ds_pred, plevels = periodogram(
                rvs=pred_ds,
                time=dates,
                err=None,
                fap=fap,
                min_period=min_period,
                max_period=max_period,
            )

            power_limit = plevels
            freqs = clp_ds_pred.freq
            power = clp_ds_pred.power

            # ±5% frequency window around target
            freq_min = 1.0 / (period_j * 1.05)
            freq_max = 1.0 / (period_j * 0.95)

            wmask = (freqs >= freq_min) & (freqs <= freq_max)
            if not np.any(wmask):
                continue

            freqs_win = freqs[wmask]
            power_win = power[wmask]

            above = power_win >= power_limit
            if not np.any(above):
                continue

            # Best peak among those above threshold
            idx_best = np.argmax(power_win * above)
            detected_freq = freqs_win[idx_best]
            detected_period = 1.0 / detected_freq
            detected_power = power_win[idx_best]
            
            # Check if detected peak is the highest peak above threshold in the full periodogram
            all_powers_above_threshold = power[power >= power_limit]
            is_highest_peak = np.isclose(detected_power, np.max(all_powers_above_threshold))
            if is_highest_peak:
                df_detections_highest_peak.loc[period_j, ds_i] += 1

            # ---- sine fit at detected period (keep your logic) ----
            omega_det = 2 * np.pi / detected_period

            def sine_model_detected(t, A, phi, offset):
                return A * np.sin(omega_det * t + phi) + offset

            t_norm = dates - dates[0]
            popt, _ = curve_fit(
                sine_model_detected, t_norm, pred_ds,
                p0=[ds_i, 0.0, 0.0]
            )
            A_fit, phi_fit, offset_fit = popt
            A_detected = abs(A_fit)

            A_diff = abs(ds_i - A_detected)
            A_perc = 100 * np.abs(ds_i - A_detected) / ds_i

            detected_phase_cycles = (phi_fit / (2 * np.pi)) % 1.0

            # Truth phase offset (cycles), recovered from stored injected phases (radians)
            inj_phase_offset_cycles = recover_phase_offset(
                dates,
                inj_phase_rad,          # radians
                period_days=period_j,   # injected period (truth)
                reference_date=dates[0]
            )

            phase_diff = circ_dist_cycles(detected_phase_cycles, inj_phase_offset_cycles)

            period_diff = abs(detected_period - period_j)

            # Store results (one “success” for this shell dataset)
            df_detections.loc[period_j, ds_i]      += 1
            df_det_count.loc[period_j, ds_i]       += 1
            df_amplitudes.loc[period_j, ds_i]      += A_diff
            df_amplitudes_perc.loc[period_j, ds_i] += A_perc
            df_phases.loc[period_j, ds_i]          += phase_diff
            df_periods.loc[period_j, ds_i]         += period_diff


# --------------------------------------------------------------
# Averages
# --------------------------------------------------------------
with np.errstate(invalid="ignore", divide="ignore"):
    df_amplitudes      = (df_amplitudes      / df_det_count).fillna(0.0)
    df_amplitudes_perc = (df_amplitudes_perc / df_det_count).clip(upper=100).fillna(0.0)
    df_phases          = (df_phases          / df_det_count).fillna(0.0)
    df_periods         = (df_periods         / df_det_count).fillna(0.0)

df_detections = df_detections / n_evaluations
df_detect_bin = (df_detections >= 0.7).astype(int)

df_detections_highest_peak = df_detections_highest_peak / n_evaluations
df_detect_highest_peak_bin = (df_detections_highest_peak >= 0.7).astype(int)

df_variance_rv       = df_variance_rv / n_evaluations
df_variance_ds       = df_variance_ds / n_evaluations
df_residuals_rv      = df_residuals_rv / n_evaluations
df_residuals_ds      = df_residuals_ds / n_evaluations
df_med_residuals_rv  = df_med_residuals_rv / n_evaluations
df_med_residuals_ds  = df_med_residuals_ds / n_evaluations


# --------------------------------------------------------------
# Save results
# --------------------------------------------------------------
df_detections.to_csv(LOCAL_OUTPUTS_DIR / f"detections_{prefix_name}_chunk{nrun}.csv")
df_detect_bin.to_csv(LOCAL_OUTPUTS_DIR / f"detections_binary_{prefix_name}_chunk{nrun}.csv")
df_detections_highest_peak.to_csv(LOCAL_OUTPUTS_DIR / f"detections_highest_peak_{prefix_name}_chunk{nrun}.csv")
df_amplitudes.to_csv(LOCAL_OUTPUTS_DIR / f"amplitudes_{prefix_name}_chunk{nrun}.csv")
df_amplitudes_perc.to_csv(LOCAL_OUTPUTS_DIR / f"amplitudes_perc_{prefix_name}_chunk{nrun}.csv")
df_phases.to_csv(LOCAL_OUTPUTS_DIR / f"phases_{prefix_name}_chunk{nrun}.csv")
df_periods.to_csv(LOCAL_OUTPUTS_DIR / f"periods_{prefix_name}_chunk{nrun}.csv")

df_variance_rv.to_csv(LOCAL_OUTPUTS_DIR / f"variance_rv_{prefix_name}_chunk{nrun}.csv")
df_variance_ds.to_csv(LOCAL_OUTPUTS_DIR / f"variance_ds_{prefix_name}_chunk{nrun}.csv")
df_residuals_rv.to_csv(LOCAL_OUTPUTS_DIR / f"residuals_rv_{prefix_name}_chunk{nrun}.csv")
df_residuals_ds.to_csv(LOCAL_OUTPUTS_DIR / f"residuals_ds_{prefix_name}_chunk{nrun}.csv")
df_med_residuals_rv.to_csv(LOCAL_OUTPUTS_DIR / f"residuals_med_rv_{prefix_name}_chunk{nrun}.csv")
df_med_residuals_ds.to_csv(LOCAL_OUTPUTS_DIR / f"residuals_med_ds_{prefix_name}_chunk{nrun}.csv")


logger.info("Done.")
