import sys
import os
import random
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# --- DopplerIANN imports ---
from doppleriann.networks import ShellCNN1D
from doppleriann.data import (
    MaskedStandardScaler3D,
    load_shell_astro_datah5,
)
from doppleriann.physics import (
    generate_periodogram_test,
    recover_phase_offset,
)
from doppleriann.utils.logger_config import logger

# ============================================================
#  Project paths & basic config
# ============================================================

# Get nrun from command line or default to 0 (dataset index)
nrun = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Current script directory / project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
LOCAL_MODELS_DIR = SCRIPT_DIR / "models/models"
LOCAL_OUTPUTS_DIR = SCRIPT_DIR / "outputs"

LOCAL_MODELS_DIR.mkdir(exist_ok=True)
LOCAL_OUTPUTS_DIR.mkdir(exist_ok=True)

logger.info(f"[INFO] Data directory:      {DATA_DIR}")
logger.info(f"[INFO] Local models dir:   {LOCAL_MODELS_DIR}")
logger.info(f"[INFO] Local outputs dir:  {LOCAL_OUTPUTS_DIR}")

# Reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ============================================================
#  High-level experiment configuration
# ============================================================

shell_type_temp = True      # True for temp shells, False for flux shells
use_residuals = True
use_density_shell_mask = True
show_pred_plots = False

n_reso = 9                  # 9 or 15
# Dataset index: we follow HO style and train on dataset idx=0
# shells_dir = DATA_DIR / f"shells{n_reso}" / f"{nrun}"
shells_dir = DATA_DIR / f"shells" / f"{nrun}"

# Training injections (same as HO)
planetary_injections = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
periods_train = [20, 40, 60, 80, 100]

# Test grid (ds, P)
ds_size_test = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
period_test = [
    10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    150, 200, 250, 300, 350, 400, 450, 500, 550
]

shell_type_str = "temp" if shell_type_temp else "flux"
spec_types = ["act"]  # 'act', 'or', etc.
str_spec_types = "_".join(spec_types)

prefix_name = f"cnnshellCV5_{n_reso}_{shell_type_str}_{str_spec_types}"
prefix_name += "_mask" if use_density_shell_mask else ""
prefix_name += "_res" if use_residuals else ""

# CNN hyperparameters (aligned with HO code)
actfn = "selu"
loss_fn = "mean_squared_error"
epochs = 1000
patience = 40
dropout_rate = 0.2

if shell_type_temp:
    bs = 128
    conv_layers = [(256, 5), (512, 5)]
    dense_layers = [512]
    learning_rate = 0.0002
else:
    bs = 256
    learning_rate = 0.0002
    conv_layers =  [(128, 3), (256, 3)]
    dense_layers =  [512]

num_folds = 5

# ============================================================
#  Utilities
# ============================================================

def make_callbacks():
    """Build fresh callbacks per fold (EarlyStopping + ReduceLROnPlateau)."""
    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=1e-5,
        restore_best_weights=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=patience // 2,
        min_delta=1e-5,
        min_lr=1e-6,
    )
    return [es, reduce_lr]


def build_folds(n_samples=2036):
    """
    Build 5 folds with sizes [408, 408, 408, 406, 406],
    shuffle indices once, then split.
    Save train/test indices for each fold to .txt.
    """
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Fixed distribution that sums to 2036
    fold_sizes = [408, 408, 408, 406, 406]
    assert sum(fold_sizes) == n_samples

    folds = []
    start = 0
    for size in fold_sizes:
        end = start + size
        folds.append(indices[start:end])
        start = end

    logger.info(f"Built {len(folds)} folds with sizes: {[len(f) for f in folds]}")

    # Save just in case
    for i, fold_idx in enumerate(folds):
        np.savetxt(
            LOCAL_OUTPUTS_DIR / f"{prefix_name}_fold{i}_test_idx.txt",
            fold_idx,
            fmt="%d",
        )

    return folds


def load_train_data(train_indices):
    """
    Load TRAIN data for given indices using the same dataset / injections
    as the HO setup.
    """
    train_params = dict(
        pis=planetary_injections,
        periods=periods_train,
        use_temp=shell_type_temp,
        use_mask=use_density_shell_mask,
        use_residuals=use_residuals,
        data_dir=shells_dir,
        selected_idx=train_indices,
    )

    data_loader = [
        load_shell_astro_datah5(spec_type=st, **train_params) for st in spec_types
    ]
    shell_train_list, astro_train_list, _, _, _ = zip(*data_loader)

    shell_data_x_train = np.concatenate(shell_train_list, axis=0)
    astrodata_train = np.concatenate(astro_train_list, axis=0)

    logger.info(f"Train shell shape: {shell_data_x_train.shape}")
    logger.info(f"Train astrodata shape: {astrodata_train.shape}")

    return shell_data_x_train, astrodata_train


def load_test_shells_for_combination(ds_i, period_j, test_indices):
    """
    Load TEST shells for a given (ds_i, period_j) and a specific set of indices.
    Used during per-fold prediction.
    """
    test_params = dict(
        pis=[ds_i],
        periods=[period_j],
        use_temp=shell_type_temp,
        use_mask=use_density_shell_mask,
        use_residuals=use_residuals,
        data_dir=shells_dir,
        selected_idx=test_indices,
    )
    data_loader_test = [
        load_shell_astro_datah5(spec_type=st, **test_params) for st in spec_types
    ]
    shell_test_list, _, _, _, _ = zip(*data_loader_test)
    shell_data_x_test = np.concatenate(shell_test_list, axis=0)
    return shell_data_x_test


def load_full_astro_for_combination(ds_i, period_j):
    """
    Load FULL astrodata (all 2036 samples) for evaluation
    for a given (ds_i, period_j).
    """
    data_params = dict(
        pis=[ds_i],
        periods=[period_j],
        use_temp=shell_type_temp,
        use_mask=use_density_shell_mask,
        use_residuals=use_residuals,
        data_dir=shells_dir,
    )
    data_loader = [
        load_shell_astro_datah5(spec_type=st, **data_params) for st in spec_types
    ]
    _, astrodata_list, _, _, _ = zip(*data_loader)
    astrodata = np.concatenate(astrodata_list, axis=0)
    return astrodata


def build_model(input_shape):
    """Instantiate a fresh ShellCNN1D (tf model) for each fold."""
    model_wrapper = ShellCNN1D(
        input_shape=input_shape,
        n_outputs=2,
        conv_layers=conv_layers,
        dense_layers=dense_layers,
        dropout=dropout_rate,
        actfn=actfn,
        mcdropout=False,
    )
    cnn = model_wrapper.model_tf()
    optimizer = Adam(learning_rate=learning_rate)
    cnn.compile(optimizer=optimizer, loss=loss_fn, metrics=["mean_absolute_error"])
    return model_wrapper, cnn


# ============================================================
#  Main CV training & prediction
# ============================================================

def run_cross_validation():
    # Time axis (aligned with 2036 spectra)
    time_df = pd.read_csv(DATA_DIR / "time_df.csv")
    # Use 'jdb' to be consistent with HO
    dates = time_df["jdb"].values

    combos = list(product(ds_size_test, period_test))
    num_comb = len(combos)

    # (num_combinations, 2036, 2) -> [RV, DS]
    all_predictions_ordered = np.zeros((num_comb, 2036, 2))

    all_train_losses = []
    all_val_losses = []

    folds = build_folds(n_samples=len(dates))

    for cv_idx, test_indices in enumerate(folds):
        logger.info(f"\n===== Fold {cv_idx+1}/{num_folds} =====")
        tf.keras.backend.clear_session()

        # Train indices = all except this fold's test indices
        test_set_mask = np.zeros(len(dates), dtype=bool)
        test_set_mask[test_indices] = True
        train_indices = np.where(~test_set_mask)[0]

        logger.info(f"Fold {cv_idx+1}: train={len(train_indices)}, test={len(test_indices)}")

        # Save training indices for this fold
        np.savetxt(
            LOCAL_OUTPUTS_DIR / f"{prefix_name}_fold{cv_idx}_train_idx.txt",
            train_indices,
            fmt="%d",
        )

        # --------- LOAD TRAIN DATA ----------
        shell_data_x_train, astrodata_train = load_train_data(train_indices)

        # --------- SCALE DATA ----------
        scalerx = MaskedStandardScaler3D()
        scalerx.fit(shell_data_x_train)
        x_train = scalerx.transform(shell_data_x_train)

        y_train = astrodata_train[:, [0, -2]]  # RV (0) and DS (-2)
        scalery = StandardScaler()
        scalery.fit(y_train)
        y_train_scaled = scalery.transform(y_train)

        # Save scalers for this fold
        with open(LOCAL_MODELS_DIR / f"{prefix_name}_fold{cv_idx}_scalerx.pkl", "wb") as f:
            pickle.dump(scalerx, f)
        with open(LOCAL_MODELS_DIR / f"{prefix_name}_fold{cv_idx}_scalery.pkl", "wb") as f:
            pickle.dump(scalery, f)

        # --------- BUILD & TRAIN MODEL ----------
        model_wrapper, cnn = build_model(
            input_shape=(x_train.shape[1], x_train.shape[2])
        )
        cnn.summary(print_fn=logger.info)

        callbacks = make_callbacks()

        history = cnn.fit(
            x_train,
            y_train_scaled,
            epochs=epochs,
            batch_size=bs,
            callbacks=callbacks,
            shuffle=True,
            validation_split=0.2,
            verbose=1,
        )

        all_train_losses.append(history.history["loss"])
        all_val_losses.append(history.history["val_loss"])

        # Save model
        model_path = LOCAL_MODELS_DIR / f"{prefix_name}_fold{cv_idx}.h5"
        cnn.save(model_path)
        logger.info(f"Saved fold {cv_idx} model to {model_path}")

        # --------- PREDICT ON THIS FOLD'S TEST INDICES FOR ALL COMBOS ----------
        for comb_idx, (ds_i, period_j) in enumerate(combos):
            shell_test = load_test_shells_for_combination(ds_i, period_j, test_indices)
            x_test = scalerx.transform(shell_test)

            pred_scaled = model_wrapper.mcdo_predict(
                x_test, cnn, mc_dropout_num=50
            )["mean"]

            pred_phys = scalery.inverse_transform(pred_scaled)  # (n_test, 2)

            # Fill in the positions corresponding to this fold's test indices
            all_predictions_ordered[comb_idx, test_indices, :] = pred_phys

            if cv_idx == 0 and comb_idx < 2:
                logger.debug(
                    f"Fold {cv_idx}, comb_idx={comb_idx}, ds={ds_i}, P={period_j}, "
                    f"pred shape={pred_phys.shape}"
                )

        logger.info(f"Fold {cv_idx+1} completed.")

    logger.info(f"all_predictions_ordered final shape: {all_predictions_ordered.shape}")

    # --------- PLOT MEAN TRAIN/VAL LOSSES ACROSS FOLDS ----------
    # Pad histories to same length (epochs) for averaging
    padded_train = np.array(
        [np.pad(l, (0, epochs - len(l)), mode="edge") for l in all_train_losses]
    )
    padded_val = np.array(
        [np.pad(l, (0, epochs - len(l)), mode="edge") for l in all_val_losses]
    )
    mean_train_loss = padded_train.mean(axis=0)
    mean_val_loss = padded_val.mean(axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_train_loss, label="Mean Training Loss")
    plt.plot(mean_val_loss, label="Mean Validation Loss")
    plt.title(f"Average Loss over {num_folds} CV folds")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOCAL_OUTPUTS_DIR / f"{prefix_name}_mean_cv_loss_nrun{nrun}.png")
    plt.close()

    # ============================================================
    #  Evaluation over the full ordered time series
    # ============================================================
    logger.info("Evaluating predictions over full 2036-sample series...")

    df_detections = pd.DataFrame(
        data=np.zeros((len(period_test), len(ds_size_test))),
        index=period_test,
        columns=ds_size_test,
    )

    df_amplitudes = pd.DataFrame(
        data=np.zeros((len(period_test), len(ds_size_test))),
        index=period_test,
        columns=ds_size_test,
    )

    df_amplitudes_perc = pd.DataFrame(
        data=np.zeros((len(period_test), len(ds_size_test))),
        index=period_test,
        columns=ds_size_test,
    )

    df_phases = pd.DataFrame(
        data=np.zeros((len(period_test), len(ds_size_test))),
        index=period_test,
        columns=ds_size_test,
    )

    df_periods = pd.DataFrame(
        data=np.zeros((len(period_test), len(ds_size_test))),
        index=period_test,
        columns=ds_size_test,
    )

    df_detections_count = pd.DataFrame(
        data=np.zeros((len(period_test), len(ds_size_test))),
        index=period_test,
        columns=ds_size_test,
    )

    fap = 0.001
    comb_idx = 0

    for i, ds_i in enumerate(ds_size_test):
        for j, period_j in enumerate(period_test):
            logger.info(f"Evaluating ds={ds_i}, P={period_j} (comb_idx={comb_idx})")

            # Predictions (full ordered time series)
            pred2 = all_predictions_ordered[comb_idx]
            pred2_rv = pred2[:, 0]
            pred2_ds = pred2[:, 1]

            # True RV/DS from full dataset
            astrodata = load_full_astro_for_combination(ds_i, period_j)

            # generate_periodogram_test signature updated like HO
            periodogram_output = generate_periodogram_test(
                real_rv=astrodata[:, 0],
                pred_rv=pred2_rv,
                pred_ds=pred2_ds,
                dates=dates,
                ds_size=ds_i,
                period=period_j,
                fap=fap,
                shell_type_str="CV5",
                min_period=5,
                max_period=1000,
                plot=True,
                savefig=False,
            )
            plt.show(block=False)
            plt.close()

            clp_ds_pred = periodogram_output["clp_ds_pred"]
            power_limit = clp_ds_pred.powerLevel(fap)

            # ±5% window around target frequency
            freq_min = 1.0 / (period_j * 1.05)
            freq_max = 1.0 / (period_j * 0.95)
            freq_window_mask = (clp_ds_pred.freq >= freq_min) & (clp_ds_pred.freq <= freq_max)

            freqs_in_window = clp_ds_pred.freq[freq_window_mask]
            powers_in_window = clp_ds_pred.power[freq_window_mask]

            above_thresh_mask = powers_in_window >= power_limit

            if np.any(above_thresh_mask):
                # strongest peak above threshold
                idx_best = np.argmax(powers_in_window * above_thresh_mask)
                detected_freq = freqs_in_window[idx_best]
                detected_period = 1.0 / detected_freq

                logger.info(
                    f"Detected signal near {period_j} d (FAP<{fap}) at {detected_period:.2f} d"
                )

                # Fit sine at detected period
                omega_det = 2 * np.pi / detected_period

                def sine_model_detected(t, A, phi, offset):
                    return A * np.sin(omega_det * t + phi) + offset

                t_norm = dates - dates[0]
                y = pred2_ds

                from scipy.optimize import curve_fit

                popt, _ = curve_fit(
                    sine_model_detected, t_norm, y, p0=[ds_i, 0, 0]
                )
                amplitude_raw, phi_fit, offset_fit = popt
                amplitude_detected = np.abs(amplitude_raw)

                amplitude_diff = np.abs(amplitude_detected - ds_i)
                amplitude_perc = 100.0 * amplitude_diff / ds_i

                detected_phase_offset = (phi_fit / (2 * np.pi)) % 1
                recovered_phase = recover_phase_offset(
                    dates, astrodata[:, -1], period_days=detected_period
                )
                phase_diff = np.abs(detected_phase_offset - recovered_phase)

                period_diff = np.abs(detected_period - period_j)

                # Aggregate
                df_detections.loc[period_j, ds_i] += 1
                df_detections_count.loc[period_j, ds_i] += 1
                df_amplitudes.loc[period_j, ds_i] += amplitude_diff
                df_amplitudes_perc.loc[period_j, ds_i] += amplitude_perc
                df_phases.loc[period_j, ds_i] += phase_diff
                df_periods.loc[period_j, ds_i] += period_diff
            else:
                logger.info(
                    f"No significant signal near {period_j} d (FAP<{fap})"
                )

            comb_idx += 1

    # Optional: normalize amplitude / phase / period by detection count
    with np.errstate(invalid="ignore", divide="ignore"):
        df_amplitudes = (df_amplitudes / df_detections_count).fillna(0.0)
        df_amplitudes_perc = (df_amplitudes_perc / df_detections_count).fillna(0.0)
        df_amplitudes_perc = df_amplitudes_perc.clip(upper=100.0)
        df_phases = (df_phases / df_detections_count).fillna(0.0)
        df_periods = (df_periods / df_detections_count).fillna(0.0)

    # Save CSV outputs
    df_detections_count.to_csv(
        LOCAL_OUTPUTS_DIR / f"detections_count_{prefix_name}_nrun{nrun}.csv"
    )
    df_detections.to_csv(
        LOCAL_OUTPUTS_DIR / f"detections_{prefix_name}_nrun{nrun}.csv"
    )
    df_phases.to_csv(
        LOCAL_OUTPUTS_DIR / f"phases_{prefix_name}_nrun{nrun}.csv"
    )
    df_amplitudes.to_csv(
        LOCAL_OUTPUTS_DIR / f"amplitudes_{prefix_name}_nrun{nrun}.csv"
    )
    df_amplitudes_perc.to_csv(
        LOCAL_OUTPUTS_DIR / f"amplitudes_perc_{prefix_name}_nrun{nrun}.csv"
    )
    df_periods.to_csv(
        LOCAL_OUTPUTS_DIR / f"periods_{prefix_name}_nrun{nrun}.csv"
    )

    logger.info("5-fold CV + evaluation finished.")


if __name__ == "__main__":
    run_cross_validation()
