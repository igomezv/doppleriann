import sys

# Get nrun from command line or environment variable
nrun = int(sys.argv[1]) if len(sys.argv) > 1 else 1

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD, AdamW

# --- DopplerIANN imports ---
from doppleriann.networks import ShellCNN1D
from doppleriann.data import (
    MaskedStandardScaler3D,
    load_shell_astro_datah5,
)
from doppleriann.physics import (
    generate_periodogram_test,
    recover_phase_offset,
    long_term_remover,
    remove_strongest_signals,
)
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
LOCAL_MODELS_DIR.mkdir(exist_ok=True)
logger.info(f"[INFO] Data directory: {DATA_DIR}")
logger.info(f"[INFO] Local models directory: {LOCAL_MODELS_DIR}")
logger.info(f"[INFO] Local outputs directory: {LOCAL_OUTPUTS_DIR}")


# Get nrun from command line or environment variable
nrun = int(sys.argv[1]) if len(sys.argv) > 1 else 1
# --- Recover phase at target period directly by fitting a sine ---


def sine_model(t, A, phi, offset):
    omega = 2 * np.pi / period_j
    return A * np.sin(omega * t + phi) + offset


shell_type_temp = True  # True for temp, False for Flux shells
use_residuals = True
use_density_shell_mask = True
train_model = False  # Set to True to train the model, False to load and predict
# INPUT: shell, output regression of RV and DS, with CNN
show_pred_plots = False
hpc_device = True
n_reso = 9
large_datadir = LARGE_DATA_DIR
shells_dir = f"{DATA_DIR}/shells{n_reso}/0/"

# Test settings
ds_size_test = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
all_periods = [
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
]

n_splits = 19  # total jobs/splits
chunk_size = int(np.ceil(len(all_periods) / n_splits))
start = (nrun - 1) * chunk_size
end = start + chunk_size
period_test = all_periods[start:end]

# Create DataFrame of zeros
df_detections = pd.DataFrame(
    data=np.zeros((len(period_test), len(ds_size_test))), index=period_test, columns=ds_size_test
)

df_amplitudes = pd.DataFrame(
    data=np.zeros((len(period_test), len(ds_size_test))), index=period_test, columns=ds_size_test
)

df_amplitudes_perc = pd.DataFrame(
    data=np.zeros((len(period_test), len(ds_size_test))), index=period_test, columns=ds_size_test
)

df_phases = pd.DataFrame(
    data=np.zeros((len(period_test), len(ds_size_test))), index=period_test, columns=ds_size_test
)

df_periods = pd.DataFrame(
    data=np.zeros((len(period_test), len(ds_size_test))), index=period_test, columns=ds_size_test
)

df_detections_count = pd.DataFrame(
    data=np.zeros((len(period_test), len(ds_size_test))), index=period_test, columns=ds_size_test
)

df_detections_highest_peak = pd.DataFrame(
    data=np.zeros((len(period_test), len(ds_size_test))), index=period_test, columns=ds_size_test
)


## Training set settings
# planetary_injection =  [0.2]
# period = [100]
shell_type_str = "temp" if shell_type_temp else "flux"
# spec_types: 'act', 'or'
spec_types = ["act"]
str_spec_types = "_".join(spec_types)


# Generate and shuffle indices
indices = list(range(2036))
random.shuffle(indices)

# Define fold sizes: 6 folds with 204, 4 folds with 203
fold_sizes = [204] * 6 + [203] * 4

# To distribute evenly, shuffle the fold sizes too
random.shuffle(fold_sizes)

# Build the folds
folds = []
start = 0
for size in fold_sizes:
    end = start + size
    folds.append(indices[start:end])
    start = end

num_cv = 10


def randomized_cv(num_cv=10, train=True, x=None, y=None, prefix_name="cnnShellCV"):
    all_train_losses = []
    all_val_losses = []

    # Prepare an array to hold predictions in the original order
    all_predictions_ordered = np.zeros((2036, 2))  # Assuming y is (2036, n_outputs)

    for i in range(num_cv):
        test_indices = folds[i]
        train_indices = [idx for j, fold in enumerate(folds) if j != i for idx in fold]
        print(f"Fold {i + 1}:")
        print(f"  Test indices length: {len(test_indices)}")
        print(f"  Train indices length: {len(train_indices)}\n")

        x_train = x[train_indices]
        x_test = x[test_indices]

        # CNN settings
        actfn = "selu"
        loss_fn = "mean_squared_error"
        epochs = 200
        patience = 40
        es = EarlyStopping(
            monitor="val_loss", patience=patience, min_delta=1e-5, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=patience // 2, min_delta=1e-5, min_lr=1e-6
        )
        callbacks = [es, reduce_lr]
        dropout_rate = 0.2
        # Optuna results for batch size, learning rate, and conv and dense layers.

        if shell_type_temp:
            bs = 128
            conv_layers = [(256, 5), (512, 5)]
            dense_layers = [512]
            learning_rate = 0.0002
        else:
            bs = 256
            learning_rate = 0.002
            conv_layers = [(128, 3), (256, 3)]
            dense_layers = [512]
            optimizer = Adam(learning_rate=learning_rate)

        optimizer = Adam(learning_rate=learning_rate)

        model = ShellCNN1D(
            input_shape=(x.shape[1], x.shape[2]),
            n_outputs=2,
            conv_layers=conv_layers,
            dense_layers=dense_layers,
            dropout=dropout_rate,
            actfn=actfn,
        )

        if train:
            y_train = y[train_indices]
            y_test = y[test_indices]

            cnn = model.model_tf()
            cnn.compile(optimizer=optimizer, loss=loss_fn, metrics=["mean_absolute_error"])
            cnn.summary()
            history = cnn.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=bs,
                callbacks=callbacks,
                shuffle=False,
                validation_data=(x_test, y_test),
            )
            all_train_losses.append(history.history["loss"])
            all_val_losses.append(history.history["val_loss"])

            cnn.save(f"{LOCAL_MODELS_DIR}/{prefix_name}_cv_{i}.h5")
        else:
            cnn = model.load_model(f"{LOCAL_MODELS_DIR}/{prefix_name}_cv_{i}.h5")
            cnn.summary()

        pred_mcdo = model.mcdo_predict(x_test, cnn)
        pred = pred_mcdo["mean"]

        # Store predictions in their correct place in the ordered array
        for idx, p in zip(test_indices, pred):
            all_predictions_ordered[idx] = p

    if train:
        mean_train_loss = np.mean(
            np.array([np.pad(l, (0, epochs - len(l)), "edge") for l in all_train_losses]), axis=0
        )
        mean_val_loss = np.mean(
            np.array([np.pad(l, (0, epochs - len(l)), "edge") for l in all_val_losses]), axis=0
        )
        # plt.figure(figsize=(10, 6))
        # plt.plot(mean_train_loss, label='Mean Training Loss')
        # plt.plot(mean_val_loss, label='Mean Validation Loss')
        # plt.title(f'Average Loss over {num_cv} Random Splits')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.savefig(f"img/{prefix_name}_mean_cv_loss.png")
        # plt.show()

    return cnn, model, all_predictions_ordered


# Number of evaluations
n_evaluations = 10
for idx_ev in range(n_evaluations):
    shells_dir = f"{DATA_DIR}/shells{n_reso}/{idx_ev + 1}/"
    for i, ds_i in enumerate(ds_size_test):
        for j, period_j in enumerate(period_test):
            logger.info(f"Evaluation {i + 1}/{n_evaluations}, ds_size: {ds_i}, period: {period_j}")
            # Train
            if shell_type_temp:
                prefix_name = (
                    f"cnnTempCV_{n_reso}_{shell_type_str}_{str_spec_types}_PI{ds_i}_P{period_j}"
                )
            else:
                prefix_name = (
                    f"cnnFluxCV_{n_reso}_{shell_type_str}_{str_spec_types}_PI{ds_i}_P{period_j}"
                )
            prefix_name += "_mask" if use_density_shell_mask else ""
            prefix_name += "_res" if use_residuals else ""

            # Common parameters for loading data
            data_params = dict(
                pis=[ds_i],
                periods=[period_j],
                use_temp=shell_type_temp,
                use_mask=use_density_shell_mask,
                use_residuals=use_residuals,
                data_dir=shells_dir,
            )

            # Load data for each spec_type using a list comprehension
            data_loader = [
                load_shell_astro_datah5(spec_type=st, **data_params) for st in spec_types
            ]
            shell_data_list, astrodata_list, density_data_list, _, _ = zip(*data_loader)

            # Concatenate data across spec_types
            shell_data_x = np.concatenate(shell_data_list, axis=0)
            density_data_x = np.concatenate(density_data_list, axis=0)
            astrodata = np.concatenate(astrodata_list, axis=0)

            logger.debug(f"shell data shape {np.shape(shell_data_x)}")
            #
            scalerx = MaskedStandardScaler3D()
            scalerx.fit(shell_data_x)
            #

            x = scalerx.transform(shell_data_x)

            # logger.info(f"X SIZE: {np.shape(x_1)}")

            # Process astrodata to extract target variable y (using first and last columns)
            y = astrodata[:, [0, -2]]
            scalery = StandardScaler()
            scalery.fit(y)
            y = scalery.transform(y)
            # Log the shapes to verify
            logger.debug(f"y SIZE: {np.shape(y)}")
            # Run randomized cross-validation
            cnn, model, pred = randomized_cv(
                num_cv=num_cv, train=train_model, x=x, y=y, prefix_name=prefix_name
            )
            time_df = pd.read_csv(f"{DATA_DIR}/time_df.csv")
            dates = time_df["jdb"].values
            pred2 = scalery.inverse_transform(pred)
            pred2_rv = pred2[:, 0]
            pred2_ds = pred2[:, 1]

            fap = 0.001

            print(dates)
            # # We can change spec_type for testing
            periodogram_output = generate_periodogram_test(
                real_rv=astrodata[:, 0],
                pred_rv=pred2_rv,
                pred_ds=pred2_ds,
                dates=dates,
                ds_size=ds_i,
                period=period_j,
                fap=fap,
                spec_type=spec_types[0],
                min_period=5,
                max_period=1000,
                plot=False,
                savefig=False,
            )

            clp_ds_pred = periodogram_output["clp_ds_pred"]
            power_limit = clp_ds_pred.powerLevel(fap)

            # Define ±5% frequency window around target frequency
            freq_min = 1.0 / (period_j * 1.05)
            freq_max = 1.0 / (period_j * 0.95)
            freq_window_mask = (clp_ds_pred.freq >= freq_min) & (clp_ds_pred.freq <= freq_max)

            freqs_in_window = clp_ds_pred.freq[freq_window_mask]
            powers_in_window = clp_ds_pred.power[freq_window_mask]

            # Check all peaks above power threshold
            above_thresh_mask = powers_in_window >= power_limit

            if np.any(above_thresh_mask):
                # Among peaks above threshold, find the one with highest power
                idx_best = np.argmax(powers_in_window * above_thresh_mask)
                detected_freq = freqs_in_window[idx_best]
                detected_period = 1.0 / detected_freq
                detected_power = powers_in_window[idx_best]
                logger.info(
                    f"Detected signal near {period_j} days (FAP < {fap}) at {detected_period:.2f} days"
                )

                # Check if detected peak is the highest peak above threshold in the full periodogram
                all_powers_above_threshold = clp_ds_pred.power[clp_ds_pred.power >= power_limit]
                is_highest_peak = np.isclose(detected_power, np.max(all_powers_above_threshold))
                if is_highest_peak:
                    df_detections_highest_peak.loc[period_j, ds_i] += 1
                    logger.debug(
                        f"Detected peak is the highest peak above threshold for period {period_j} days"
                    )
                else:
                    logger.debug(
                        f"Detected peak is NOT the highest peak above threshold for period {period_j} days"
                    )

                # Fit sine wave at detected period
                omega_det = 2 * np.pi / detected_period

                def sine_model_detected(t, A, phi, offset):
                    return A * np.sin(omega_det * t + phi) + offset

                t_norm = dates - dates[0]
                y = pred2_ds
                popt, _ = curve_fit(sine_model_detected, t_norm, y, p0=[ds_i, 0, 0])
                amplitude_raw, phi_fit, offset_fit = popt
                amplitude_detected = np.abs(amplitude_raw)

                # Amplitude error
                amplitude_diff = np.abs(ds_i - amplitude_detected)
                logger.debug("Difference between amplitudes (detected): %s", amplitude_diff)
                amplitude_perc = 100 * (ds_i - amplitude_detected) / ds_i

                # Phase offset at detected period
                detected_phase_offset = (phi_fit / (2 * np.pi)) % 1
                recovered_phase = recover_phase_offset(
                    dates, astrodata[:, -1], period_days=detected_period
                )
                phase_diff = np.abs(detected_phase_offset - recovered_phase)
                logger.debug("Recovered phase at detected period: %s", detected_phase_offset)
                logger.debug("Phase difference (detected): %s", phase_diff)

                # Period difference
                period_diff = np.abs(detected_period - period_j)

                # Save results
                df_detections.loc[period_j, ds_i] += 1
                df_detections_count.loc[period_j, ds_i] += 1
                df_amplitudes.loc[period_j, ds_i] += amplitude_diff
                df_amplitudes_perc.loc[period_j, ds_i] += amplitude_perc
                df_phases.loc[period_j, ds_i] += phase_diff
                df_periods.loc[period_j, ds_i] += period_diff

            else:
                logger.debug(f"No significant peak found near {period_j} days (±5%)")

with np.errstate(invalid="ignore", divide="ignore"):
    df_amplitudes = (df_amplitudes / df_detections_count).fillna(0.0)
    df_amplitudes_perc = (df_amplitudes_perc / df_detections_count).fillna(0.0)
    df_amplitudes_perc = df_amplitudes_perc.clip(upper=100)
    df_phases = (df_phases / df_detections_count).fillna(0.0)
    df_periods = (df_periods / df_detections_count).fillna(0.0)


df_detections = df_detections / n_evaluations
df_detect_binary = (df_detections >= 0.7).astype(int)

# Aggregate highest peak detection
df_detections_highest_peak = df_detections_highest_peak / n_evaluations
df_detect_highest_peak_binary = (df_detections_highest_peak >= 0.7).astype(int)

df_detections.to_csv(f"{LOCAL_OUTPUTS_DIR}/detections_{prefix_name}.csv")
df_detect_binary.to_csv(f"{LOCAL_OUTPUTS_DIR}/detections_binary_{prefix_name}.csv")
df_detect_highest_peak_binary.to_csv(
    f"{LOCAL_OUTPUTS_DIR}/detections_highest_peak_{prefix_name}.csv"
)
df_phases.to_csv(f"{LOCAL_OUTPUTS_DIR}/phases_{prefix_name}.csv")
df_amplitudes.to_csv(f"{LOCAL_OUTPUTS_DIR}/amplitudes_{prefix_name}.csv")
df_amplitudes_perc.to_csv(f"{LOCAL_OUTPUTS_DIR}/amplitudes_perc_{prefix_name}.csv")
df_periods.to_csv(f"{LOCAL_OUTPUTS_DIR}/periods_{prefix_name}.csv")

# df_detections = df_detections / n_evaluations
# df_detect_binary = (df_detections >= 0.7).astype(int)
# df_phases = df_phases / n_evaluations
# df_amplitudes = df_amplitudes / n_evaluations
# df_periods = df_periods / n_evaluations
# df_detections.to_csv(f'data/detection_test/CV/chunks/detections_{prefix_name}_{nrun}.csv')
# df_detect_binary.to_csv(f'data/detection_test/CV/chunks/detections_binary_{prefix_name}_{nrun}.csv')
# df_phases.to_csv(f'data/detection_test/CV/chunks/phases_{prefix_name}_{nrun}.csv')
# df_amplitudes.to_csv(f'data/detection_test/CV/chunks/amplitudes_{prefix_name}_{nrun}.csv')
# df_periods.to_csv(f'data/detection_test/CV/chunks/periods_{prefix_name}_{nrun}.csv')

logger.info("detections \n", df_detect_binary)
logger.info("amplitudes \n", df_amplitudes)
logger.info("phases \n", df_phases)
