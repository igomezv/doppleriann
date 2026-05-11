shell_type_temp = True # True for temp, False for Flux shells

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD, AdamW
from tensorflow.keras.losses import Huber, MeanSquaredError

# --- DopplerIANN imports ---
from doppleriann.networks import ShellCNN1D
from doppleriann.data import (
    MaskedStandardScaler3D,
    load_shell_astro_datah5,
)
from doppleriann.physics import (
    recover_phase_offset,
    generate_periodogram_test,
    circ_dist_cycles,
    long_term_remover,
    sinusoidal_model,
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

# np.random.seed(42)

# INPUT: shell, output regression of RV and DS, with CNN
load_model = True # True to load a trained model, False to train a new model
show_pred_plots = False
hpc_device = True
ds_size_test = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
period_test = [10, 20 , 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
# Create DataFrame of zeros
df_detections = pd.DataFrame(
                data=np.zeros((len(period_test), len(ds_size_test))),
                index=period_test,
                columns=ds_size_test
)

df_amplitudes = pd.DataFrame(
                data=np.zeros((len(period_test), len(ds_size_test))),
                index=period_test,
                columns=ds_size_test
)

df_amplitudes_perc = pd.DataFrame(
                data=np.zeros((len(period_test), len(ds_size_test))),
                index=period_test,
                columns=ds_size_test
)

df_phases = pd.DataFrame(
                data=np.zeros((len(period_test), len(ds_size_test))),
                index=period_test,
                columns=ds_size_test
)

df_periods = pd.DataFrame(
                data=np.zeros((len(period_test), len(ds_size_test))),
                index=period_test,
                columns=ds_size_test
)

df_detections_count = pd.DataFrame(
    data=np.zeros((len(period_test), len(ds_size_test))),
    index=period_test,
    columns=ds_size_test
)


df_variance_rv = pd.DataFrame(
                data=np.zeros((len(period_test), len(ds_size_test))),
                index=period_test,
                columns=ds_size_test
)

df_variance_ds = pd.DataFrame(
                data=np.zeros((len(period_test), len(ds_size_test))),
                index=period_test,
                columns=ds_size_test
)

df_residuals_rv = pd.DataFrame(
                data=np.zeros((len(period_test), len(ds_size_test))),
                index=period_test,
                columns=ds_size_test
)

df_residuals_ds = pd.DataFrame(
                data=np.zeros((len(period_test), len(ds_size_test))),
                index=period_test,
                columns=ds_size_test
)


df_med_residuals_rv = pd.DataFrame(
                data=np.zeros((len(period_test), len(ds_size_test))),
                index=period_test,
                columns=ds_size_test
)

df_med_residuals_ds = pd.DataFrame(
                data=np.zeros((len(period_test), len(ds_size_test))),
                index=period_test,
                columns=ds_size_test
)

print(df_detections)

n_reso = 9 # 9 or 15
large_datadir = LARGE_DATA_DIR
shells_dir = f'{DATA_DIR}/shells/0/'
# 1636 random indices for training  around 80% of the total
# 400 random indices for testing  around 20% of the total
# Testing with unseen elements.
random_idx_train = np.load(f"{DATA_DIR}/random_idx_train.npy")
ntrain = len(random_idx_train)
# 400 random indices for testing  around 20% of the total
# Testing with unseen elements.
random_idx_test = np.load(f"{DATA_DIR}/random_idx_test.npy")
random_idx_test = np.arange(0, len(random_idx_test))
random_idx_train = np.arange(len(random_idx_test), ntrain+len(random_idx_test))
use_residuals = True
use_density_shell_mask = True

planetary_injections =  [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
periods_train = [20, 40, 60, 80, 100]

# planetary_injections =  [0.1, 0.2, 1.0]
# periods_train = [20, 50, 100]

shell_type_str = 'temp' if shell_type_temp else 'flux'
# spec_types: 'act', 'or'
spec_types = ['act']

# CNN settings
actfn = 'selu'
loss_fn = 'mean_squared_error'
epochs = 1000
# Callbacks
patience = 40
es = EarlyStopping(monitor='val_loss', patience=patience, min_delta=1e-5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=patience//2, min_delta=1e-5, min_lr=1e-6)
callbacks = [es, reduce_lr]
dropout_rate = 0.2
# Optuna results for batch size, learning rate, dropout and conv and dense layers.
# bs = 256
# learning_rate = 0.0033
# conv_layers =  [(256, 3), (512, 3)]
# dense_layers =  [512]
       
if shell_type_temp:
    bs = 128
    conv_layers = [(128, 5), (256, 5)]
    dense_layers = [512]
    learning_rate = 0.0002
else:
    bs = 256
    learning_rate = 0.002
    conv_layers =  [(128, 3), (256, 3)]
    dense_layers =  [512]
    optimizer = Adam(learning_rate=learning_rate)


optimizer = Adam(learning_rate=learning_rate)

str_spec_types = '_'.join(spec_types)
prefix_name = f'cnnshell_{n_reso}_{shell_type_str}_{str_spec_types}'
prefix_name += '_mask' if use_density_shell_mask else ''
prefix_name += '_res' if use_residuals else ''

# Common parameters for loading data
data_params = dict(
    pis=planetary_injections,
    periods=periods_train,
    use_temp=shell_type_temp,
    use_mask=use_density_shell_mask,
    use_residuals=use_residuals,
    data_dir=shells_dir,
    selected_idx=random_idx_train)

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

model = ShellCNN1D(input_shape=(x.shape[1], x.shape[2]), n_outputs=2, conv_layers=conv_layers, dense_layers=dense_layers, dropout=dropout_rate, actfn=actfn)

if load_model:
    cnn = model.load_model(f'{LOCAL_MODELS_DIR}/{prefix_name}.h5')
    cnn.summary()
else:
    cnn = model.model_tf()
    # Compile the model using mse
    cnn.compile(optimizer=optimizer, loss=loss_fn, metrics=['mean_absolute_error'])
    cnn.summary()
    # Train the model
    history = cnn.fit(x, y, epochs=epochs, batch_size=bs, validation_split=0.2, callbacks=callbacks, shuffle=True)
    loss = cnn.history.history['loss']
    val_loss = cnn.history.history['val_loss']
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Dense-MLP Loss | mse_train {:.4f} | mse_val: {:.4f} '.format(loss[-1], val_loss[-1]))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{LOCAL_OUTPUTS_DIR}/{prefix_name}_lossfn.png", dpi=300)
    plt.show()
    # Save models
    cnn.save(f"{LOCAL_MODELS_DIR}/{prefix_name}.h5")



spec_type = 'act'
# Number of evaluations
n_evaluations = 10
for idx_ev in range(n_evaluations):
    for i, ds_i in enumerate(ds_size_test):
        for j, period_j in enumerate(period_test):
            logger.info(f"Evaluation {i+1}/{n_evaluations}, ds_size: {ds_i}, period: {period_j}, spec_type: {spec_type}")
            # Convert to frequency
            target_freq = 1.0 / period_j
            # Load test set
            shells_dir_test = f'{DATA_DIR}/shells/{idx_ev}/'
            test_set2, astrodatatest2, _, _, _ = load_shell_astro_datah5(pis=[ds_i], periods=[period_j], spec_type=spec_type, use_temp=shell_type_temp,
                                                                        use_mask=use_density_shell_mask, use_residuals=use_residuals,
                                                                        selected_idx=random_idx_test, data_dir=shells_dir_test)

            # columns = ['rv', 'rv_err', 'fwhm', 'fwhm_err', 'bis', 'ds', 'phase']
            phases_inj = astrodatatest2[:, -1]

            time_df = pd.read_csv(f'{DATA_DIR}/time_df.csv')
            dates = time_df['jdb'].values
            dates = dates[random_idx_test]
            # print("dates", dates, len(dates))
            test_set2 = scalerx.transform(test_set2)
            # Predict the test set
            pred2_mcdo = model.mcdo_predict(test_set2, cnn, mc_dropout_num=100)
            pred2 = pred2_mcdo['mean']
            pred2_std = pred2_mcdo['std']
            logger.debug("shape predictions", np.shape(pred2))
            logger.debug(np.shape(pred2[0]))
            pred2 = scalery.inverse_transform(pred2)
            pred2_std = scalery.scale_ * pred2_std
            pred2_rv = pred2[:, 0]
            pred2_rv_std = pred2_std[:, 0]
            pred2_ds = pred2[:, 1]
            # pred2_ds_detrended = long_term_remover(dates, pred2_ds, degree=1)
            pred2_ds_std = pred2_std[:, 1]
            # res_rv = np.abs(astrodatatest2[:, 0] - pred2_rv)
            # res_ds = np.abs(astrodatatest2[:, -2] - pred2_ds)
            err_rv = astrodatatest2[:, 0] - pred2_rv
            err_ds = astrodatatest2[:, -2] - pred2_ds

            rmse_rv = np.sqrt(np.mean(err_rv**2))
            rmse_ds = np.sqrt(np.mean(err_ds**2))

            fap = 0.001
            # # We can change spec_type for testing
            periodogram_output = generate_periodogram_test(
                real_rv=astrodatatest2[:, 0],
                pred_rv=pred2_rv,
                pred_ds=pred2_ds,
                dates=dates,
                ds_size=ds_i,
                period=period_j,
                fap=fap,
                spec_type=spec_type,
                min_period=5,
                max_period=1000,
                plot=False,
                savefig=False
            )

            clp_ds_pred = periodogram_output['clp_ds_pred']
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
                # idx_best = np.argmax(powers_in_window * above_thresh_mask)
                # detected_freq = freqs_in_window[idx_best]
                idx_best = np.argmax(powers_in_window[above_thresh_mask])
                detected_freq = freqs_in_window[above_thresh_mask][idx_best]
                detected_period = 1. / detected_freq
                logger.info(f"Detected signal near {period_j} days (FAP < {fap}) at {detected_period:.2f} days")

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
                amplitude_diff = np.abs(amplitude_detected -ds_i)
                logger.debug("Difference between amplitudes (detected): %s", amplitude_diff)
                amplitude_perc = 100 * np.abs(ds_i - amplitude_detected) / ds_i

                # Phase offset at detected period
                detected_phase_offset = (phi_fit / (2 * np.pi)) % 1
                # Injected phase offset recovery
                inj_phase_offset = recover_phase_offset(
                                                        dates,
                                                        astrodatatest2[:, -1],          # radians
                                                        period_days=period_j,           # injected period (truth)
                                                        reference_date=dates[0],        # reference date for phase calculation
                                                        )
                phase_diff = circ_dist_cycles(detected_phase_offset, inj_phase_offset)

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

            df_variance_rv.loc[period_j, ds_i] += np.median(pred2_rv_std)
            df_variance_ds.loc[period_j, ds_i] += np.median(pred2_ds_std)
            df_med_residuals_rv.loc[period_j, ds_i] += np.median(rmse_rv)
            df_med_residuals_ds.loc[period_j, ds_i] += np.median(rmse_ds)
            df_residuals_rv.loc[period_j, ds_i] += rmse_rv
            df_residuals_ds.loc[period_j, ds_i] += rmse_ds
            

with np.errstate(invalid='ignore', divide='ignore'):
    df_amplitudes = (df_amplitudes / df_detections_count).fillna(0.0)
    df_amplitudes_perc = (df_amplitudes_perc / df_detections_count).fillna(0.0)
    df_amplitudes_perc = df_amplitudes_perc.clip(upper=100)
    df_phases = (df_phases / df_detections_count).fillna(0.0)
    df_periods = (df_periods / df_detections_count).fillna(0.0)


df_variance_rv = df_variance_rv / n_evaluations
df_variance_ds = df_variance_ds / n_evaluations
df_residuals_rv = df_residuals_rv / n_evaluations
df_residuals_ds = df_residuals_ds / n_evaluations
df_med_residuals_rv = df_med_residuals_rv / n_evaluations
df_med_residuals_ds = df_med_residuals_ds / n_evaluations
df_detections = df_detections / n_evaluations
df_detect_binary = (df_detections >= 0.7).astype(int)

df_detections.to_csv(f'{LOCAL_OUTPUTS_DIR}/detections_{prefix_name}.csv')
df_detect_binary.to_csv(f'{LOCAL_OUTPUTS_DIR}/detections_binary_{prefix_name}.csv')
df_variance_rv.to_csv(f'{LOCAL_OUTPUTS_DIR}/variance_rv_{prefix_name}.csv')
df_variance_ds.to_csv(f'{LOCAL_OUTPUTS_DIR}/variance_ds_{prefix_name}.csv')
df_residuals_rv.to_csv(f'{LOCAL_OUTPUTS_DIR}/residuals_rv_{prefix_name}.csv')
df_residuals_ds.to_csv(f'{LOCAL_OUTPUTS_DIR}/residuals_ds_{prefix_name}.csv')
df_med_residuals_rv.to_csv(f'{LOCAL_OUTPUTS_DIR}/residuals_med_rv_{prefix_name}.csv')
df_med_residuals_ds.to_csv(f'{LOCAL_OUTPUTS_DIR}/residuals_med_ds_{prefix_name}.csv')
df_phases.to_csv(f'{LOCAL_OUTPUTS_DIR}/phases_{prefix_name}.csv')
df_amplitudes.to_csv(f'{LOCAL_OUTPUTS_DIR}/amplitudes_{prefix_name}.csv')
df_amplitudes_perc.to_csv(f'{LOCAL_OUTPUTS_DIR}/amplitudes_perc_{prefix_name}.csv') 
df_periods.to_csv(f'{LOCAL_OUTPUTS_DIR}/periods_{prefix_name}.csv')

logger.info("detections \n", df_detect_binary)
logger.info("amplitudes \n", df_amplitudes)
logger.info("phases \n", df_phases)