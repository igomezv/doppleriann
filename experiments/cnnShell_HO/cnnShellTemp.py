shell_type_temp = True  # True for temp, False for Flux shells

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"[INFO] Data directory: {DATA_DIR}")
logger.info(f"[INFO] Local models directory: {LOCAL_MODELS_DIR}")
logger.info(f"[INFO] Local outputs directory: {LOCAL_OUTPUTS_DIR}")


np.random.seed(0)

# INPUT: shell, output regression of RV and DS, with CNN
load_model = True  # True to load a trained model, False to train a new model
show_pred_plots = False
hpc_device = True
ds_size_test = 0.35
period_test = 50
n_reso = 9  # 9 or 15
large_datadir = LARGE_DATA_DIR
shells_dir = f"{DATA_DIR}/shells/0/"
# 1636 random indices for training  around 80% of the total
random_idx_train = np.load(f"{DATA_DIR}/random_idx_train.npy")
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
# Optuna results for batch size, learning rate, and conv and dense layers.
bs = 128
conv_layers = [(256, 5), (512, 5)]
dense_layers = [512]
learning_rate = 0.0002

optimizer = Adam(learning_rate=learning_rate)

str_spec_types = "_".join(spec_types)
prefix_name = f"cnnshell_{n_reso}_{shell_type_str}_{str_spec_types}"
# prefix_name = f"cnnshellFIRST_{n_reso}_{shell_type_str}_{str_spec_types}"
# prefix_name = f"cnnshellLAST_{n_reso}_{shell_type_str}_{str_spec_types}"

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

# Process astrodata to extract target variable y (using RV and DS columns)
y = astrodata[:, [0, -2]]
scalery = StandardScaler()
scalery.fit(y)
y = scalery.transform(y)

print("idx random train")
print(random_idx_train)
print("idx random test")
print(random_idx_test)
logger.info(f"X SIZE: {np.shape(x)} | y SIZE: {np.shape(y)}")

model = ShellCNN1D(
    input_shape=(x.shape[1], x.shape[2]),
    n_outputs=2,
    conv_layers=conv_layers,
    dense_layers=dense_layers,
    dropout=dropout_rate,
    actfn=actfn,
)

if load_model:
    cnn = model.load_model(f"{LOCAL_MODELS_DIR}/{prefix_name}.h5")
    cnn.summary()
else:
    cnn = model.model_tf()
    # Compile the model using mse
    cnn.compile(optimizer=optimizer, loss=loss_fn, metrics=["mean_absolute_error"])
    cnn.summary()
    # Train the model
    history = cnn.fit(
        x, y, epochs=epochs, batch_size=bs, validation_split=0.2, callbacks=callbacks, shuffle=True
    )
    loss = cnn.history.history["loss"]
    val_loss = cnn.history.history["val_loss"]
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title(
        "Dense-MLP Loss | mse_train {:.4f} | mse_val: {:.4f} ".format(loss[-1], val_loss[-1])
    )
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{LOCAL_OUTPUTS_DIR}/{prefix_name}_lossfn.png")
    plt.show()
    # Save models
    cnn.save(f"{LOCAL_MODELS_DIR}/{prefix_name}.h5")


ds_size_test = 0.35
period_test = 50
spec_type = "act"
shells_dir_test = f"{DATA_DIR}/shells/1/"
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
time_df = pd.read_csv(f"{DATA_DIR}/time_df.csv")
dates = time_df["jdb"].values
dates = dates[random_idx_test]
print("dates", dates, len(dates))
test_set2 = scalerx.transform(test_set2)
# Predict the test set
pred2_mcdo = model.mcdo_predict(test_set2, cnn, mc_dropout_num=100)
pred2 = pred2_mcdo["mean"]
logger.info(f"shape predictions {np.shape(pred2)}")
logger.info(f"first prediction shape {np.shape(pred2[0])}")

pred2 = scalery.inverse_transform(pred2)
np.save(
    LOCAL_OUTPUTS_DIR / f"{prefix_name}_pred_rv_ds_test_ds{ds_size_test}_P{int(period_test)}.npy",
    pred2,
)
pred2_rv = pred2[:, 0]
pred2_ds = pred2[:, 1]
# pred2_ds_detrended = long_term_remover(dates, pred2_ds, degree=1)

fap = 0.001
# # We can change spec_type for testing
periodogram_output = generate_periodogram_test(
    real_rv=astrodatatest2[:, 0],
    pred_rv=pred2_rv,
    pred_ds=pred2_ds,
    dates=dates,
    ds_size=ds_size_test,
    period=period_test,
    fap=fap,
    shell_type_str="HO",
    plot=True,
    savefig=True,
    min_period=5,
    max_period=900,
)
fig = periodogram_output["fig"]

plt.savefig(f"{LOCAL_OUTPUTS_DIR}/{prefix_name}_periodogram_test_{period_test}days.png")
plt.savefig(f"{LOCAL_OUTPUTS_DIR}/{prefix_name}_periodogram_test_{period_test}days.pdf")
plt.show()

clp_ds_pred = periodogram_output["clp_ds_pred"]
print("clp_ds_pred", clp_ds_pred)
# Get index associated with highest power
ifmax = np.argmax(clp_ds_pred.power)
# and highest power and associated frequency
pmax = clp_ds_pred.power[ifmax]
fmax = clp_ds_pred.freq[ifmax]
# Convert frequency into period
hpp = 1.0 / fmax
print("Highest-power period: ", hpp)
target_freq = 1.0 / period_test
i_closest = np.argmin(np.abs(clp_ds_pred.freq - target_freq))
power_at_target = clp_ds_pred.power[i_closest]
amplitude_at_target = np.sqrt(2 * power_at_target)
print("Power at test period: ", power_at_target)
results = clp_ds_pred.info()
phase = results["phase"]
amplitude = results["amplitude"]
print("Amplitude at test period: ", amplitude_at_target)
print("% Difference between amplitudes: ", 100 * (np.abs(amplitude - ds_size_test) / ds_size_test))
print("Phase: ", phase)
recovered_phase = recover_phase_offset(
    dates, astrodatatest2[:, -1], period_days=results["best_sine_period"]
)
print("Recovered phase:", recovered_phase)
print("Difference between phases:", np.abs(phase - recovered_phase))
