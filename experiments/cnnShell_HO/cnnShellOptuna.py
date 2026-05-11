import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import tensorflow as tf
import optuna

# --- DopplerIANN imports ---
from doppleriann.networks import ShellCNN1D
from doppleriann.data import (
    MaskedStandardScaler3D,
    load_shell_astro_datah5,
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


logger.info("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

np.random.seed(42)

shell_type_temp = True # True for temp, False for Flux shells
use_residuals = True
use_density_shell_mask = True

# INPUT: shell, output regression of RV and DS, with CNN
hpc_device = True
n_reso = 9 # 9 or 15
large_datadir = LARGE_DATA_DIR
shells_dir = f'{DATA_DIR}/shells{n_reso}/0/'
# 1636 random indices for training  around 80% of the total
random_idx_train = np.load(f'{DATA_DIR}/random_idx_train.npy')
# random_idx_test = np.arange(2036)[-1000:]  # last spectra for testing

## Training set settings
# working pretty well, [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0], using activ as training type and temp shells.
# planetary_injections =  [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
# periods_train = [20, 40, 60, 80, 100]
planetary_injections =  [0.0]
periods_train = [20]

shell_type_str = 'temp' if shell_type_temp else 'flux'
# spec_types: 'act', 'or'
spec_types = ['act']

str_spec_types = '_'.join(spec_types)
prefix_name = f'cnnShell_{n_reso}_{shell_type_str}_{str_spec_types}'
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
shell_data_list, astrodata_list, density_data_list, _, _ = zip(*data_loader)

# Concatenate data across spec_types
shell_data_x = np.concatenate(shell_data_list, axis=0)
astrodata = np.concatenate(astrodata_list, axis=0)

logger.info(f"shell data shape {np.shape(shell_data_x)}")

scalerx1 = MaskedStandardScaler3D()

scalerx1.fit(shell_data_x)

x_1 = scalerx1.transform(shell_data_x)

logger.info(f"min shell: {np.min(shell_data_x)}, max shell: {np.max(shell_data_x)}")
logger.info(f"X SIZE: {np.shape(x_1)}")

# Process astrodata to extract target variable y (using first and last columns)
y = astrodata[:, [0, -2]]
scalery = StandardScaler()
scalery.fit(y)
y = scalery.transform(y)

# Log the shapes to verify
logger.info(f"y SIZE: {np.shape(y)}")

epochs = 200
patience = 40

def opt_spec_net(trial):
    actfn = 'selu'
    # Batch size combinations
    batch_size = trial.suggest_categorical("batch_size", [128, 256])
    # 4 combinations of conv layers
    conv_layers =  trial.suggest_categorical("conv_layers", [[(128, 3), (256, 3)], [(256, 3), (512, 3)],
                                                            [(128, 5), (256, 5)], [(256, 5), (512, 5)]])
    # Combinations of dense layers
    dense_layers = trial.suggest_categorical("dense_layers", [[512], [512, 256], [512, 256, 128], [256], [256, 128], [256, 128, 64]])
    # Learning rate logarithm sampling
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    es = EarlyStopping(monitor='val_loss', patience=patience, min_delta=1e-5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=patience//2, min_delta=1e-5, min_lr=1e-6)
    callbacks = [es, reduce_lr]
    optimizer = Adam(learning_rate=learning_rate)
    # Dropout suggestions
    # dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.4)
    dropout_rate = 0.2
    model = ShellCNN1D(input_shape=(x_1.shape[1], x_1.shape[2]), n_outputs=2, conv_layers=conv_layers, dense_layers=dense_layers, dropout=dropout_rate, actfn=actfn)
    cnn = model.model_tf()
    # Compile the model using mse
    loss_fn = 'mean_squared_error'
    cnn.compile(optimizer=optimizer, loss=loss_fn, metrics=['mean_absolute_error'])
    # Train the model
    cnn.fit(x_1, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=callbacks, shuffle=True)
    val_loss = cnn.history.history['val_loss']
    # Save models
    # cnn.save("models/{}.h5".format(prefix_name))
    # Save model if it's the best so far
    trial_score = val_loss[-1]
    # if trial.study.best_trial is None or trial_score < trial.study.best_trial.value:
    #     cnn.save("models/{}.h5".format(prefix_name))
    #     print(f" Saved new best model with val_loss = {trial_score:.6f}")

    return trial_score  # Return the last validation loss as the objective value


# Run optimization
sampler = optuna.samplers.NSGAIISampler()
study = optuna.create_study(directions=["minimize"], sampler=sampler)
study.optimize(opt_spec_net, n_trials=50)

# Save best trial to a text file
with open(f"{LOCAL_OUTPUTS_DIR}/best_trial_results_{prefix_name}.txt", "w") as f:
    print(f"Best Trial MSE: {study.best_trial.value:.6f}\n")
    print("Best Hyperparameters:\n")
    f.write(f"Best Trial MSE: {study.best_trial.value:.6f}\n")
    f.write("Best Hyperparameters:\n")
    for key, value in study.best_trial.params.items():
        f.write(f"  {key}: {value}\n")
