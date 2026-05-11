import numpy as np
import pandas as pd
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
from doppleriann.physics import generate_periodogram_test
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

shell_type_temp = False # True for temp, False for Flux shells
use_residuals = True
use_density_shell_mask = True
train_model = True  # Set to True to train the model, False to load and predict
# INPUT: shell, output regression of RV and DS, with CNN
show_pred_plots = False
hpc_device = True
n_reso = 9 
large_datadir = LARGE_DATA_DIR
shells_dir = f'{DATA_DIR}/shells{n_reso}/0/'

## Training set settings
planetary_injection =  [0.2]  
period = [100]
shell_type_str = 'temp' if shell_type_temp else 'flux'
# spec_types: 'act', 'or'
spec_types = ['act']  
str_spec_types = '_'.join(spec_types)                                                   
prefix_name = f'cnnFluxCV_{n_reso}_{shell_type_str}_{str_spec_types}_PI{planetary_injection[0]}_P{period[0]}'
prefix_name += '_mask' if use_density_shell_mask else ''
prefix_name += '_res' if use_residuals else ''

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
def randomized_cv(num_cv=10, train=True, x=None, y=None):
    all_train_losses = []
    all_val_losses = []
    
    # Prepare an array to hold predictions in the original order
    all_predictions_ordered = np.zeros((2036, 2))  # Assuming y is (2036, n_outputs)

    for i in range(num_cv):
        test_indices = folds[i]
        train_indices = [idx for j, fold in enumerate(folds) if j != i for idx in fold]
        print(f"Fold {i+1}:")
        print(f"  Test indices length: {len(test_indices)}")
        print(f"  Train indices length: {len(train_indices)}\n")

        x_train = x[train_indices]
        x_test = x[test_indices]

        # CNN settings
        actfn = 'selu'
        loss_fn = 'mean_squared_error'
        epochs = 500
        patience = 40
        es = EarlyStopping(monitor='val_loss', patience=patience, min_delta=1e-5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=patience//2, min_delta=1e-5, min_lr=1e-6)
        callbacks = [es, reduce_lr]
        dropout_rate = 0.2
        # Optuna results for batch size, learning rate, and conv and dense layers.
        bs = 256
        learning_rate = 0.002
        conv_layers =  [(128, 3), (256, 3)]
        dense_layers =  [512]

        optimizer = Adam(learning_rate=learning_rate)

        model = ShellCNN1D(
            input_shape=(x.shape[1], x.shape[2]),
            n_outputs=2,
            conv_layers=conv_layers,
            dense_layers=dense_layers,
            dropout=dropout_rate,
            actfn=actfn
        )

        if train:
            y_train = y[train_indices]
            y_test = y[test_indices]

            cnn = model.model_tf()
            cnn.compile(optimizer=optimizer, loss=loss_fn, metrics=['mean_absolute_error'])
            cnn.summary()
            history = cnn.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=bs,
                callbacks=callbacks,
                shuffle=False,
                validation_data=(x_test, y_test)
            )
            all_train_losses.append(history.history['loss'])
            all_val_losses.append(history.history['val_loss'])

            cnn.save(f"{LOCAL_MODELS_DIR}/{prefix_name}_cv_{i}.h5")
        else:
            cnn = model.load_model(f'{LOCAL_MODELS_DIR}/{prefix_name}_cv_{i}.h5')
            cnn.summary()

        pred_mcdo = model.mcdo_predict(x_test, cnn)
        pred = pred_mcdo['mean']

        # Store predictions in their correct place in the ordered array
        for idx, p in zip(test_indices, pred):
            all_predictions_ordered[idx] = p

    if train:
        mean_train_loss = np.mean(np.array([np.pad(l, (0, epochs-len(l)), 'edge') for l in all_train_losses]), axis=0)
        mean_val_loss = np.mean(np.array([np.pad(l, (0, epochs-len(l)), 'edge') for l in all_val_losses]), axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(mean_train_loss, label='Mean Training Loss')
        plt.plot(mean_val_loss, label='Mean Validation Loss')
        plt.title(f'Average Loss over {num_cv} Random Splits')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{LOCAL_OUTPUTS_DIR}/{prefix_name}_mean_cv_loss.png")
        plt.show()

    return cnn, model, all_predictions_ordered

# Train
# Common parameters for loading data
data_params = dict(
    pis=planetary_injection,
    periods=period,
    use_temp=shell_type_temp,
    use_mask=use_density_shell_mask,
    use_residuals=use_residuals,
    data_dir=shells_dir)

# Load data for each spec_type using a list comprehension
data_loader = [load_shell_astro_datah5(spec_type=st, **data_params) for st in spec_types]
shell_data_list, astrodata_list, density_data_list, _, _ = zip(*data_loader)

# Concatenate data across spec_types
shell_data_x = np.concatenate(shell_data_list, axis=0)
density_data_x = np.concatenate(density_data_list, axis=0)
astrodata = np.concatenate(astrodata_list, axis=0)


logger.info(f"shell data shape {np.shape(shell_data_x)}")
#
scalerx = MaskedStandardScaler3D()
scalerx.fit(shell_data_x)
#

x = scalerx.transform(shell_data_x)

logger.info(f"min shell: {np.min(shell_data_x)}, max shell: {np.max(shell_data_x)}")
# logger.info(f"X SIZE: {np.shape(x_1)}")

# Process astrodata to extract target variable y (using first and last columns)
y = astrodata[:, [0, -2]]
scalery = StandardScaler()
scalery.fit(y)
y = scalery.transform(y)
# Log the shapes to verify
logger.info(f"y SIZE: {np.shape(y)}")
# Run randomized cross-validation
cnn, model, pred = randomized_cv(num_cv=num_cv, train=train_model, x=x, y=y)
time_df = pd.read_csv(f'{DATA_DIR}/time_df.csv')
dates = pd.DatetimeIndex(time_df.date).to_julian_date()
pred2 = scalery.inverse_transform(pred)
pred2_rv = pred2[:, 0]
pred2_ds = pred2[:, 1]

fap = 0.001

print(dates)
# # We can change spec_type for testing
periodogram_output = generate_periodogram_test(real_rv=astrodata[:, 0], pred_rv=pred2_rv, pred_ds=pred2_ds, 
                                               dates=dates, ds_size=planetary_injection[0], period=period[0],
                                               fap=fap, spec_type=spec_types[0], min_period=5, max_period=1000,
                                               plot=True, savefig=True)
fig = periodogram_output['fig']
plt.show()

clp_ds_pred = periodogram_output['clp_ds_pred']
print("clp_ds_pred", clp_ds_pred)
# Get index associated with highest power
ifmax = np.argmax(clp_ds_pred.power)
# and highest power and associated frequency
pmax = clp_ds_pred.power[ifmax]
fmax = clp_ds_pred.freq[ifmax]
# Convert frequency into period
hpp = 1./fmax
print("Highest-power period: ", hpp)
phase = (dates % hpp) / hpp
print("Phase: ", phase)
