shell_type_temp = False # True for temp, False for Flux shells

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
    periodogram,
    generate_periodogram_test,
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


np.random.seed(0)

# INPUT: shell, output regression of RV and DS, with CNN
load_model = False # True to load a trained model, False to train a new model
show_pred_plots = False
hpc_device = True
ds_size_test = 0.15
period_test = 100
n_reso = 9 # 9 or 15
large_datadir = LARGE_DATA_DIR
shells_dir = f'{DATA_DIR}/shells/0/'
# 1636 random indices for training  around 80% of the total
random_idx_train = np.load(f'{DATA_DIR}/random_idx_train.npy')
# 400 random indices for testing  around 20% of the total
# Testing with unseen elements.
random_idx_test = np.load(f'{DATA_DIR}/random_idx_test.npy')
# Testing with the first N spectra (including elements of the training set)
# random_idx_test = np.arange(2036)[:2000]
use_residuals = True
use_density_shell_mask = True
## Training set settings
planetary_injections =  [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
periods_train = [20, 40, 60, 80, 100]

shell_type_str = 'temp' if shell_type_temp else 'flux'
# spec_types: 'act', 'or'
spec_types = ['act']

# CNN settings
actfn = 'selu'
loss_fn = 'mean_squared_error'
epochs = 1000
# Callbacks
patience = 20
es = EarlyStopping(monitor='val_loss', patience=patience, min_delta=1e-5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=patience//2, min_delta=1e-5, min_lr=1e-6)
callbacks = [es, reduce_lr]
dropout_rate = 0.2
# Optuna results for batch size, learning rate, dropout and conv and dense layers.
# bs = 256
# learning_rate = 0.0033
# conv_layers =  [(256, 3), (512, 3)]
# dense_layers =  [512]
bs = 256
learning_rate = 0.0002
conv_layers =  [(128, 3), (256, 3)]
dense_layers =  [512]

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
    plt.savefig(f"{LOCAL_OUTPUTS_DIR}/{prefix_name}_lossfn.png")
    plt.show()
    # Save models
    cnn.save(f"{LOCAL_MODELS_DIR}/{prefix_name}.h5")


ds_size_test = 0.3
period_test = 50
spec_type = 'act'
test_set2, astrodatatest2, _, _, _ = load_shell_astro_datah5(pis=[ds_size_test], periods=[period_test], spec_type=spec_type, use_temp=shell_type_temp,
                                                                    use_mask=use_density_shell_mask, use_residuals=use_residuals,
                                                                    selected_idx=random_idx_test, data_dir=shells_dir)

# columns = ['rv', 'rv_err', 'fwhm', 'fwhm_err', 'bis', 'ds', 'phase']
phases_inj = astrodatatest2[:, -1]
print("phases", phases_inj, len(phases_inj))

time_df = pd.read_csv(f'{DATA_DIR}/time_df.csv')
dates = time_df['jdb'].values
dates = dates[random_idx_test]
print("dates", dates, len(dates))
test_set2 = scalerx.transform(test_set2)
# Predict the test set
pred2_mcdo = model.mcdo_predict(test_set2, cnn, mc_dropout_num=100)
pred2 = pred2_mcdo['mean']
logger.info("shape predictions", np.shape(pred2))
logger.info(np.shape(pred2[0]))
pred2 = scalery.inverse_transform(pred2)
pred2_rv = pred2[:, 0]
pred2_ds = pred2[:, 1]

fap = 0.001
# # We can change spec_type for testing
periodogram_output = generate_periodogram_test(real_rv=astrodatatest2[:, 0], pred_rv=pred2_rv, pred_ds=pred2_ds, dates=dates, ds_size=ds_size_test, period=period_test, fap=fap, spec_type=spec_type, min_period=5, max_period=1000, plot=True, savefig=True)
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

