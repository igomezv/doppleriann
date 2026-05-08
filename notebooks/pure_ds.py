shell_type_temp = True # True for temp, False for Flux shells

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- DopplerIANN imports ---
from doppleriann.data import (
    MaskedStandardScaler3D,
    load_shell_astro_datah5,
)
from doppleriann.physics import extract_shape_shell
from doppleriann.utils.logger_config import logger

np.random.seed(42)

# INPUT: shell, output regression of RV and DS, with CNN
load_model = True # True to load a trained model, False to train a new model
show_pred_plots = False
hpc_device = True
# ds_size_test = 0.3
# period_test = 80
n_reso = 9 # 9 or 15
large_datadir = '..data/' if hpc_device else '/media/isidro/data/data/harpn/'
shells_dir = f'data/shellsHidden'

use_residuals = True
use_density_shell_mask = True
spec_type = ['act']
# ## Training set settings
# # planetary_injections =  [0.3, 0.5, 1.0]
# # # First test
# # periods_train = [10, 20, 30, 40, 50, 100]
# # second test
# planetary_injections =  [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
# periods_train = [20, 40, 60, 80, 100]


ds_size_dop=0.5
period_dop=100 # it can be 10, 20, 30, 40,50,  ..., 100
shells_dir_test = f'data/shellsHidden/'
shells_dop, astrodatatest_dop, _, _, _ = load_shell_astro_datah5(pis=[0.5], periods=[100], spec_type='act', use_temp=shell_type_temp,
                                                             use_mask=use_density_shell_mask, use_residuals=use_residuals,
                                                             data_dir=shells_dir_test)

shells_or, astrodatatest_or, _, _, _ = load_shell_astro_datah5(pis=[0.0], periods=[20], spec_type=spec_type[0], use_temp=shell_type_temp,
                                                             use_mask=use_density_shell_mask, use_residuals=use_residuals,
                                                             data_dir=shells_dir_test)

frame = np.argmax(astrodatatest_or[:, -2])

shape_shells = []

for i in range(len(shells_dop)):
    shape_shell = extract_shape_shell(shells_or[i], shells_dop[i])
    shape_shells.append(shape_shell)


def plot_shell_comparison(shells_or, shells_dop, shape_shells, indices=[0, 1, 2]):
    n_plot = len(indices)
    fig, axs = plt.subplots(n_plot, 3, figsize=(12, 3.5 * n_plot))
    v_lim_or = np.max([np.abs(shells_or[2][frame]), np.abs(shells_or[3][frame])]) * 0.8
    v_lim_shape = np.max([np.abs(shape_shells[2][frame]), np.abs(shape_shells[3][frame])]) * 10.0

    cmap = 'seismic'

    for idx, i in enumerate(indices):
        shells = [shells_or[i], shells_dop[i], shape_shells[i]]
        titles = [f"Original Shell [{i}]", "Doppler Shell", "Shape Shell"]
        labels = [r"$\Delta T$", r"$\Delta T$", r"$\Delta_{\rm shape}$"]

        for j in range(3):
            if j == 2:
                im = axs[idx, j].imshow(shells[j], cmap=cmap, vmin=-v_lim_shape, vmax=v_lim_shape, origin='lower', aspect='auto')
            else:
                im = axs[idx, j].imshow(shells[j], cmap=cmap, vmin=-v_lim_or, vmax=v_lim_or, origin='lower', aspect='auto')
            axs[idx, j].set_title(titles[j], fontsize=10)
            axs[idx, j].set_xticks([0, shells[j].shape[1] // 2, shells[j].shape[1] - 1])
            axs[idx, j].set_xticklabels(["-15", "0", "15"])
            axs[idx, j].set_yticks([0, shells[j].shape[0] // 2, shells[j].shape[0] - 1])
            axs[idx, j].set_yticklabels(["0.0", "0.5", "1.0"])
            axs[idx, j].tick_params(labelsize=9)
            if j == 0:
                axs[idx, j].set_ylabel("Normalized Flux", fontsize=11)
            axs[idx, j].set_xlabel(r"$\partial T/\partial v$", fontsize=11)

            # Add colorbar
            cbar = fig.colorbar(im, ax=axs[idx, j], fraction=0.046, pad=0.02)
            cbar.set_label(labels[j], fontsize=10)
            cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()

plot_shell_comparison(shells_or, shells_dop, shape_shells, indices=[0, 1, 2])