import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from doppleriann.data import load_shell_astro_datah5

# Configuration
n_reso = 9
doppler_shift = 5.0
spec_type = 'or'
data_dir = f'data/shells{n_reso}/0/'

# Load data
def load_all_shells(use_temp):
    return [
        load_shell_astro_datah5(pis=[0.0], periods=[20], use_mask=False, use_residuals=False,
                                spec_type=spec_type, use_temp=use_temp, data_dir=data_dir),
        load_shell_astro_datah5(pis=[doppler_shift], periods=[20], use_mask=False, use_residuals=False,
                                spec_type=spec_type, use_temp=use_temp, data_dir=data_dir),
        load_shell_astro_datah5(pis=[0.0], periods=[20], use_mask=True, use_residuals=True,
                                spec_type=spec_type, use_temp=use_temp, data_dir=data_dir),
        load_shell_astro_datah5(pis=[doppler_shift], periods=[20], use_mask=True, use_residuals=True,
                                spec_type=spec_type, use_temp=use_temp, data_dir=data_dir),
    ]

flux_data = load_all_shells(False)
temp_data = load_all_shells(True)
shells_flux = [d[0] for d in flux_data]
astro_flux = [d[1] for d in flux_data]
shells_temp = [d[0] for d in temp_data]
astro_temp = [d[1] for d in temp_data]

# Frame selection
frame = np.argmax(astro_flux[1][:, -2])
frame = 1000
print(f"Frame with max flux DS: {frame}, value: {astro_flux[1][frame, -2]}")

# Compute vlims per group
v_flux_unmasked = np.max([np.abs(shells_flux[0][frame]), np.abs(shells_flux[1][frame])]) * 0.8
v_flux_masked = np.max([np.abs(shells_flux[2][frame]), np.abs(shells_flux[3][frame])]) * 0.8
v_temp_unmasked = np.max([np.abs(shells_temp[0][frame]), np.abs(shells_temp[1][frame])]) * 0.8
v_temp_masked = np.max([np.abs(shells_temp[2][frame]), np.abs(shells_temp[3][frame])]) * 0.8

# Set up figure and custom grid
fig = plt.figure(figsize=(15, 7))
gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 0.24, 1, 1], wspace=0.2, hspace=0.2)
axes = np.empty((2, 4), dtype=object)

# Create axes (skip column 2 for spacing)
for row in range(2):
    for i in range(4):
        col = i if i < 2 else i + 1  # insert gap at column 2
        axes[row, i] = fig.add_subplot(gs[row, col])

cmap = 'seismic'
titles = ["Shell without shift", "Shell with shift", "Masked shell without shift", "Masked shell with shift"]

def plot_cell(ax, shell, ds, vlim, cmap, show_title=False, ylabel=None, xlabel=None, xtick_labels=None):
    im = ax.imshow(shell, cmap=cmap, vmin=-vlim, vmax=vlim,
                   origin='lower', aspect='equal')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    ax.set_yticks([0, shell.shape[0]//2, shell.shape[0]-1])
    ax.set_yticklabels(["0.0", "0.5", "1.0"])
    ax.set_xticks([0, shell.shape[1]//2, shell.shape[1]-1])
    if xtick_labels:
        ax.set_xticklabels(xtick_labels)
    if show_title:
        ax.set_title(f"{titles[i]}\nDS = {ds:.2f} m/s", fontsize=10)
    ax.tick_params(labelsize=9)
    return im

# Plot Flux (top row)
for i in range(4):
    vlim = v_flux_unmasked if i < 2 else v_flux_masked
    plot_cell(
        axes[0, i], shells_flux[i][frame], astro_flux[i][frame, -2],
        vlim, cmap,
        show_title=True,
        ylabel="Normalized Flux" if i == 0 else None,
        xlabel=r"$\partial F/\partial v$",
        xtick_labels=["-15", "0", "15"]
    )

# Plot Temp (bottom row)
for i in range(4):
    vlim = v_temp_unmasked if i < 2 else v_temp_masked
    plot_cell(
        axes[1, i], shells_temp[i][frame], astro_temp[i][frame, -2],
        vlim, cmap,
        show_title=False,
        ylabel="Normalized  Temperature" if i == 0 else None,
        xlabel=r"$\partial T/\partial v$",
        xtick_labels=["-30", "0", "30"]
    )

# Colorbars
cb_flux_unmasked = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=-v_flux_unmasked, vmax=v_flux_unmasked), cmap=cmap)
cb_flux_masked = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=-v_flux_masked, vmax=v_flux_masked), cmap=cmap)
cb_temp_unmasked = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=-v_temp_unmasked, vmax=v_temp_unmasked), cmap=cmap)
cb_temp_masked = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=-v_temp_masked, vmax=v_temp_masked), cmap=cmap)

fig.colorbar(cb_flux_unmasked, ax=axes[0, :2], fraction=0.05, pad=0.02, label=r"$\Delta F$ ")
fig.colorbar(cb_flux_masked, ax=axes[0, 2:], fraction=0.05, pad=0.02, label=r"$\Delta F$ ")
fig.colorbar(cb_temp_unmasked, ax=axes[1, :2], fraction=0.05, pad=0.02, label=r"$\Delta T$ ")
fig.colorbar(cb_temp_masked, ax=axes[1, 2:], fraction=0.05, pad=0.02, label=r"$\Delta T$ ")

plt.tight_layout(rect=[0, 0, 0.92, 1])
plt.savefig("shells_separate_colorbars_spaced.png", dpi=500)
plt.show()
