import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

# --- DopplerIANN imports ---
from doppleriann.physics import astro_data
from doppleriann.utils.logger_config import logger

hpc_device = True
large_data_dir = (
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'large_data'))
    if hpc_device
    else '/media/isidro/data/large_data/harpn/'
)
# Load data
print("loading data")
data_dir = 'data/'
time_df = pd.read_csv('data/time_df.csv')
dates = pd.DatetimeIndex(time_df.date).to_julian_date()

wave_cols = np.loadtxt(data_dir + 'wavelengths.txt')

spectra_orig_np = np.load(data_dir + 'spectra_orig.npy')
spectra_active_np = np.load(data_dir + 'spectra_active.npy')

logger.info("Generating astro data for spectra with activity")
astro_data_active = astro_data(spectra_active_np, wave_cols)
astro_active_df = pd.DataFrame(astro_data_active, columns=["rv", "rv_err", "fwhm", "fwhm_err", "bis"])
astro_active_csv_file = os.path.join(data_dir, "astro_data_active.csv")
astro_active_df.to_csv(astro_active_csv_file, index=False)
logger.info(f"CCF data from spectra with activity saved to: {astro_active_csv_file}, {np.shape(astro_data_active)}")

logger.info("Generating astro data for spectra without activity")
astro_data_orig = astro_data(spectra_orig_np, wave_cols)
astro_orig_df = pd.DataFrame(astro_data_orig, columns=["rv", "rv_err", "fwhm", "fwhm_err", "bis"])
astro_orig_csv_file = os.path.join(data_dir, "astro_data_orig.csv")
astro_orig_df.to_csv(astro_orig_csv_file, index=False)
logger.info(f"CCF data from spectra without activity saved to: {astro_orig_csv_file}, {np.shape(astro_data_orig)}")
