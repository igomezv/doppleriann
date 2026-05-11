import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

# Ensure local modules are accessible when running from /test or elsewhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from doppleriann.utils.logger_config import logger
from doppleriann.physics import SpectrumData, interp_temp_given_flux
from doppleriann.data import load_hdf5_data  # optional helper, for consistency



device_hpc = True
large_data_dir = (
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'large_data'))
    if device_hpc
    else '/media/isidro/data/large_data/harpn/'
)
os.makedirs(large_data_dir, exist_ok=True)
# from Khaled's temperature calculations
# We use the same mask for flux and temperature shells to have a fair comparison
maskfile = 'data/mask_kitcat_NEW_kitcat_CCF_mask_Sun.npz'
# Files for temperature shells
file_path = 'data/T1o2_spec.csv'
t_data = pd.read_csv(file_path)
waves_temp = t_data['wave'].values
temp_temp = t_data['T1o2'].values
logger.info("shape temp", np.shape(temp_temp))
shape_temp = np.shape(temp_temp)
temp_scaler = MinMaxScaler(feature_range=(0, 1))
temp_scaler.fit(temp_temp.reshape(-1, 1))
temp_temp = temp_scaler.transform(temp_temp.reshape(-1,1)).reshape(shape_temp)
spectra_err_val = np.load('data/spectra_orig_err.npy')

# First we load the data
spec_types = ['act', 'or']

# dgen = DataGenerator(large_data_dir)
def load_spectra(spec_type_str):
        if spec_type_str == 'sim':
            # Simulated spectra, removing outliers
            spectra_full = np.load(os.path.join(large_data_dir, 'sims_and_real.npy'))
            spectra_values = spectra_full[3612:, :]
            outliers_idx = [idx for idx in [246, 249, 1196, 1453, 2176]]
            spectra_values = np.delete(spectra_values, outliers_idx, axis=0)
            # Time information
            time_info = np.load('data/SDO_simulation_GPU_SOAP_RV_data.npz')['time']
            time_info = np.delete(time_info, outliers_idx, axis=0)
            waves_obs = np.loadtxt('data/waves_filt_obs.txt')
        elif spec_type_str == 'or':
            # Spectra with corrected values
            spectra_values = np.load('data/spectra_orig.npy')
            # Time information
            time_df = pd.read_csv('data/time_df.csv')
            time_info = pd.DatetimeIndex(time_df.date).to_julian_date()
            waves_obs = np.loadtxt('data/wavelengths.txt')
        else:
            # spec_type_str == 'act' by default
            # Spectra without corrected values, considering activity
            spectra_values = np.load('data/spectra_active.npy')
            # Time information
            time_df = pd.read_csv('data/time_df.csv')
            time_info = pd.DatetimeIndex(time_df.date).to_julian_date()
            waves_obs = np.loadtxt('data/wavelengths.txt')
        return spectra_values, time_info, waves_obs


# Now we generate the temperature spectra given the full HARPS N flux spectra data.
for spec_type in spec_types:
    spectra_data, dates, waves_obs = load_spectra(spec_type)
    flux_master_obs = np.median(spectra_data, axis=0).reshape(1, len(waves_obs))
    flux_master_obs_err = np.median(spectra_err_val, axis=0).reshape(1, len(waves_obs))
    # Temp shells
    interpT = interp1d(waves_temp, temp_temp, fill_value='extrapolate')
    temp_waves_obs = interpT(waves_obs)
    temp_val = np.zeros_like(spectra_data)
    for i, spectrum in enumerate(spectra_data):
        temp_val[i, :], _ = interp_temp_given_flux(spectrum, flux_master_obs, temp_waves_obs)

    np.save(f'data/temp_{spec_type}.npy', temp_val)
    logger.info(f"Saving temperature values for {spec_type} spectra")
    # Now we filter kitkat kahled mask for flux and temperature
    spec = SpectrumData(wavelengths=waves_obs)
    _, filtered_index = spec.kitcat_filtering_mask(spectra_data, mask_dir=maskfile)
    np.savetxt('data/waves_kitcat.txt', waves_obs[filtered_index])
    np.save(os.path.join(large_data_dir, f'spectra_kitcat_{spec_type}.npy'), spectra_data[:, filtered_index])
    np.save(os.path.join(large_data_dir, f'temp_kitcat_{spec_type}.npy'), temp_val[:, filtered_index])
    logger.info(f"Saving filtered spectra and temperature for {spec_type} spectra")
