"""
-------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
-------------------------------------------------------

Tools to transform and process stellar spectra for DopplerIANN.
Includes methods to generate shell representations, apply Doppler shifts,
mask the spectra, or get the temperature given the flux through interpolation.

"""


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from ..utils.logger_config import logger


class SpectrumData:
    """
    Class providing spectral transformation utilities such as
    shell representation generation, Doppler shift injection,
    and spectral filtering.

    Parameters
    ----------
    wavelengths : array-like
        Array of wavelength values (in Angstroms or nanometers)
        corresponding to the input spectra.
    """
    def __init__(self, wavelengths):
        self.wavelengths = wavelengths
        # Speed of light in m/s
        self.c = 299792458.

    def shell_diagram(self, spec_input, master_spec, spec_err, n_reso,
                      scale_factor=1.0, num_limits_factor=0.0):
        """
        Generate a shell representation of the spectrum based on gradients
        and flux values.

        Parameters
        ----------
        spec_input : array
            Input observed spectrum.
        master_spec : array
            Reference or master spectrum.
        spec_err : array
            Spectrum uncertainties or noise estimates.
        n_reso : int
            Resolution of the shell map (number of bins along each axis).
        scale_factor : float, optional
            Scaling factor for the velocity gradient (default 1.0).
        num_limits_factor : float, optional
            Minimum fraction of valid points per bin (default 0.0).

        Returns
        -------
        shell_stats : dict
            Dictionary containing:
                - shell_df : pd.DataFrame
                    2D shell map of mean flux residuals.
                - density_map : np.ndarray
                    Density weighting map.
                - mean_grad_map : np.ndarray
                    Mean gradient per bin.
                - mean_wave_map : np.ndarray
                    Mean wavelength per bin.
        """
        c = self.c / scale_factor
        master = master_spec[1:].copy()
        spec = spec_input[1:].copy()
        spec_err = spec_err[1:].copy()
        wave = self.wavelengths[1:].copy()

        grad_lambda = np.gradient(master, wave)
        grad = wave * grad_lambda / c
        delta_obs = spec - master

        y_space = np.linspace(0.05, 0.95, n_reso + 1)
        grad_max = np.max(np.abs(grad))
        grad_space = np.linspace(-grad_max, grad_max, n_reso + 1)

        grad_bins = np.digitize(grad, grad_space)
        y_bins = np.digitize(spec, y_space)

        shell_map = np.zeros((n_reso, n_reso))
        density_map = np.zeros((n_reso, n_reso))
        mean_grad_map = np.zeros((n_reso, n_reso))
        mean_wave_map = np.zeros((n_reso, n_reso))

        num_limits = len(spec_input) * num_limits_factor

        for i_g in range(1, n_reso + 1):
            for i_f in range(1, n_reso + 1):
                index = (grad_bins == i_g) & (y_bins == i_f)
                valid_points = np.argwhere(index).size
                if valid_points > num_limits:
                    shell_map[i_f - 1, i_g - 1] = np.mean(delta_obs[index])
                    grad_space_sq = ((grad_space[i_g - 1] + grad_space[i_g]) / 2) ** 2
                    spec_err_sq = np.mean(spec_err[index]) ** 2 if valid_points > 0 else 0
                    density_map[i_f - 1, i_g - 1] = (grad_space_sq / spec_err_sq) * valid_points if spec_err_sq > 0 else 0
                    
                    # Store mean gradient and mean wavelength per bin.
                    mean_grad_map[i_f - 1, i_g - 1] = np.mean(grad[index])
                    mean_wave_map[i_f - 1, i_g - 1] = np.mean(wave[index])
                else:
                    shell_map[i_f - 1, i_g - 1] = 0
                    density_map[i_f - 1, i_g - 1] = 0
                    mean_grad_map[i_f - 1, i_g - 1] = 0
                    mean_wave_map[i_f - 1, i_g - 1] = 0

        y_m = (y_space[:-1] + y_space[1:]) / 2
        grad_m = (grad_space[:-1] + grad_space[1:]) / 2
        
        shell_df = pd.DataFrame(shell_map, columns=grad_m.round(2), index=np.flip(y_m.round(2)))

        # Package additional statistics
        shell_stats = {
            'shell_df': shell_df,
            'density_map': density_map,
            'mean_grad_map': mean_grad_map,
            'mean_wave_map': mean_wave_map,
        }

        return shell_stats


    def planet_inj(self, full_spec_data, dates, doppler_shift_amplitude,
                   period_days=20.0, reference_date=None, phase_offset=0):
        """
        Injects a planetary signal by applying a Doppler shift to the spectral time series given a period and semi amplitude.

        Parameters
        ----------
        full_spec_data : ndarray of shape (N, M)
            The full spectral time series (N spectra × M wavelengths).
        dates : ndarray of shape (N,)
            Observation times in Julian dates.
        doppler_shift_amplitude : float
            Semi-amplitude of the Doppler shift in m/s.
        period_days : float, optional
            Orbital period in days. Default is 20.
        reference_date : float, optional
            Reference Julian date for phase calculation. If None, uses the first date.
        phase_offset : float, optional
            Phase offset in radians applied to the sinusoidal signal. Default is 0.

        Returns
        -------
        injected_spectra : ndarray of shape (N, M)
            Spectra after applying Doppler shifts (planet injection).
        modulated_doppler_shifts : ndarray of shape (N,)
            Doppler shifts applied to each spectrum, in m/s.
        phase_values : ndarray of shape (N,)
            Phase values used to compute the Doppler shifts.
        """

        # Use custom reference date if provided, otherwise default to the first observation.
        if reference_date is None:
            reference_date = dates[0]

        phase_values = (2 * np.pi * (dates - reference_date) / period_days)
        # Apply a random phase shift to introduce variability
        phase_values = (phase_values + phase_offset) % (2 * np.pi)

        modulated_doppler_shifts = doppler_shift_amplitude * np.sin(phase_values)
    
        if len(full_spec_data) < len(modulated_doppler_shifts):
            modulated_doppler_shifts = modulated_doppler_shifts[:len(full_spec_data)]
            phase_values = phase_values[:len(full_spec_data)]

        master_spec = np.mean(full_spec_data, axis=0)
        gradient = np.gradient(master_spec, self.wavelengths)
        wave_factor = self.wavelengths / self.c
        injected_spectra = np.zeros_like(full_spec_data)

        for i, doppler_shift in enumerate(modulated_doppler_shifts):
            delta_flux = gradient * wave_factor * doppler_shift
            injected_spectra[i, :] = full_spec_data[i, :] + delta_flux

        return injected_spectra, modulated_doppler_shifts, phase_values
    
    
    def inject_random_doppler_shifts(self, full_spec_data, 
                                     doppler_shift_range=(0.05, 0.2), seed=None):
        """
        Apply random Doppler shifts to each spectrum independently.

        Parameters
        ----------
        full_spec_data : ndarray
            Spectral time series (N spectra x M wavelengths).
        doppler_shift_range : tuple of float, optional
            Min and max Doppler shifts (m/s).
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        injected_spectra : ndarray
            Spectra after random Doppler shifts.
        doppler_shifts : ndarray
            Applied Doppler shifts in m/s.
        phases_dummy : ndarray
            Placeholder zero phases (for compatibility).
        """
        if seed is not None:
            np.random.seed(seed)

        N, _ = full_spec_data.shape

        # Generate random Doppler shifts per spectrum
        doppler_shifts = np.random.uniform(doppler_shift_range[0], doppler_shift_range[1], size=N)

        # Dummy phases (for compatibility with other functions)
        phases_dummy = np.zeros_like(doppler_shifts)

        # Compute reference spectrum gradient
        master_spec = np.mean(full_spec_data, axis=0)
        gradient = np.gradient(master_spec, self.wavelengths)
        wave_factor = self.wavelengths / self.c

        # Apply Doppler shifts to each spectrum
        injected_spectra = np.zeros_like(full_spec_data)
        for i, v in enumerate(doppler_shifts):
            delta_flux = gradient * wave_factor * v
            injected_spectra[i, :] = full_spec_data[i, :] + delta_flux

        return injected_spectra, doppler_shifts, phases_dummy


    def kitcat_filtering_mask(self, full_spec_data, 
                              mask_dir='data/mask_kitcatkitcat_CCF_mask_Sun.npz'):
        """
        Apply a spectral mask based on the KITCAT CCF line mask.

        Parameters
        ----------
        full_spec_data : ndarray
            Input spectra to be filtered.
        mask_dir : str
            Path to the KITCAT CCF mask file (.npz).

        Returns
        -------
        filtered_flux_dataset : ndarray
            Spectra after applying the line mask.
        filtered_index : ndarray
            Indices of wavelengths that passed the mask filter.
        """
        mask_data = np.load(mask_dir)
        try:
            mask_wave = mask_data['mask_wave']
            mask_weight = mask_data['mask_weight']
            mask_left = mask_data['mask_left']
            mask_right = mask_data['mask_right']
        except:
            mask_wave   = mask_data['wave']
            mask_weight = mask_data['line_depth']
            mask_left   = mask_data['wave_left']
            mask_right  = mask_data['wave_right']
        
        filtered_flux_dataset = np.zeros_like(full_spec_data)
        final_mask = np.zeros_like(self.wavelengths, dtype=bool)  # Boolean mask for all wavelengths

        for idx, spectral_flux in enumerate(full_spec_data):
            filtered_flux = np.zeros_like(spectral_flux)
            for i in range(len(mask_wave)):
                mask_region = (self.wavelengths >= mask_left[i]) & (self.wavelengths <= mask_right[i])
                filtered_flux[mask_region] = spectral_flux[mask_region]  # * mask_weight[i] Commented this factor to avoid a weighted flux.
                final_mask |= mask_region 

            filtered_flux_dataset[idx, :] = filtered_flux
        # Get the indexes of the filtered wavelengths
        filtered_index = np.where(final_mask)[0]  

        return filtered_flux_dataset, filtered_index


def interp_temp_given_flux(target_flux, master_flux, temp_values):
    """
    Interpolate temperature values corresponding to target fluxes,
    restricted to monotonic segments of the reference flux array.

    Parameters
    ----------
    target_flux : ndarray
        Target flux values for interpolation.
    master_flux : ndarray
        Reference flux array (must contain monotonic chunks).
    temp_values : ndarray
        Temperature values corresponding to master_flux.

    Returns
    -------
    interpolated_temp : ndarray
        Interpolated temperature values.
    accepted_idx : ndarray of bool
        Boolean array indicating successfully interpolated points.
    """
    target_flux = target_flux.flatten()
    master_flux = master_flux.flatten()
    temp_values = temp_values.flatten()

    interpolated_temp = np.zeros_like(target_flux)
    accepted_idx = np.zeros_like(target_flux, dtype=bool) 

    monotonic_regions = []  # List of tuples (start_idx, end_idx) for monotonic chunks
    start_idx = 0

    for i in range(1, len(master_flux)):
        # Check if the sequence remains monotonic
        if (master_flux[i] > master_flux[i - 1] and master_flux[start_idx] <= master_flux[start_idx + 1]) or \
           (master_flux[i] < master_flux[i - 1] and master_flux[start_idx] >= master_flux[start_idx + 1]):
            continue  
        else:
            # End of the current monotonic chunk
            if i - start_idx > 1:  # Only consider chunks with more than 1 point
                monotonic_regions.append((start_idx, i - 1))
            start_idx = i 

    # Add the final chunk if it is monotonic
    if len(master_flux) - start_idx > 1:
        if (master_flux[start_idx] <= master_flux[start_idx + 1]):  # Increasing
            if all(master_flux[j] <= master_flux[j + 1] for j in range(start_idx, len(master_flux) - 1)):
                monotonic_regions.append((start_idx, len(master_flux) - 1))
        elif (master_flux[start_idx] >= master_flux[start_idx + 1]):  # Decreasing
            if all(master_flux[j] >= master_flux[j + 1] for j in range(start_idx, len(master_flux) - 1)):
                monotonic_regions.append((start_idx, len(master_flux) - 1))

    # Perform interpolation within monotonic regions
    cinterp = 0
    for start, end in monotonic_regions:
        interp_fn = interp1d(master_flux[start:end + 1], temp_values[start:end + 1]) 
        for i in range(start, end + 1):
            if master_flux[start] <= target_flux[i] <= master_flux[end] or (master_flux[start] >= target_flux[i] >= master_flux[end]):
                interpolated_temp[i] = interp_fn(target_flux[i])
                accepted_idx[i] = True 
                cinterp += 1
            else:
                # Exclude values outside the interpolation range
                interpolated_temp[i] = temp_values[i]  
                accepted_idx[i] = False
    
    interpolated_temp[~accepted_idx] = temp_values[~accepted_idx]
    return interpolated_temp, accepted_idx