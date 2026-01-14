"""
-------------------------------------------------------
2025
by X. Dumusque, Y. Zhao and I. Gomez-Vargas 
-------------------------------------------------------

Cross-Correlation Function (CCF) tools for stellar spectra analysis.
Includes CCF template generation, Doppler shift computation,
and activity indicator extraction (RV, FWHM, BIS).
"""

import os
import sys
import subprocess
import numpy as np
import tempfile
from pathlib import Path
from scipy.interpolate import interp1d
from importlib import resources
from ..utils.logger_config import logger

# speed of light in m/s
c = 299792458.


def _ensure_fit_CCF_compiled():
    """
    Ensures that the C wrapper `fit_CCF.c` is compiled and importable.
    If compilation fails, users can set wrapper=False to use the C++ version.
    """
    try:
        with resources.path("doppleriann.physics.ccf_resources", "fit_CCF.c") as c_path:
            build_dir = c_path.parent
            so_name = f"fit_CCF.cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so"
            so_path = build_dir / so_name

            # Already compiled
            if so_path.exists():
                logger.info(f"Found compiled fit_CCF: {so_path.name}")
                return so_path

            # Try compilation
            logger.info(f"Compiling {c_path.name} ...")
            subprocess.run(
                [sys.executable, "setup_fit_CCF_PPP.py", "build_ext", "--inplace"],
                cwd=build_dir,
                check=True,
            )
            logger.info("fit_CCF compiled successfully.")
            return so_path

    except Exception as e:
        logger.warning(
            f"Failed to compile fit_CCF.c ({e}). "
            "You can use the C++ version instead:\n"
            "  1. Ensure GSL >= 2.6 is installed.\n"
            "  2. Compile manually:\n"
            "     g++ BIS_FIT2.cpp -o BIS_FIT2 `gsl-config --cflags --libs`\n"
            "  3. Then use wrapper=False in CCFcalculator()."
        )
        return None
    

class CCFcalculator:
    """
    Calculates cross-correlation functions (CCFs) from stellar spectra
    using a line mask template.

    Parameters
    ----------
    wrapper : bool, optional
        If True, runs in Python mode only. If False, uses compiled C++ code for BIS fitting.
    wavelimits : tuple, optional
        Lower and upper wavelength limits (default: 4000.0 - 6500.0 A).
    ccf_windows : int, optional
        Velocity range for the CCF (default: 10000 m/s).
    ccf_size : float, optional
        Step size in velocity space for the CCF (default: 125 m/s).
    map_fn : function, optional
        Map function (default: Python built-in map).
    """

    def __init__(self, wrapper=True, wavelimits=(4000.0, 6500.0), ccf_windows=10000, ccf_size=125.,
                 map_fn=map):
        self.wrapper = wrapper
        self.CCF_windows = ccf_windows
        self.CCF_size = ccf_size
        self.wavelength = np.arange(wavelimits[0], wavelimits[1], 0.005)
        self.vrad_ccf2 = np.arange(-1 * self.CCF_windows, self.CCF_windows + 1., self.CCF_size)

        # Load mask file
        mask_path = Path(__file__).parent / "ccf_resources" / "G2_mask.txt"
        if not mask_path.exists():
            raise FileNotFoundError(f"Required mask file not found: {mask_path}")
        templates = np.loadtxt(mask_path)

        step_size = 0.005
        wave_extend = 100.0

        freq_line = templates[:, 0]
        contrast_line = templates[:, 1]

        index_lines = (freq_line > self.wavelength.min()) & (freq_line < self.wavelength.max()) & (contrast_line > 0.1)
        freq_line_G2 = freq_line[index_lines]
        contrast_line_G2 = contrast_line[index_lines]

        no_points = int(wave_extend/step_size)
        wavelength_before = np.linspace(np.min(self.wavelength)-step_size*no_points,np.min(self.wavelength), no_points)
        wavelength_after = np.linspace(self.wavelength[-1], self.wavelength[-1]+step_size*no_points, no_points)
        wavelength_extend = np.concatenate((wavelength_before, self.wavelength, wavelength_after))

        self.mask_template = self.calculate_CCF_1(self.vrad_ccf2, self.wavelength, freq_line_G2, contrast_line_G2, wavelength_extend)

        # Compile or load wrapper
        if self.wrapper:
            so_path = _ensure_fit_CCF_compiled()
            if so_path and so_path.exists():
                sys.path.append(str(so_path.parent))
                import fit_CCF  # noqa: F401
                logger.info("fit_CCF wrapper loaded successfully.")
            else:
                logger.warning("Falling back to C++ version (wrapper=False).")
                self.wrapper = False

        # --- Handle the C++ fallback (BIS_FIT2.cpp) ---
        if not self.wrapper:
            try:
                with resources.path("doppleriann.physics.ccf_resources", "BIS_FIT2.cpp") as cpp_path:
                    exe_file = cpp_path.with_suffix('')  # removes .cpp → becomes 'BIS_FIT2'

                    if exe_file.exists():
                        logger.info(f"Using existing binary: {exe_file.name}")
                    else:
                        logger.info(f"Compiling BIS_FIT2.cpp → {exe_file}")
                        subprocess.run(
                            [
                                "bash",
                                "-c",
                                f"g++ '{cpp_path}' -o '{exe_file}' $(gsl-config --cflags --libs)"
                            ],
                            check=True,
                        )
                        logger.info("BIS_FIT2 compiled successfully.")
            except Exception as e:
                logger.error(
                    f"Failed to compile BIS_FIT2.cpp: {e}\n"
                    "Please ensure GSL ≥ 2.6 is installed and accessible via `gsl-config`."
                )


    def calculate_CCF_1(self, velocity_array, wavelength, wavelength_line, weight_line, wavelength_extend):
        """
        Builds a CCF mask template by shifting spectral lines across a velocity grid.

        Parameters
        ----------
        velocity_array : ndarray
            Array of velocity values (m/s).
        wavelength : ndarray
            Base wavelength grid.
        wavelength_line : ndarray
            Mask line wavelengths.
        weight_line : ndarray
            Mask line weights.
        wavelength_extend : ndarray
            Extended wavelength grid for interpolation.

        Returns
        -------
        mask_template : ndarray
            2D CCF mask array.
        """
        mask_template = np.zeros((velocity_array.size, wavelength.size))
        index_old_begin = 20000  # wavelength_before.size
        index_old_end = 20000  # wavelength_after.size
        begin_wave = wavelength_extend.min()

        for i in np.arange(velocity_array.size):
            wavelength_line_shift = wavelength_line * (1 + velocity_array[i] / c)
            mask_corr = self.calc_mask(wavelength_extend, wavelength_line_shift, weight_line, begin_wave)
            mask_corr = mask_corr[index_old_begin:-index_old_end]
            mask_template[i, :] = mask_corr

        return mask_template

    def calculate_CCF_2(self, spectrum):
        """
        Computes the CCF by correlating an observed spectrum with the precomputed mask template.

        Parameters
        ----------
        spectrum : ndarray
            Observed stellar spectrum on the defined wavelength grid.

        Returns
        -------
        CCF : ndarray
            Computed cross-correlation function.
        """
        # Transpose mask_template for efficient matrix multiplication
        mask_template_transposed = self.mask_template.T
        # Perform matrix multiplication between spectrum and transposed mask_template
        CCF = np.dot(spectrum, mask_template_transposed)
        return CCF

    def Delta_wavelength(self, v, wavelength0):
        """
        Converts a velocity shift into a wavelength difference using the relativistic Doppler formula.

        Parameters
        ----------
        v : float
            Velocity shift (m/s).
        wavelength0 : float
            Reference wavelength.

        Returns
        -------
        delta_wavelength : float
            Wavelength difference corresponding to the velocity shift.
        """
        beta = v / c
        delta_wavelength = wavelength0 * (np.sqrt((1 + beta) / (1 - beta)) - 1)
        return delta_wavelength

    def calc_mask(self, wavelength_extend, wavelength_line_shift, weight_line, begin_wave, mask_width=820, hole_width=0):
        """
        Constructs a line mask shifted in wavelength space for each velocity step.

        Parameters
        ----------
        wavelength_extend : ndarray
            Extended wavelength grid.
        wavelength_line_shift : ndarray
            Doppler-shifted line wavelengths.
        weight_line : ndarray
            Mask line weights.
        begin_wave : float
            Starting wavelength of the extended grid.
        mask_width : float, optional
            Width of mask regions (default: 820 m/s converted to wavelength units).
        hole_width : float, optional
            Optional hole width for mask gaps.

        Returns
        -------
        mask : ndarray
            1D array containing the generated CCF mask.
        """
        # Transform the width into wavelength space.
        hole_width = np.array([self.Delta_wavelength(mask_width, wavelength_line_shift[i]) for i in
                               np.arange(len(wavelength_line_shift))])

        begining_mask_hole = wavelength_line_shift - hole_width / 2.
        end_mask_hole = wavelength_line_shift + hole_width / 2.

        index_begining_mask_hole = []
        index_end_mask_hole = []

        freq_step_before_mask_hole = []
        freq_step_after_mask_hole = []
        bg_wave = wavelength_extend.min()

        for i in np.arange(len(wavelength_line_shift)):
            aa = int(np.ceil((begining_mask_hole[i] - begin_wave) / 0.005))
            bb = int(np.ceil((end_mask_hole[i] - begin_wave) / 0.005) - 1)

            index_begining_mask_hole.append(aa)
            index_end_mask_hole.append(bb)

            freq_step_before_mask_hole.append(wavelength_extend[aa] - wavelength_extend[aa - 1])
            freq_step_after_mask_hole.append(wavelength_extend[bb + 1] - wavelength_extend[bb])

        mask = np.zeros(wavelength_extend.size)
        a = np.array(index_begining_mask_hole)
        b = np.array(index_end_mask_hole)

        freq_step_before_mask_hole = np.array(freq_step_before_mask_hole)
        freq_step_after_mask_hole = np.array(freq_step_after_mask_hole)

        fraction_pixel_before_mask_hole = np.abs(wavelength_extend[a] - begining_mask_hole) / freq_step_before_mask_hole
        fraction_pixel_after_mask_hole = np.abs(wavelength_extend[b] - end_mask_hole) / freq_step_after_mask_hole

        for i in np.arange(a.size):
            mask[a[i]:b[i]] = [weight_line[i]] * (b[i] - a[i])
            mask[a[i] - 1] = weight_line[i] * fraction_pixel_before_mask_hole[i]
            mask[b[i]] = weight_line[i] * fraction_pixel_after_mask_hole[i]

        return mask

    def ccf_calculator(self, spectra, waves):
        """
        Computes the Cross-Correlation Function (CCF) for an input spectrum.

        Parameters
        ----------
        spectra : ndarray
            Input stellar spectrum.
        waves : ndarray
            Wavelength grid of the input spectrum.

        Returns
        -------
        If wrapper is True:
            ccf_data : ndarray
                Array containing velocity and CCF values.
            ccf_error : ndarray
                Estimated errors for the CCF.
        If wrapper is False:
            ccf_data : ndarray
            ccf_error : ndarray
            span_harps : float
                Line bisector inverse slope (BIS).
            vrad_harps : float
                Measured radial velocity.
            vrad_err : float
                Radial velocity uncertainty.
            fwhm_harps : float
                Full width at half maximum of the CCF.
            fwhm_err : float
                Uncertainty of FWHM.
        """
        nb_zeros_on_sides = 5
        period_appodisation = int(((len(self.vrad_ccf2)-1)/2.))
        len_appodisation = int(period_appodisation/2.)
        a = np.arange(len(self.vrad_ccf2))
        b = 0.5*np.cos(2*np.pi/period_appodisation*a-np.pi)+0.5
        appod = np.concatenate([np.zeros(nb_zeros_on_sides), b[:len_appodisation], np.ones(len(self.vrad_ccf2)-period_appodisation-2*nb_zeros_on_sides),b[:len_appodisation][::-1], np.zeros(nb_zeros_on_sides)])

        spec_func = interp1d(waves, spectra)
        spec_harps = spec_func(self.wavelength)

        CCF_active_sun = self.calculate_CCF_2(spec_harps)

        CCF_active_sun /=np.max(CCF_active_sun)
        CCF_active_sun  = 1-((-CCF_active_sun+1)*appod)

        ccf_data = np.array([self.vrad_ccf2, CCF_active_sun]).T
        ccf_error = np.ones(self.vrad_ccf2.size).T/self.vrad_ccf2.size/100

        if not self.wrapper:
            try:
                # Locate C++ binary
                with resources.path("doppleriann.physics.ccf_resources", "BIS_FIT2") as exe_path:
                    run_dir = exe_path.parent

                    # Write input files in that directory
                    np.savetxt(run_dir / "CCF_data.txt", ccf_data)
                    np.savetxt(run_dir / "CCF_error.txt", ccf_error)

                    # Run the binary inside its own directory
                    cmd = [str(exe_path), str(int(self.vrad_ccf2.size))]
                    result = subprocess.run(cmd, cwd=run_dir, check=False)

                    if result.returncode != 0:
                        print("BIS_FIT2 execution failed. Check GSL or binary permissions.")
                        return ccf_data, ccf_error, np.nan, np.nan, np.nan, np.nan, np.nan

                    # Read C++ output files
                    par_c_path = run_dir / "ccf_parameter.bin"
                    par_err_path = run_dir / "ccf_Error_parameter.bin"

                    if not (par_c_path.exists() and par_err_path.exists()):
                        print("Missing C++ output files. The CCF computation may have failed.")
                        return ccf_data, ccf_error, np.nan, np.nan, np.nan, np.nan, np.nan

                    par_c = np.fromfile(par_c_path, dtype="double")
                    par_err = np.fromfile(par_err_path, dtype="double")

                    span_harps = par_c[2]
                    vrad_harps = par_c[3]
                    fwhm_harps = par_c[4]
                    vrad_err = par_err[3]
                    fwhm_err = par_err[4]

                    return ccf_data, ccf_error, span_harps, vrad_harps, vrad_err, fwhm_harps, fwhm_err

            except Exception as e:
                print(f"C++ CCF computation failed: {e}")
                return ccf_data, ccf_error, np.nan, np.nan, np.nan, np.nan, np.nan
        
        else:
            return ccf_data, ccf_error

def astro_data(X, wave_cols, wrapper=True):
    """
    Extracts radial velocity (RV), FWHM, BIS, and their uncertainties from a set of spectra.

    Parameters
    ----------
    X : ndarray
        Input spectra array (n_samples, n_wavelengths).
    wave_cols : ndarray
        Wavelength grid for the spectra.
    wrapper : bool, optional
        Whether to use the Python-only mode (default: True).

    Returns
    -------
    astro_array : ndarray
        Array of shape (N, 5):
            - column 0: radial velocity (RV)
            - column 1: RV error
            - column 2: FWHM
            - column 3: FWHM error
            - column 4: BIS (bisector span)
    """
    ccfobj = CCFcalculator(wrapper=wrapper)
    astro_array = np.zeros((len(X), 5))

    for i in range(len(X)):

        if wrapper:
            import fit_CCF
            ccf_data, ccf_errors = ccfobj.ccf_calculator(X[i], wave_cols)
            fit = fit_CCF.gauss_bis(ccf_data[:, 0], ccf_data[:, 1], ccf_errors, np.ones(len(X[i])))
            # RV
            astro_array[i, 0] = fit[3]
            # RV error
            astro_array[i, 1] = fit[7]
            # FMWH
            astro_array[i, 2] = fit[4]
            # FMWH error
            astro_array[i, 3] = fit[8]
            # BIS
            astro_array[i, 4] = fit[10]

        else:
            ccf_data, ccf_error, span_harps, vrad_harps, vrad_err, fwhm_harps, fwhm_err = ccfobj.ccf_calculator(X[i], wave_cols)
            # RV
            astro_array[i, 0] = vrad_harps
            # RV error
            astro_array[i, 1] = vrad_err
            # FMWH
            astro_array[i, 2] = fwhm_harps
            # FMWH error
            astro_array[i, 3] = fwhm_err
            # BIS
            astro_array[i, 4] = span_harps

    return astro_array
