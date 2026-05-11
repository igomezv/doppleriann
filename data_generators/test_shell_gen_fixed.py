# ===============================================
#  Shells Generator – Controlled injections
# ===============================================
import os
import sys
import numpy as np
import pandas as pd
from itertools import product
import multiprocessing as mp

# --- DopplerIANN imports ---
from doppleriann.data.shell_generation import generate_data
from doppleriann.utils.logger_config import logger

# -----------------------------------------------
# SLURM array index → directory ID
# -----------------------------------------------
try:
    idx_data = int(sys.argv[1])
except (IndexError, ValueError):
    idx_data = 0

device_hpc = True
spec_type = "act"


def h5_exists(period, doppler_shift, spec_type, data_dir):
    """Return True if all expected HDF5 files for this combo already exist."""
    flux_file = os.path.join(data_dir, f"flux_PI{doppler_shift}_P{period}_{spec_type}.h5")
    temp_file = os.path.join(data_dir, f"temp_PI{doppler_shift}_P{period}_{spec_type}.h5")
    return os.path.exists(flux_file) and os.path.exists(temp_file)


# -----------------------------------------------
# Output directory
# -----------------------------------------------
outputdir_pfx = f"shells/{idx_data}"
data_dir = os.path.join("data", outputdir_pfx)
os.makedirs(data_dir, exist_ok=True)

logger.info("Starting controlled shell generation (idx_data = %s)", idx_data)

# -----------------------------------------------
# Base input directory
# -----------------------------------------------
if device_hpc:
    large_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "large_data"))
else:
    large_data_dir = "/media/isidro/data/large_data/harpn"

dates = None
waves_obs = None
flux_err = None
temp_err = None
spec_vals_flux = None
spec_vals_temp = None
flux_master = None
temp_master = None


def initialize_inputs():
    global dates, waves_obs, flux_err, temp_err
    global spec_vals_flux, spec_vals_temp, flux_master, temp_master

    time_df = pd.read_csv("data/time_df.csv")
    dates = pd.DatetimeIndex(time_df.date).to_julian_date()
    waves_obs = np.loadtxt("data/waves_kitcat.txt")

    flux_err = np.load(os.path.join(large_data_dir, "spectra_kitcat_or_err.npy"))
    temp_err = np.load(os.path.join(large_data_dir, "temp_kitcat_or_err.npy"))

    spec_vals_flux = np.load(os.path.join(large_data_dir, f"spectra_kitcat_{spec_type}.npy"))
    spec_vals_temp = np.load(os.path.join(large_data_dir, f"temp_kitcat_{spec_type}.npy"))

    flux_master = np.mean(spec_vals_flux, axis=0)
    temp_master = np.mean(spec_vals_temp, axis=0)

    logger.info(
        f"Loaded spectra for {spec_type}: flux={spec_vals_flux.shape}, temp={spec_vals_temp.shape}"
    )


# -----------------------------------------------
# Injection parameters
# -----------------------------------------------
n_reso = 9

periods = [
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
]

if idx_data == 0:
    doppler_shift_amplitudes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0, 2.0, 5.0]
else:
    doppler_shift_amplitudes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

param_combinations = list(product(periods, doppler_shift_amplitudes))


# ===============================================
# Worker function
# RETURNS: (period, amplitude, phase)
# ===============================================
def run_ref(params):
    period, doppler_shift = params

    # Skip if result already exists
    if h5_exists(period, doppler_shift, spec_type, data_dir):
        logger.info(f"Skipping existing P={period} K={doppler_shift}")
        return (period, doppler_shift, None)  # phase=None means skipped

    phase = np.random.uniform(0, 2 * np.pi)

    generate_data(
        num_it=0,
        spec_flux=spec_vals_flux,
        spec_temp=spec_vals_temp,
        spec_flux_master=flux_master,
        spec_temp_master=temp_master,
        flux_err_val=flux_err,
        temp_err_val=temp_err,
        waves_obs=waves_obs,
        dates=dates,
        data_dir=data_dir,
        n_reso=n_reso,
        spec_type=spec_type,
        random_shifts=False,
        period=period,
        doppler_shift=doppler_shift,
        phase_offset=phase,
    )

    return (period, doppler_shift, phase)


# ===============================================
# Main execution
# ===============================================
def main():
    initialize_inputs()
    logger.info(f"Running {len(param_combinations)} controlled injection combinations")

    n_cpus = max(1, mp.cpu_count() - 1)
    logger.info(f"Using {n_cpus} CPU cores...")

    # map returns list of return values from run_ref
    with mp.Pool(n_cpus) as pool:
        results = pool.map(run_ref, param_combinations)

    # Save (P, K, PHASE)
    txt_file = os.path.join(data_dir, "injection_phases.txt")
    with open(txt_file, "w") as f:
        f.write("# period_days   doppler_shift   phase_radians\n")
        for P, K, PH in results:
            if PH is not None:  # skipped ones have PH=None
                f.write(f"{P:6d}   {K:.3f}   {PH:.6f}\n")

    logger.info(f"Saved injection phase file: {txt_file}")


# ===============================================
# Entry point
# ===============================================
if __name__ == "__main__":
    import time

    t0 = time.time()
    main()
    tf = (time.time() - t0) / 60
    logger.info(f"Execution time: {tf:.2f} minutes")
