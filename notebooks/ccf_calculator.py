# ---
# Imports
# ---

import numpy as np
import matplotlib.pyplot as plt
from doppleriann.physics.CCFcalculator import CCFcalculator

# ---
# Generate mock spectrum
# ---

# Wavelength grid (Ångström)
wavelength = np.linspace(5000, 6000, 2000)

# Create a simple Gaussian absorption line at 5500 Å
line_center = 5500.0
depth = 0.1
sigma = 0.5

spectrum = 1.0 - depth * np.exp(-0.5 * ((wavelength - line_center) / sigma) ** 2)

# Add mild noise to make it more realistic
noise = np.random.normal(0, 0.001, size=wavelength.shape)
spectrum_noisy = spectrum + noise

# Plot synthetic spectrum
plt.figure(figsize=(8, 3))
plt.plot(wavelength, spectrum_noisy, "k", lw=1)
plt.title("Mock Stellar Spectrum with a Gaussian Absorption Line")
plt.xlabel("Wavelength (Å)")
plt.ylabel("Normalized Flux")
plt.show()

# ---
# Compute CCF with and without wrapper
# ---

print("Running CCF (wrapper=False mode)...")
ccf_py = CCFcalculator(wrapper=False, wavelimits=(wavelength.min(), wavelength.max()))
result_py = ccf_py.ccf_calculator(spectrum_noisy, wavelength)
ccf_data_py, ccf_err_py = result_py[0], result_py[1]

print("Running CCF (C wrapper mode)...")
ccf_cpp = CCFcalculator(wrapper=True, wavelimits=(wavelength.min(), wavelength.max()))
result_cpp = ccf_cpp.ccf_calculator(spectrum_noisy, wavelength)
ccf_data_cpp, ccf_err_cpp = result_cpp[0], result_cpp[1]

# ---
# Plot comparison
# ---

plt.figure(figsize=(8, 4))
plt.plot(ccf_data_py[:, 0] / 1000, ccf_data_py[:, 1], "b-", lw=2, label="wrapper=False")
plt.plot(
    ccf_data_cpp[:, 0] / 1000, ccf_data_cpp[:, 1], "r--", lw=1.5, label="C Wrapper (fit_CCF.so)"
)
plt.xlabel("Velocity (km/s)")
plt.ylabel("CCF")
plt.title("CCF Comparison — Python vs. C Wrapper")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ---
# Inspect outputs
# ---

print(
    f"wrapper=False → velocity range: {ccf_data_py[:, 0].min() / 1000:.1f} to {ccf_data_py[:, 0].max() / 1000:.1f} km/s"
)
print(f"Mean CCF value (wrapper=False): {ccf_data_py[:, 1].mean():.4f}")
print(f"Mean CCF value (C wrapper): {ccf_data_cpp[:, 1].mean():.4f}")
