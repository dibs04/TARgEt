# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 13:23:06 2025

@author: a5149691
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# Simulation Parameters
n_antennas = 3
wavelength = 4  # cm
spacing = 2     # cm (lambda/2)
angles_deg = [-40, 40]  # Angles of the two targets in degrees
angles_rad = np.deg2rad(angles_deg)
n_snapshots = 500
snr_db = 10
correlation_coefficient = 1.0  # Correlation between the two sources

# Steering Vector Function for ULA
def steering_vector(angle_rad, n_antennas, wavelength, spacing):
    v = np.exp(-1j * 2 * np.pi * spacing / wavelength * np.sin(angle_rad) * np.arange(n_antennas))
    return v.reshape(-1, 1)

# Generate Steering Vectors
a1 = steering_vector(angles_rad[0], n_antennas, wavelength, spacing)
a2 = steering_vector(angles_rad[1], n_antennas, wavelength, spacing)
A = np.hstack((a1, a2))

# Generate Correlated Source Signals
s1 = np.random.randn(n_snapshots) + 1j * np.random.randn(n_snapshots)
s2_uncorrelated = np.random.randn(n_snapshots) + 1j * np.random.randn(n_snapshots)
s2 = correlation_coefficient * s1 + np.sqrt(1 - correlation_coefficient**2) * s2_uncorrelated
S = np.array([s1, s2])

# Generate Noise
noise_power = 10**(-snr_db / 10)
noise = np.sqrt(noise_power / 2) * (np.random.randn(n_antennas, n_snapshots) + 1j * np.random.randn(n_antennas, n_snapshots))

# Received Signal
y = A @ S + noise

# Calculate Sample Covariance Matrix (Ryy)
Ryy = (1 / n_snapshots) * (y @ y.conj().T)

# Perform Eigen Decomposition of Ryy
eigenvalues_Ryy = np.linalg.eigvalsh(Ryy)
eigenvalues_Ryy_sorted = np.sort(np.abs(eigenvalues_Ryy))[::-1]

# Forward-Backward Averaging
J = np.eye(n_antennas)[::-1]
Ryy_conj = np.conjugate(Ryy)
Ryy_fb = 0.5 * (Ryy + J @ Ryy_conj @ J)

# Perform Eigen Decomposition of Ryy_fb
eigenvalues_Ryy_fb = np.linalg.eigvalsh(Ryy_fb)
eigenvalues_Ryy_fb_sorted = np.sort(np.abs(eigenvalues_Ryy_fb))[::-1]

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(np.arange(1, n_antennas + 1), eigenvalues_Ryy_sorted, marker='o')
plt.title('Eigenvalues of Sample Covariance (Correlated Sources)')
plt.xlabel('Eigenvalue Index')
plt.ylabel('Eigenvalue Magnitude')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(np.arange(1, n_antennas + 1), eigenvalues_Ryy_fb_sorted, marker='x')
plt.title('Eigenvalues after Forward-Backward Averaging')
plt.xlabel('Eigenvalue Index')
plt.ylabel('Eigenvalue Magnitude')
plt.grid(True)

plt.tight_layout()
plt.show()

print("Eigenvalues of Sample Covariance (Correlated Sources) dB:", 10*np.log10(eigenvalues_Ryy_sorted))
print("Eigenvalues after Forward-Backward Averaging dB:", 10*np.log10(eigenvalues_Ryy_fb_sorted))