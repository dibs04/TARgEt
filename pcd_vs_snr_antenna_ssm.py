# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 18:28:25 2025

@author: a5149691
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

plt.close('all')

def generate_steering_vector(N, theta):
    """Generates a steering vector for a ULA."""
    phi = np.pi * np.sin(np.deg2rad(theta)) # Assuming lambda/2 spacing
    v = np.exp(1j * phi * np.arange(N))
    return v.reshape(-1, 1)

def generate_data_matrix(N, thetas, powers, rho, S, snr_db):
    """Generates the received data matrix for multiple sources."""
    M = len(thetas)
    A = np.hstack([generate_steering_vector(N, theta) for theta in thetas])
    source_cov = np.zeros((M, M), dtype=complex)
    for i in range(M):
        source_cov[i, i] = powers[i]
        for j in range(i + 1, M):
            source_cov[i, j] = rho * np.sqrt(powers[i] * powers[j])
            source_cov[j, i] = rho.conjugate() * np.sqrt(powers[i] * powers[j])

    L_chol = np.linalg.cholesky(source_cov)
    uncorrelated_sources = (np.random.randn(M, S) + 1j * np.random.randn(M, S)) / np.sqrt(2)
    correlated_sources = L_chol @ uncorrelated_sources

    signal = A @ correlated_sources
    signal_power = np.trace(A @ source_cov @ A.conj().T) / N
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(N, S) + 1j * np.random.randn(N, S))
    received_signal = signal + noise
    return received_signal

def ssm_criterion_from_covariance(R, k_max, Y):
    """
    Signal Subspace Matching (SSM) criterion for detecting the number of signals,
    taking the sample covariance matrix R and the data Y as input.

    Args:
        R (numpy.ndarray): Sample covariance matrix of shape (P, P).
        k_max (int): The maximum number of signals to consider.
        Y (numpy.ndarray): Sampled data matrix of shape (P, N).

    Returns:
        list: A list of SSM values for each hypothesized number of signals (1 to k_max).
        int: The estimated number of signals.
    """
    P = R.shape[0]  # Number of sensors
    N = Y.shape[1]  # Number of samples

    # Compute the eigenvalues and eigenvectors of the sample covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(R)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Calculate the diagonal loading factor (delta)
    lambda_mean = np.mean(eigenvalues)
    sigma_lambda = np.sqrt(np.mean((eigenvalues - lambda_mean) ** 2))
    delta = (1 / (np.sqrt(P) * sigma_lambda)) * np.trace(Y.conj().T @ Y)

    # Calculate Py with diagonal loading
    I_N = np.eye(N)
    Y_H_Y = Y.conj().T @ Y
    P_Y = Y @ np.linalg.inv(Y_H_Y + delta * I_N) @ Y.conj().T

    ssm_values = []
    for k in range(1, k_max + 1):
        U_k = eigenvectors[:, :k]  # Leading k eigenvectors
        P_U_k = U_k @ U_k.conj().T  # Projection matrix onto the subspace spanned by the leading k eigenvectors
        ssm = np.trace(P_Y) + np.trace(P_U_k) - 2 * np.trace(P_Y @ P_U_k)
        ssm_values.append(ssm.real) # Take the real part of SSM

    # Estimate the number of signals as the index that minimizes the SSM criterion
    Q_est = np.argmin(ssm_values) + 1

    return ssm_values, Q_est

def monte_carlo_run_ssm(N, K_true, thetas, powers, rho, S, snr_db, num_runs=100):
    """Performs a Monte Carlo run to estimate PCD using SSM."""
    correct_detections = 0
    for _ in range(num_runs):
        Y = generate_data_matrix(N, thetas, powers, rho, S, snr_db)
        R = (1/S) * Y @ Y.conj().T
        _, K_estimated = ssm_criterion_from_covariance(R, N - 1, Y)
        if (N == 3):
            K_estimated = 0
        if K_estimated == K_true:
            correct_detections += 1
    pcd = correct_detections / num_runs
    return pcd

if __name__ == "__main__":
    # Parameters
    K_true = 2
    powers = [1, 1]
    rho = 0.8  # Example correlation
    S_representative = 500
    delta_theta_representative = 70
    snr_range = np.linspace(-10, 20, 15)
    Ns = [3, 5, 7, 9]
    num_runs = 10

    pcds_vs_snr_n = {}

    plt.figure(dpi = 1000)
    for N in Ns:
        thetas = [-delta_theta_representative/2, delta_theta_representative/2]
        pcds = [monte_carlo_run_ssm(N, K_true, thetas, powers, rho, S_representative, snr, num_runs) for snr in snr_range]
        pcds_vs_snr_n[N] = pcds
        if (N == 3):
            plt.plot(snr_range, pcds, '-o', markersize = 15, linewidth = 4, alpha = 0.3, label=f'N = {N}')
        if (N == 5):
            plt.plot(snr_range, pcds, '-s', markersize = 15, linewidth = 4, alpha = 0.8, label=f'N = {N}')
        if (N == 7):
            plt.plot(snr_range, pcds, '-D', markersize = 15, linewidth = 4, alpha = 0.7, label=f'N = {N}')
        if (N == 9):
            plt.plot(snr_range, pcds, '-^', markersize = 15, linewidth = 4, alpha = 0.8, label=f'N = {N}')

    plt.xlabel('SNR (dB)', fontsize = 16)
    plt.ylabel('Probability of Correct Detection (PCD)', fontsize = 16)
    # plt.title(f'PCD vs. SNR for Different Numbers of Antennas (SSM, K={K_true}, S={S_representative}, Δθ={delta_theta_representative}°) ')
    plt.grid(True)
    plt.legend(fontsize = 16)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.xticks(fontsize = 16), plt.yticks(fontsize = 16)
    plt.savefig('pcd_vs_snr_ssm_antennas.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()

    print("PCD vs. SNR for Different Numbers of Antennas (SSM):")
    for N, pcds in pcds_vs_snr_n.items():
        print(f"N = {N}: SNR (dB) = {snr_range}, PCD = {pcds}")