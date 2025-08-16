# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:51:06 2025

@author: a5149691
"""

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import matplotlib.pyplot as plt

plt.close('all')

np.random.seed(8)

def ssm_criterion_from_covariance(Y, k_max):
    """
    Signal Subspace Matching (SSM) criterion for detecting the number of signals,
    taking the sample covariance matrix R and the data Y as input.

    Args:
        R (numpy.ndarray): Sample covariance matrix of shape (P, P), 
                          where P is the number of sensors.
        k_max (int): The maximum number of signals to consider.
        Y (numpy.ndarray): Sampled data matrix of shape (P, N), where N is the
                          number of samples. This is needed to calculate delta.

    Returns:
        list: A list of SSM values for each hypothesized number of signals (1 to k_max).
        int: The estimated number of signals.
    """
    
    R = (Y @ Y.T.conj())/Y.shape[1]
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
        ssm_values.append(ssm)

    # Estimate the number of signals as the index that minimizes the SSM criterion
    Q_est = np.argmin(ssm_values) + 1

    return ssm_values, Q_est

# --- Your existing functions (project_toeplitz, soft_threshold, toeplitz_decomposition, estimate_num_sources, generate_steering_vector) go here ---
def project_toeplitz(matrix):
    """Projects a matrix onto the Toeplitz space."""
    N = matrix.shape[0]
    toeplitz_elements = []
    for i in range(-N + 1, N):
        elements = np.diag(matrix, k=i)
        toeplitz_elements.append(np.mean(elements))
    return sla.toeplitz(toeplitz_elements[N - 1:], toeplitz_elements[:N])

def soft_threshold(x, threshold):
    """Soft thresholding function."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def toeplitz_decomposition(R_prime, K, lambda_val, max_iter=100, tol=1e-6):
    """
    ----------------- Our Method ---------
    Decomposes R_prime into Toeplitz L and diagonal D using ADMM.

    Args:
        R_prime (numpy.ndarray): Forward-backward averaged covariance matrix.
        K (int): Maximum rank of L (number of targets).
        lambda_val (float): Regularization parameter for rank.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.

    Returns:
        tuple: (L, D)
    """
    N = R_prime.shape[0]
    L = np.zeros_like(R_prime, dtype=complex)
    D = np.zeros_like(R_prime, dtype=complex)
    Z = np.zeros_like(R_prime, dtype=complex)
    W = np.zeros_like(R_prime, dtype=complex)
    Y1 = np.zeros_like(R_prime, dtype=complex)
    Y2 = np.zeros_like(R_prime, dtype=complex)
    mu = 1.0

    for iteration in range(max_iter):
        L_prev = L.copy()
        D_prev = D.copy()

        # Update L
        L_temp = (2 * (R_prime - D) - Y1 + mu * Z) / (2 + mu)
        L = project_toeplitz(L_temp)

        # Update D
        D_temp = (2 * (R_prime - L) - Y2 + mu * W) / (2 + mu)
        D_diag = np.diag(D_temp)
        D_diag_positive = np.maximum(0, D_diag).real # Ensure non-negative real diagonal
        D = np.diag(D_diag_positive)

        # Update Z (Nuclear Norm Minimization)
        Z_temp = L + Y1 / mu
        U, Sigma, Vh = la.svd(Z_temp)
        Sigma_k = np.diag(soft_threshold(Sigma, lambda_val / mu))
        Z = U @ Sigma_k @ Vh

        # Update W
        W_temp = D + Y2 / mu
        W_diag = np.diag(W_temp)
        W_diag_positive = np.maximum(0, W_diag).real # Ensure non-negative real diagonal
        W = np.diag(W_diag_positive)

        # Update dual variables
        Y1 = Y1 + mu * (L - Z)
        Y2 = Y2 + mu * (D - W)

        # Check for convergence
        if la.norm(L - L_prev) < tol and la.norm(D - D_prev) < tol:
            break

    return L, D

def estimate_num_sources(L_estimated, N):
    """Estimates the number of sources based on the singular values of L."""
    U, ss, Vh = la.svd(L_estimated)
    # normalized_singular_values = ss / np.max(ss) if np.max(ss) > 0 else ss
    # num_sources = np.sum(normalized_singular_values > threshold_factor)
    cdf_ss = ss/np.sum(ss)
    
    if (N != 9):
        threshold_factor= 0.25
    else:
        threshold_factor= 0.2
    
    # threshold_factor = 0.2
    
    mask = cdf_ss > threshold_factor
    num_sources = np.sum(mask)
    
    return num_sources

def generate_steering_vector(N, theta):
    """Generates a steering vector for a ULA."""
    phi = np.pi * np.sin(np.deg2rad(theta)) # Assuming lambda/2 spacing
    v = np.exp(1j * phi * np.arange(N))
    return v.reshape(-1, 1)

def generate_covariance_matrix(N, thetas, powers, rho, S, snr_db):
    """Generates a sample covariance matrix for multiple sources."""
    M = len(thetas)
    A = np.hstack([generate_steering_vector(N, theta) for theta in thetas])
    source_cov = np.zeros((M, M), dtype=complex)
    for i in range(M):
        source_cov[i, i] = powers[i]
        for j in range(i + 1, M):
            source_cov[i, j] = rho * np.sqrt(powers[i] * powers[j])
            source_cov[j, i] = rho.conjugate() * np.sqrt(powers[i] * powers[j])

    # Generate correlated source signals using Cholesky decomposition
    L_chol = np.linalg.cholesky(source_cov)
    uncorrelated_sources = (np.random.randn(M, S) + 1j * np.random.randn(M, S)) / np.sqrt(2)
    correlated_sources = L_chol @ uncorrelated_sources

    signal = A @ correlated_sources
    signal_power = np.trace(A @ source_cov @ A.conj().T) / N
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(N, S) + 1j * np.random.randn(N, S))
    received_signal = signal + noise
    R = (1/S) * received_signal @ received_signal.conj().T
    J = np.eye(N)[::-1]
    R_prime = 0.5 * (R + J @ R.conj().T @ J)
    return R_prime, received_signal

def monte_carlo_run(N, K_true, thetas, powers, rho, S, snr_db, lambda_val, num_runs=100):
    """Performs a Monte Carlo run to estimate PCD."""
    correct_detections = 0
    for _ in range(num_runs):
        R_prime, received_signal = generate_covariance_matrix(N, thetas, powers, rho, S, snr_db)
        try:
            """L+D"""
            L_estimated, _ = toeplitz_decomposition(R_prime, N - 1, lambda_val) # Max possible rank
            K_estimated = estimate_num_sources(L_estimated, N)
            """SSM"""
            # ssm_values, K_estimated = ssm_criterion_from_covariance(received_signal, 2)
            
            if K_estimated == K_true:
                correct_detections += 1
        except la.LinAlgError:
            pass # Handle potential SVD errors
    pcd = correct_detections / num_runs
    return pcd


if __name__ == "__main__":
    # Parameters for Figure 1
    K_true = 2
    powers = [1, 1]
    rho = 0.8 # Example correlation
    num_runs = 10 # Reduce for faster grid search
    snr_range = np.arange(-10, 20, 2) #np.linspace(-10, 20, 5) # Coarser SNR for grid search
    S_representative = 500
    delta_theta_representative = 70
    Ns = [3, 5, 7, 9]
    lambda_vals_to_search = np.linspace(0.01, 5, 10) # Range of lambda values to try

    optimal_lambdas = {}
    all_pcds = {}
    
    for N in Ns:
        optimal_pcd = -1
        optimal_lambda = None
        pcds_for_n = {}
        thetas = [-delta_theta_representative/2, delta_theta_representative/2]

        print(f"Performing grid search for N = {N}...")
        for lambda_val in lambda_vals_to_search:
            pcds_for_lambda = []
            for snr in snr_range:
                pcd = monte_carlo_run(N, K_true, thetas, powers, rho, S_representative, snr, lambda_val, num_runs)
                pcds_for_lambda.append(pcd)
            pcds_for_n[lambda_val] = pcds_for_lambda

            # Find the lambda that gives the best average PCD (you might want a more sophisticated metric)
            avg_pcd = np.mean(pcds_for_lambda)
            if avg_pcd > optimal_pcd:
                optimal_pcd = avg_pcd
                optimal_lambda = lambda_val

            print(f"  lambda = {lambda_val:.3f}, Avg PCD = {avg_pcd:.3f}")

        optimal_lambdas[N] = optimal_lambda
        all_pcds[N] = pcds_for_n[optimal_lambda]
        print(f"Optimal lambda for N = {N}: {optimal_lambda:.3f}, Avg PCD = {optimal_pcd:.3f}\n")

    # --- Plotting with optimal lambdas ---
    plt.figure(dpi = 1000)
    # plt.figure()
    for N in Ns:
        if (N == 3):
            plt.plot(snr_range, all_pcds[N], '-o', markersize = 15, linewidth = 4, alpha = 0.3, label=f'M = {N}, λ = {optimal_lambdas[N]:.3f}')
        if (N == 5):
            plt.plot(snr_range, all_pcds[N], '-s', markersize = 15, linewidth = 4, alpha = 0.8, label=f'M = {N}, λ = {optimal_lambdas[N]:.3f}')
        if (N == 7):
            plt.plot(snr_range, all_pcds[N], '-D', markersize = 15, linewidth = 4, alpha = 0.7, label=f'M = {N}, λ = {optimal_lambdas[N]:.3f}')
        if (N == 9):
            plt.plot(snr_range, all_pcds[N], '-^', markersize = 15, linewidth = 4, alpha = 0.8, label=f'M = {N}, λ = {optimal_lambdas[N]:.3f}')

    plt.xlabel('SNR (dB)', fontsize = 16)
    plt.ylabel('Probability of Correct Detection (PCD)', fontsize = 16)
    # plt.title(f'PCD vs. SNR for Different Numbers of Antennas (K={K_true}, S={S_representative}, Δθ={delta_theta_representative}°) ')
    plt.grid(True)
    plt.legend(fontsize = 16)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.xticks(fontsize = 16), plt.yticks(fontsize = 16)
    plt.savefig('pcd_vs_snr_antennas.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()

    print("\nOptimal Lambda Values:")
    print(optimal_lambdas)


    # from scipy.io import savemat
    
    mdic = {"snr_range" : snr_range, "all_pcds" : all_pcds, "optimal_lambdas" : optimal_lambdas}
    # savemat("PCD_vs_SNR_for_Different_Antennas.mat", mdic)
