# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 15:54:59 2025

@author: a5149691
"""

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import matplotlib.pyplot as plt

plt.close('all')

np.random.seed(40)

# --- Helper Functions (same as before) ---
def generate_steering_vector(N, theta):
    phi = np.pi * np.sin(np.deg2rad(theta))
    v = np.exp(1j * phi * np.arange(N))
    return v.reshape(-1, 1)

def generate_data_matrix(N, thetas, powers, rho, S, snr_db):
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

def toeplitz_matrix(c):
    N = len(c)
    r = c.conj()
    return sla.toeplitz(c, r)

def project_toeplitz(matrix):
    N = matrix.shape[0]
    toeplitz_elements = []
    for i in range(-N + 1, N):
        elements = np.diag(matrix, k=i)
        toeplitz_elements.append(np.mean(elements))
    return sla.toeplitz(toeplitz_elements[N - 1:], toeplitz_elements[:N])

# --- CMRA with ADMM ---
def cmra_solver_admm(R_hat, N, lambda_reg, rho_admm, max_iter=100, tol=1e-4):
    """Solves the CMRA optimization problem using ADMM."""

    # Initialize variables
    T = np.zeros_like(R_hat, dtype=complex)  # Toeplitz matrix
    U = np.zeros_like(R_hat, dtype=complex)  # Auxiliary variable
    Lambda = np.zeros_like(R_hat, dtype=complex)  # Lagrange multiplier

    for i in range(max_iter):
        T_old = T.copy()

        # Update T (Toeplitz matrix)
        T = project_toeplitz(U - Lambda / rho_admm)

        # Update U
        temp = T + Lambda / rho_admm
        U = (temp + temp.conj().T) / 2  # Ensure Hermitian
        U = U + lambda_reg / rho_admm * R_hat
        eigvals, eigvecs = la.eigh(U)
        eigvals = np.maximum(eigvals, 0)  # Project onto PSD cone
        U = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T

        # Update Lambda
        Lambda = Lambda + rho_admm * (T - U)

        # Check convergence
        if la.norm(T - T_old) < tol:
            break

    return T

def monte_carlo_run_cmra_admm(N, K_true, thetas, powers, rho, S, snr_db, lambda_reg_admm, num_runs=100):
    correct_detections = 0
    for _ in range(num_runs):
        Y = generate_data_matrix(N, thetas, powers, rho, S, snr_db)
        R_hat = (1 / S) * Y @ Y.conj().T
        
        J = np.eye(N)[::-1]
        R_hat = 0.5 * (R_hat + J @ R_hat.conj().T @ J)
        
        # W_hat = (1 / S) * R_hat.T @ R_hat  # Simplified W_hat -  Not needed for this ADMM implementation

        lambda_reg = lambda_reg_admm #0.1  # Regularization parameter - tune this
        rho_admm = 1.0  # ADMM penalty parameter - tune this

        try:
            T_hat = cmra_solver_admm(R_hat, N, lambda_reg, rho_admm)
            eigenvalues = np.linalg.eigvalsh(T_hat)
            eigenvalues.sort()
            lambda_max = eigenvalues[-1]
            kappa = 0.05 * lambda_max
            K_estimated = np.sum(eigenvalues > kappa)

            if K_estimated == K_true:
                correct_detections += 1
        except Exception as e:
            print(f"CMRA (ADMM) encountered an error: {e}")
            pass
    pcd = correct_detections / num_runs
    return pcd

# --- L+D and SSM (same as before) ---
def project_toeplitz(matrix):
    N = matrix.shape[0]
    toeplitz_elements = []
    for i in range(-N + 1, N):
        elements = np.diag(matrix, k=i)
        toeplitz_elements.append(np.mean(elements))
    return sla.toeplitz(toeplitz_elements[N - 1:], toeplitz_elements[:N])

def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def toeplitz_decomposition(R_prime, K, lambda_val, max_iter=100, tol=1e-6):
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
        L_temp = (2 * (R_prime - D) - Y1 + mu * Z) / (2 + mu)
        L = project_toeplitz(L_temp)
        D_temp = (2 * (R_prime - L) - Y2 + mu * W) / (2 + mu)
        D_diag = np.diag(D_temp)
        D_diag_positive = np.maximum(0, D_diag).real
        D = np.diag(D_diag_positive)
        Z_temp = L + Y1 / mu
        U, Sigma, Vh = la.svd(Z_temp)
        Sigma_k = np.diag(soft_threshold(Sigma, lambda_val / mu))
        Z = U @ Sigma_k @ Vh
        W_temp = D + Y2 / mu
        W_diag = np.diag(W_temp)
        W_diag_positive = np.maximum(0, W_diag).real
        W = np.diag(W_diag_positive)
        Y1 = Y1 + mu * (L - Z)
        Y2 = Y2 + mu * (D - W)
        if la.norm(L - L_prev) < tol and la.norm(D - D_prev) < tol:
            break
    return L, D

def estimate_num_sources_ld(L_estimated, N):
    """Estimates the number of sources based on the singular values of L."""
    U, ss, Vh = la.svd(L_estimated)
    # normalized_singular_values = ss / np.max(ss) if np.max(ss) > 0 else ss
    # num_sources = np.sum(normalized_singular_values > threshold_factor)
    cdf_ss = ss/np.sum(ss)
    
    if (N != 9):
        threshold_factor= 0.25
    else:
        threshold_factor= 0.35
    
    mask = cdf_ss > threshold_factor
    num_sources = np.sum(mask)
    
    return num_sources

def monte_carlo_run_ld(N, K_true, thetas, powers, rho, S, snr_db, lambda_val, num_runs=100):
    correct_detections = 0
    for _ in range(num_runs):
        Y = generate_data_matrix(N, thetas, powers, rho, S, snr_db)
        R = (1/S) * Y @ Y.conj().T
        J = np.eye(N)[::-1]
        R_prime = 0.5 * (R + J @ R.conj().T @ J)
        try:
            L_estimated, _ = toeplitz_decomposition(R_prime, N - 1, lambda_val)
            K_estimated = estimate_num_sources_ld(L_estimated, N)
            if K_estimated == K_true:
                correct_detections += 1
        except la.LinAlgError:
            pass
    pcd = correct_detections / num_runs
    return pcd

def ssm_criterion_from_covariance(R, k_max, Y):
    P = R.shape[0]
    N_samples = Y.shape[1]
    eigenvalues, eigenvectors = np.linalg.eig(R)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    lambda_mean = np.mean(eigenvalues)
    sigma_lambda = np.sqrt(np.mean((eigenvalues - lambda_mean) ** 2))
    delta = (1 / (np.sqrt(P) * sigma_lambda)) * np.trace(Y.conj().T @ Y)
    I_N = np.eye(N_samples)
    Y_H_Y = Y.conj().T @ Y
    P_Y = Y @ np.linalg.inv(Y_H_Y + delta * I_N) @ Y.conj().T
    ssm_values = []
    for k in range(1, k_max + 1):
        U_k = eigenvectors[:, :k]
        P_U_k = U_k @ U_k.conj().T
        ssm = np.trace(P_Y) + np.trace(P_U_k) - 2 * np.trace(P_Y @ P_U_k)
        ssm_values.append(ssm.real)
    Q_est = np.argmin(ssm_values) + 1
    return ssm_values, Q_est

def monte_carlo_run_ssm(N, K_true, thetas, powers, rho, S, snr_db, num_runs=100):
    correct_detections = 0
    for _ in range(num_runs):
        Y = generate_data_matrix(N, thetas, powers, rho, S, snr_db)
        R = (1/S) * Y @ Y.conj().T
        _, K_estimated = ssm_criterion_from_covariance(R, N - 1, Y)
        if K_estimated == K_true:
            correct_detections += 1
    pcd = correct_detections / num_runs
    return pcd

# --- Main Execution ---
if __name__ == "__main__":
    # Parameters
    N = 3
    K_true = 2
    powers = [1, 1]
    rho = 0.8
    S = 200
    delta_theta = 60
    thetas = [-delta_theta / 2, delta_theta / 2]
    snr_range = np.arange(-10, 20, 2) # np.linspace(-10, 20, 15)
    num_runs = 500

    # Parameter tuning
    lambda_val_ld = 10  # L+D lambda
    # eta_cmra = 1e-2  # CMRA eta (for CVXOPT version)
    lambda_reg_admm = 10 # CMRA ADMM lambda
    rho_admm = 1.0 # CMRA ADMM rho

    pcds_ld = [monte_carlo_run_ld(N, K_true, thetas, powers, rho, S, snr, lambda_val_ld, num_runs) for snr in snr_range]
    pcds_ssm = [monte_carlo_run_ssm(N, K_true, thetas, powers, rho, S, snr, num_runs) for snr in snr_range]
    pcds_cmra_admm = [monte_carlo_run_cmra_admm(N, K_true, thetas, powers, rho, S, snr, lambda_reg_admm, num_runs) for snr in snr_range]

    # Plotting
    plt.figure(dpi = 1000)
    plt.plot(snr_range, pcds_ld, label=f'TARgEt (M={N}, λ={lambda_val_ld:.3f})', marker='^', markersize = 15, linewidth = 4, alpha = 0.6)
    plt.plot(snr_range, pcds_ssm, label=f'SSM (M={N})', marker='s', markersize = 15, linewidth = 4, alpha = 0.7)
    plt.plot(snr_range, pcds_cmra_admm, label=f'CMRA-ADMM (M={N})', marker='o', markersize = 15, linewidth = 4, alpha = 0.7)

    plt.xlabel('SNR (dB)', fontsize = 16)
    plt.ylabel('Probability of Correct Detection (PCD)', fontsize = 16)
    # plt.title(f'PCD vs. SNR (N={N}, K={K_true}, S={S}, Δθ={delta_theta})')
    plt.grid(True)
    plt.legend(fontsize = 16)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.xticks(fontsize = 16), plt.yticks(fontsize = 16)
    plt.savefig('pcd_vs_snr_ssm_StSD.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()
    
    # from scipy.io import savemat
    
    mdict = {"snr_range" : snr_range, "TARgEt_PCD_lambda_10_M_3" : pcds_ld, "SSM_PCD_M_3" : pcds_ssm, "CMRA_ADMM_PCD_M_3" : pcds_cmra_admm}
    
    # savemat("PCD_vs_SNR_for_Different_Methods.mat", mdict)