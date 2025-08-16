import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import matplotlib.pyplot as plt

plt.close('all')

np.random.seed(40)

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
        D_diag_positive = np.maximum(0, D_diag)
        D = np.diag(D_diag_positive)

        # Update Z (Nuclear Norm Minimization)
        Z_temp = L + Y1 / mu
        U, Sigma, Vh = la.svd(Z_temp)
        Sigma_k = np.diag(soft_threshold(Sigma, lambda_val / mu))
        Z = U @ Sigma_k @ Vh

        # Update W
        W_temp = D + Y2 / mu
        W_diag = np.diag(W_temp)
        W_diag_positive = np.maximum(0, W_diag)
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
        threshold_factor= 0.9 #0.35 # has to be 0.20
    
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

    signal_cov = A @ source_cov @ A.conj().T
    signal_power = np.trace(signal_cov) / N
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(N, S) + 1j * np.random.randn(N, S))
    received_signal = A @ np.sqrt(np.diag(powers)) @ (np.random.randn(M, S) + 1j * np.random.randn(M, S)) + noise
    R = (1/S) * received_signal @ received_signal.conj().T
    J = np.eye(N)[::-1]
    R_prime = 0.5 * (R + J @ R.conj().T @ J)
    return R_prime


def monte_carlo_run(N, K_true, thetas, powers, rho, S, snr_db, lambda_val, num_runs=10):
    """Performs a Monte Carlo run to estimate PCD."""
    correct_detections = 0
    for _ in range(num_runs):
        R_prime = generate_covariance_matrix(N, thetas, powers, rho, S, snr_db)
        try:
            L_estimated, _ = toeplitz_decomposition(R_prime, N - 1, lambda_val) # Max possible rank
            K_estimated = estimate_num_sources(L_estimated, N)
            if K_estimated == K_true:
                correct_detections += 1
        except la.LinAlgError:
            pass # Handle potential SVD errors

    pcd = correct_detections / num_runs
    return pcd

if __name__ == "__main__":
    # Common parameters
    K_true = 2
    powers = [1, 1]
    rho = 0.8 # Uncorrelated sources for simplicity
    # lambda_val = 1
    num_runs = 500
    snr_range = np.arange(-10, 20, 2) # np.linspace(-10, 20, 15)
    S_representative = 500
    delta_theta_representative = 60
    
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['font.family'] = 'serif'
        

    ### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # # Figure 2: PCD vs. SNR for Different S
    Ss = [200, 400, 600, 800]
    pcds_vs_snr_s = []
    plt.figure(dpi = 1000)
    # plt.figure()
    N_representative = 3
    lambda_val = 6
    thetas = [-delta_theta_representative//2, delta_theta_representative//2]
    for S in Ss:
        pcds = [monte_carlo_run(N_representative, K_true, thetas, powers, rho, S, snr, lambda_val, num_runs) for snr in snr_range]
        pcds_vs_snr_s.append(pcds)
        if (S == 200):
            plt.plot(snr_range, pcds, '-o', markersize = 15, linewidth = 4, alpha = 0.3, label=f'Q = {S}')
        if (S == 400):
            plt.plot(snr_range, pcds, '-s', markersize = 15, linewidth = 4, alpha = 0.8, label=f'Q = {S}')
        if (S == 600):
            plt.plot(snr_range, pcds, '-D', markersize = 15, linewidth = 4, alpha = 0.7, label=f'Q = {S}')
        if (S == 800):
            plt.plot(snr_range, pcds, '-^', markersize = 15, linewidth = 4, alpha = 0.8, label=f'Q = {S}')
    plt.xlabel('SNR (dB)', fontsize = 16)
    plt.ylabel('Probability of Correct Detection (PCD)', fontsize = 16)
    # plt.title(f'PCD vs. SNR for Different Numbers of Snapshots (N={N_representative}, K={K_true}, Δθ={delta_theta_representative}°) ')
    plt.grid(True)
    plt.legend(fontsize = 16)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.xticks(fontsize = 16), plt.yticks(fontsize = 16)
    plt.savefig('pcd_vs_snr_s.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()
    
    # from scipy.io import savemat
    
    mdict = {"snr_range" : snr_range, "T_200" : pcds_vs_snr_s[0], "T_400" : pcds_vs_snr_s[1], "T_600" : pcds_vs_snr_s[2], "T_800" : pcds_vs_snr_s[3]}
    
    # savemat("PCD_vs_SNR_for_Different_Snapshots.mat", mdict)    
    
    ### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    ### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Figure 3: PCD vs. SNR for Different Delta Theta
    # delta_thetas = np.arange(25, 80, 10)
    # pcds_vs_snr_dtheta = []
    # plt.figure(dpi = 100)
    # # plt.figure()
    # N_representative = 3
    # S_representative = 200
    # for delta_theta in delta_thetas:
    #     thetas = [-delta_theta//2, delta_theta//2]
    #     pcds = [monte_carlo_run(N_representative, K_true, thetas, powers, rho, S_representative, snr, lambda_val, num_runs) for snr in snr_range]
    #     pcds_vs_snr_dtheta.append(pcds)
    #     if (delta_theta == 25):
    #         plt.plot(snr_range, pcds, '-H', markersize = 15, linewidth = 4, alpha = 1, label=f'Δθ = {delta_theta}°')
    #     if (delta_theta == 35):
    #         plt.plot(snr_range, pcds, '-P', markersize = 15, linewidth = 4, alpha = 0.8, label=f'Δθ = {delta_theta}°')
    #     if (delta_theta == 45):
    #         plt.plot(snr_range, pcds, '-o', markersize = 15, linewidth = 4, alpha = 0.3, label=f'Δθ = {delta_theta}°')
    #     if (delta_theta == 55):
    #         plt.plot(snr_range, pcds, '-s', markersize = 15, linewidth = 4, alpha = 0.8, label=f'Δθ = {delta_theta}°')
    #     if (delta_theta == 65):
    #         plt.plot(snr_range, pcds, '-D', markersize = 15, linewidth = 4, alpha = 0.7, label=f'Δθ = {delta_theta}°')
    #     if (delta_theta == 75):
    #         plt.plot(snr_range, pcds, '-^', markersize = 15, linewidth = 4, alpha = 0.8, label=f'Δθ = {delta_theta}°')
    # plt.xlabel('SNR (dB)', fontsize = 16)
    # plt.ylabel('Probability of Correct Detection (PCD)', fontsize = 16)
    # # plt.title(f'PCD vs. SNR for Different Angle Separations (N={N_representative}, K={K_true}, S={S_representative})')
    # plt.grid(True)
    # plt.legend(fontsize = 16)
    # plt.ylim(0, 1.05)
    # plt.tight_layout()
    # plt.xticks(fontsize = 16), plt.yticks(fontsize = 16)
    # plt.savefig('pcd_vs_snr_dtheta.pdf', format = 'pdf', bbox_inches = 'tight')
    # plt.show()
    
    # from scipy.io import savemat
    
    # mdict = {"snr_range" : snr_range, "delta_thetas" : delta_thetas, "delat_theta_25" : pcds_vs_snr_dtheta[0], "delta_theta_35" : pcds_vs_snr_dtheta[1], "delta_theta_45" : pcds_vs_snr_dtheta[2], "delta_theta_55" : pcds_vs_snr_dtheta[3], "delta_theta_65" : pcds_vs_snr_dtheta[4], "delta_theta_75" : pcds_vs_snr_dtheta[5]}
    
    # savemat("PCD_vs_SNR_for_Different_DeltaThetas.mat", mdict)
    ### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%