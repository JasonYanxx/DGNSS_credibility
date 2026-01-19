"""
Diagnostic System for DGNSS Monitoring Network (Commissioning Protocol)
Strictly implements the 'Block Averaging -> TIM Framework' pipeline.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from lib.algorithm_comparison import NCI_NLL_ES_Algorithm, AlgorithmResult
from scipy.stats import chisquare, chi2, kstest

class BlockAverager:
    """
    Stage 1: Block Averaging
    Decorrelates colored noise by averaging over time windows.
    """
    def __init__(self, T_batch: float = 30.0, f_hz: float = 1.0):
        self.M = int(f_hz * T_batch)  # Block size
        
    def process(self, error_vecs: List[np.ndarray], cov_matrices: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute block means and approximate block covariances.
        
        Args:
            error_vecs: List of (3,) position error vectors
            cov_matrices: List of (3,3) covariance matrices
            
        Returns:
            block_means: (K, 3) array of averaged errors
            block_covs: (K, 3, 3) array of averaged covariances (scaled by 1/M)
            block_indices: (K,) array of indices (center of block)
        """
        n_samples = len(error_vecs)
        n_blocks = n_samples // self.M
        
        if n_blocks == 0:
            return np.array([]), np.array([]), np.array([])
            
        block_means = []
        block_covs = []
        block_indices = []
        
        for k in range(n_blocks):
            start_idx = k * self.M
            end_idx = (k + 1) * self.M
            
            # Extract block
            batch_errors = np.array(error_vecs[start_idx:end_idx]) # (M, 3)
            batch_covs = np.array(cov_matrices[start_idx:end_idx]) # (M, 3, 3)
            
            # Compute Mean
            y_k = np.mean(batch_errors, axis=0)
            
            # Compute Covariance: R_batch approx (1/M) * mean(Sigma)
            # This is valid for the "Whitened" noise assumption in the low-freq domain.
            R_k = np.mean(batch_covs, axis=0) / self.M
            
            block_means.append(y_k)
            block_covs.append(R_k)
            block_indices.append(start_idx + self.M // 2)
            
        return np.array(block_means), np.array(block_covs), np.array(block_indices)


class BlockAveragerEffective:
    """
    Stage 1: Block Averaging with Effective Sample Size Correction
    
    Corrects block covariance using effective sample size (N_eff) to account
    for correlated noise (multipath, AR(1) processes).
    """
    def __init__(self, T_batch: float = 30.0, f_hz: float = 1.0, 
                 method: str = 'spectral_scaling', kappa: float = 0.5):
        """
        Initialize block averager with effective sample size correction.
        
        Args:
            T_batch: Block duration in seconds
            f_hz: Sampling frequency in Hz
            method: Correction method ('effective_size' or 'spectral_scaling')
            kappa: Power exponent for spectral scaling method (default 0.5, range [0.5, 0.8])
        """
        self.M = int(f_hz * T_batch)  # Block size
        self.method = method
        self.kappa = kappa
        
    def compute_lag1_autocorrelation(self, residuals: np.ndarray) -> float:
        """
        Compute Lag-1 autocorrelation coefficient rho for AR(1) process.
        
        Args:
            residuals: Array of residuals in the block (N, d) where d is dimension
            
        Returns:
            rho: Lag-1 autocorrelation coefficient (scalar or array)
        """
        if len(residuals) < 2:
            return 0.0
        
        # Center the residuals
        res_centered = residuals - np.mean(residuals, axis=0)
        
        # For multi-dimensional case, compute correlation per dimension and average
        # Or compute based on magnitude for simplicity
        if len(residuals.shape) == 2 and residuals.shape[1] > 1:
            # Use magnitude for multi-dimensional case
            res_mag = np.linalg.norm(res_centered, axis=1)
            res_centered = res_mag[:, np.newaxis]
        
        # Compute Lag-1 correlation
        if len(res_centered.shape) == 2 and res_centered.shape[1] == 1:
            res_centered = res_centered.flatten()
        
        numerator = np.dot(res_centered[:-1], res_centered[1:])
        denominator = np.dot(res_centered, res_centered)
        
        if denominator == 0:
            return 0.0
        
        rho = numerator / denominator
        
        # Limit rho to prevent numerical issues (0 <= rho < 1)
        rho = max(0.0, min(0.99, rho))
        
        return float(rho)
    
    def compute_effective_variance(self, residuals: np.ndarray, raw_variance: np.ndarray) -> np.ndarray:
        """
        Compute corrected block variance using effective sample size.
        
        For AR(1) process with correlation rho, the variance of the mean is:
        Var(mean) = (sigma^2 / N) * VIF
        where VIF = (1 + rho) / (1 - rho)
        
        Therefore, N_eff = N * (1 - rho) / (1 + rho)
        and corrected variance = raw_variance / N_eff = (raw_variance / N) * VIF
        
        Args:
            residuals: Array of residuals in the block (M, 3)
            raw_variance: Mean of raw covariance matrices in the block (3, 3)
            
        Returns:
            corrected_variance: Corrected block covariance (3, 3)
        """
        N = len(residuals)
        
        # Compute Lag-1 autocorrelation
        rho = self.compute_lag1_autocorrelation(residuals)
        
        # Compute VIF (Variance Inflation Factor)
        if rho >= 1.0:
            vif = 1e6  # Very large if rho approaches 1
        elif rho < 0:
            vif = 1.0  # No inflation if negative correlation
        else:
            vif = (1.0 + rho) / (1.0 - rho)
        
        # Compute effective sample size
        N_eff = N * (1.0 - rho) / (1.0 + rho) if rho < 1.0 else 1.0
        N_eff = max(1.0, N_eff)  # Ensure at least 1
        
        # Original naive formula: R_k = mean(Sigma) / N
        # Corrected formula: R_k = mean(Sigma) / N_eff = mean(Sigma) / N * VIF
        corrected_variance = raw_variance / N_eff
        
        return corrected_variance
    
    def compute_spectral_scaling_variance(self, raw_variance: np.ndarray, N: int) -> np.ndarray:
        """
        Compute corrected block variance using spectral scaling method.
        
        For colored noise (multipath), variance doesn't decay as 1/N but as 1/N^kappa
        where kappa is typically in [0.5, 0.8].
        
        Args:
            raw_variance: Mean of raw covariance matrices in the block (3, 3)
            N: Number of samples in the block
            
        Returns:
            corrected_variance: Corrected block covariance (3, 3)
        """
        # Original: R_k = mean(Sigma) / N
        # Corrected: R_k = mean(Sigma) / N^kappa
        corrected_variance = raw_variance / (N ** self.kappa)
        
        return corrected_variance
        
    def process(self, error_vecs: List[np.ndarray], cov_matrices: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute block means and corrected block covariances using effective sample size.
        
        Args:
            error_vecs: List of (3,) position error vectors
            cov_matrices: List of (3,3) covariance matrices
            
        Returns:
            block_means: (K, 3) array of averaged errors
            block_covs: (K, 3, 3) array of corrected averaged covariances
            block_indices: (K,) array of indices (center of block)
        """
        n_samples = len(error_vecs)
        n_blocks = n_samples // self.M
        
        if n_blocks == 0:
            return np.array([]), np.array([]), np.array([])
            
        block_means = []
        block_covs = []
        block_indices = []
        
        for k in range(n_blocks):
            start_idx = k * self.M
            end_idx = (k + 1) * self.M
            
            # Extract block
            batch_errors = np.array(error_vecs[start_idx:end_idx])  # (M, 3)
            batch_covs = np.array(cov_matrices[start_idx:end_idx])  # (M, 3, 3)
            
            # Compute Mean (unchanged)
            y_k = np.mean(batch_errors, axis=0)
            
            # Compute mean of raw covariance matrices
            raw_mean_cov = np.mean(batch_covs, axis=0)
            
            # Apply correction based on method
            if self.method == 'effective_size':
                # Method 1: Effective Sample Size correction
                R_k = self.compute_effective_variance(batch_errors, raw_mean_cov)
            elif self.method == 'spectral_scaling':
                # Method 2: Spectral Scaling correction
                R_k = self.compute_spectral_scaling_variance(raw_mean_cov, self.M)
            else:
                # Fallback: Original naive formula
                R_k = raw_mean_cov / self.M
            
            block_means.append(y_k)
            block_covs.append(R_k)
            block_indices.append(start_idx + self.M // 2)
            
        return np.array(block_means), np.array(block_covs), np.array(block_indices)


class DiagnosticSystem:
    """
    Implements the full diagnosis pipeline defined in research plan.md:
    1. Block Averaging (Time Domain Decoupling)
    2. TIM Manuscript Methodology (ELT + Directional Probing on Block Means)
    """
    def __init__(self, state_dim: int = 3):
        self.averager = BlockAveragerEffective()
        self.tim_algo = NCI_NLL_ES_Algorithm(state_dim=state_dim)
        
    def run(self, error_vecs: List[np.ndarray], cov_matrices: List[np.ndarray]):
        #------------------- 1. Block Averaging -------------------------------------#
        means, covs, indices = self.averager.process(error_vecs, cov_matrices)
        
        if len(means) == 0:
            return None
        
        #------------------- 2. TIM Framework Diagnosis -----------------------------#
        # Apply the full TIM methodology (ELT, NCI, NLL, ES, Probing) on the Block Means.
        # We treat block means as the "errors" (residuals) to be diagnosed.
        # true_states = 0 (since means are errors)
        # estimated_states = means
        # claimed_covariances = covs
        
        zeros = np.zeros_like(means)
        
        # Run the full algorithm
        result: AlgorithmResult = self.tim_algo.run_algorithm(zeros, means, covs)
        
        # Prepare bias estimate for visualization (if ELT detected bias)
        # Note: The algorithm detects SMM but doesn't explicitly return the mu_hat vector 
        # in the result struct, only the classification. 
        # We can re-estimate it here simply for the plot if needed.
        if "Bias" in result.classification:
            mu_hat = np.mean(means, axis=0)
        else:
            mu_hat = np.zeros(3)
            
        smm_estimate_all = np.tile(mu_hat, (len(means), 1))
        
        return {
            "block_indices": indices,
            "block_means": means,
            "smm_estimate": smm_estimate_all,
            "p_value": np.full(len(means), result.p_elt), # ELT p-value
            "postprocessed_states": means,
            "calibration_alpha": 1.0,
            "classification": result.classification,
            "algo_result": result
        }


class DiaNew:
    def __init__(self, state_dim: int = 3, n_samples: int = 1000, block_window_s: float = 900.0):
        """
        Initialize the algorithm.
        
        Args:
            state_dim: Dimension of the state vector
            n_samples: Number of Monte Carlo samples for ES calculation
        """
        self.state_dim = state_dim
        self.n_samples = n_samples
        self.block_window_s = block_window_s

        # Default thresholds (can be calibrated via bootstrap)
        self.tau_nci = 0.5  # dB - used for NCI classification
        self.tau_nll = 0.1  # Not used in SRD approach, kept for compatibility
        self.tau_es = 0.1   # Not used in SRD approach, kept for compatibility
        self.nees_chi2_alpha = 0.01 
        self.alpha_sig = 0.05
        self.prop_scale = 2
        
    def compute_nees_nci(self, true_states: np.ndarray, estimated_states: np.ndarray, 
                   claimed_covariances: np.ndarray) -> float:
        """
        Compute Normalized Covariance Index (NCI).
        
        NCI = 10 * (mean(log10(epsilon)) - mean(log10(epsilon_star)))
        where epsilon = e^T * Sigma_tilde^(-1) * e
        and epsilon_star = e^T * M^(-1) * e, with M = Sigma + mu*mu^T
        
        Args:
            true_states: True state vectors (N x d)
            estimated_states: Estimated state vectors (N x d) 
            claimed_covariances: Claimed covariance matrices (N x d x d)
            
        Returns:
            NCI value in dB
        """
        N = len(true_states)
        errors = true_states - estimated_states
        
        # Compute ness values
        nees_values = []
        for k in range(N):
            e_k = errors[k]
            Sigma_tilde_k = claimed_covariances[k]
            
            # Add regularization to avoid numerical issues
            Sigma_tilde_k_reg = Sigma_tilde_k + np.eye(self.state_dim) * 1e-8
            
            try:
                epsilon_k = e_k.T @ np.linalg.inv(Sigma_tilde_k_reg) @ e_k
                nees_values.append(epsilon_k)
            except np.linalg.LinAlgError:
                nees_values.append(1000.0)  # Large value for singular matrix
        

        # Use the Kolmogorov-Smirnov test for goodness-of-fit to chi2
        eps_arr = np.array(nees_values)
        ks_stat, nees_ks_pvalue = kstest(eps_arr, lambda x: chi2.cdf(x, df=self.state_dim))

        # Estimate M = Sigma + mu*mu^T from samples
        error_mean = np.mean(errors, axis=0)
        error_cov = np.cov(errors.T)
        M = error_cov + np.outer(error_mean, error_mean)
        
        # Add regularization
        M_reg = M + np.eye(self.state_dim) * 1e-8
        
        # Compute epsilon_star values
        epsilon_star_values = []
        for k in range(N):
            e_k = errors[k]
            try:
                epsilon_star_k = e_k.T @ np.linalg.inv(M_reg) @ e_k
                epsilon_star_values.append(epsilon_star_k)
            except np.linalg.LinAlgError:
                epsilon_star_values.append(1000.0)
        
        # Compute NCI
        log_epsilon_mean = np.mean(np.log10(nees_values))
        log_epsilon_star_mean = np.mean(np.log10(epsilon_star_values))
        
        nci = 10 * (log_epsilon_mean - log_epsilon_star_mean)
        return nees_ks_pvalue,nci
    
    def compute_nll(self, true_state: np.ndarray, estimated_state: np.ndarray, 
                   covariance: np.ndarray) -> float:
        """
        Compute Negative Log-Likelihood.
        
        NLL(F_k, x_k) = 0.5 * (x_k - x_hat_k)^T * Sigma^(-1) * (x_k - x_hat_k) 
                       + 0.5 * log|Sigma_k| + d/2 * log(2*pi)
        
        Args:
            true_state: True state vector
            estimated_state: Estimated state vector
            covariance: Covariance matrix
            
        Returns:
            NLL value
        """
        error = true_state - estimated_state
        
        # Add regularization
        cov_reg = covariance + np.eye(self.state_dim) * 1e-8
        
        try:
            # Quadratic term
            quad_term = 0.5 * error.T @ np.linalg.inv(cov_reg) @ error
            
            # Log determinant term
            log_det_term = 0.5 * np.log(np.linalg.det(cov_reg))
            
            # Constant term
            const_term = 0.5 * self.state_dim * np.log(2 * np.pi)
            
            nll = quad_term + log_det_term + const_term
            return float(nll)
            
        except np.linalg.LinAlgError:
            return 1000.0  # Large value for singular matrix
    
    def _matrix_inverse_sqrt(self, covariance: np.ndarray) -> np.ndarray:
        """Compute covariance^(-1/2) with regularization for stability."""
        cov_reg = covariance + np.eye(self.state_dim) * 1e-8
        try:
            # Use eigen-decomposition for symmetric PSD matrices
            eigenvalues, eigenvectors = np.linalg.eigh(cov_reg)
            inv_sqrt_eigenvalues = 1.0 / np.sqrt(np.clip(eigenvalues, 1e-12, None))
            inv_sqrt = (eigenvectors * inv_sqrt_eigenvalues) @ eigenvectors.T
            return inv_sqrt
        except np.linalg.LinAlgError:
            # Fallback to identity scaling
            return np.eye(self.state_dim)
    
    def compute_elt(self, true_states: np.ndarray, estimated_states: np.ndarray,
                    claimed_covariances: np.ndarray,
                    B: int = 2000, max_pairs: int = 50000) -> Tuple[int, float]:
        """
        Energy Location Test (ELT) via sign-flip randomization.
        Returns (ELT_indicator, p_value).
        """
        N = len(true_states)
        errors = true_states - estimated_states
        # Per-sample whitening: s_k = Sigma_tilde^{-1/2} e_k
        s_list = []
        for k in range(N):
            inv_sqrt = self._matrix_inverse_sqrt(claimed_covariances[k])
            s_list.append(inv_sqrt @ errors[k])
        s = np.vstack(s_list)
        
        # Helper to compute T-hat from a set of vectors (possibly sign-flipped)
        def compute_T(value_vectors: np.ndarray) -> float:
            # Subsample pairs for efficiency if needed
            if N < 2:
                return 0.0
            total_pairs = N * (N - 1) // 2
            if total_pairs <= max_pairs:
                # Compute all pairs
                # Sample all pairs indices
                idx_i, idx_j = np.triu_indices(N, k=1)
            else:
                # Randomly sample pairs
                rng_i = np.random.randint(0, N, size=max_pairs)
                rng_j = np.random.randint(0, N, size=max_pairs)
                mask = rng_i != rng_j
                idx_i = rng_i[mask]
                idx_j = rng_j[mask]
            v_i = value_vectors[idx_i]
            v_j = value_vectors[idx_j]
            plus = np.linalg.norm(v_i + v_j, axis=1)
            minus = np.linalg.norm(v_i - v_j, axis=1)
            T_hat = (2.0 / (N * (N - 1))) * np.sum(plus - minus)
            return float(T_hat)
        
        T_obs = compute_T(s)
        # Randomization with Rademacher signs
        count_ge = 0
        for _ in range(B):
            signs = np.random.choice([-1.0, 1.0], size=N)
            s_flip = s * signs[:, None]
            T_b = compute_T(s_flip)
            if T_b >= T_obs:
                count_ge += 1
        p_val = (1 + count_ge) / (B + 1)
        ELT = 1 if p_val < self.alpha_sig else 0
        return ELT, float(p_val)
    
    def compute_energy_score(self, true_state: np.ndarray, estimated_state: np.ndarray,
                           covariance: np.ndarray) -> float:
        """
        Compute Energy Score.
        
        ES(F_k, x_k) = E[||Y - x_k||_2] - 0.5 * E[||Y - Y'||_2]
        where Y, Y' ~ F_k = N(x_hat_k, Sigma_k)
        
        Args:
            true_state: True state vector
            estimated_state: Estimated state vector
            covariance: Covariance matrix
            
        Returns:
            Energy Score value
        """
        # Add regularization
        cov_reg = covariance + np.eye(self.state_dim) * 1e-8
        
        try:
            # Generate samples from estimated distribution
            samples = np.random.multivariate_normal(estimated_state, cov_reg, self.n_samples)
            
            # First term: E[||Y - x_k||_2]
            distances_to_true = np.linalg.norm(samples - true_state, axis=1)
            term1 = np.mean(distances_to_true)
            
            # Second term: 0.5 * E[||Y - Y'||_2]
            n_pairs = min(200, self.n_samples // 2)
            idx1 = np.random.choice(self.n_samples, n_pairs, replace=False)
            idx2 = np.random.choice(self.n_samples, n_pairs, replace=False)
            
            pairwise_distances = np.linalg.norm(samples[idx1] - samples[idx2], axis=1)
            term2 = 0.5 * np.mean(pairwise_distances)
            
            return term1 - term2
            
        except np.linalg.LinAlgError:
            return 1000.0  # Large value for singular matrix
    
    def compute_nll_es_for_k(self, true_states: np.ndarray, estimated_states: np.ndarray,
                                 claimed_covariances: np.ndarray, k: float) -> Tuple[float, float]:
        """
        Compute mean NLL and ES for a given scale factor k.

        Args:
            true_states: True state vectors (N x d)
            estimated_states: Estimated state vectors (N x d)
            claimed_covariances: Claimed covariance matrices (N x d x d)
            k: scale factor

        Returns:
            Tuple of (mean_nll, mean_es)
        """
        N = len(true_states)
        nll_list = []
        es_list = []
        for i in range(N):
            scaled_cov = k * claimed_covariances[i]
            nll_k = self.compute_nll(true_states[i], estimated_states[i], scaled_cov)
            es_k = self.compute_energy_score(true_states[i], estimated_states[i], scaled_cov)
            nll_list.append(nll_k)
            es_list.append(es_k)
        mean_nll = np.mean(nll_list)
        mean_es = np.mean(es_list)
        return mean_nll, mean_es
        
    def compute_directional_probes(self, true_states: np.ndarray, estimated_states: np.ndarray,
                                 claimed_covariances: np.ndarray) -> Tuple[Dict, Dict, Dict]:
        """
        Compute directional probes using NLL and ES at different scales.
        
        Args:
            true_states: True state vectors (N x d)
            estimated_states: Estimated state vectors (N x d)
            claimed_covariances: Claimed covariance matrices (N x d x d)
            
        Returns:
            Tuple of (nll_values, es_values, deltas)
        """
        N = len(true_states)

        # Compute directional differences
        # These are used to compute Slope Relative Differences (SRD) in classification:
        # SRD_{NLL} = abs((2×|Δ⁻_NLL| - |Δ⁺_NLL|) / |Δ⁺_NLL|)
        # SRD_{ES} = abs((2×|Δ⁻_ES| - |Δ⁺_ES|) / |Δ⁺_ES|)
        # For calibrated: Δ⁻_NLL > 0, Δ⁺_NLL > 0, Δ⁻_ES > 0, Δ⁺_ES > 0
        nll_mean, es_mean = self.compute_nll_es_for_k(true_states, estimated_states, claimed_covariances, 1)
        nll_mean_k, es_mean_k = self.compute_nll_es_for_k(true_states, estimated_states, claimed_covariances, self.prop_scale)
        nll_mean_rk, es_mean_rk = self.compute_nll_es_for_k(true_states, estimated_states, claimed_covariances, 1/self.prop_scale)
        deltas = {
            'delta_nll_minus': nll_mean_rk - nll_mean,
            'delta_nll_plus': nll_mean_k - nll_mean,
            'delta_es_minus': es_mean_rk - es_mean,
            'delta_es_plus': es_mean_k - es_mean
        }

        return nll_mean, es_mean, deltas

    def calculate_probe(self, nees_ks_pvalue: float, nci: float, deltas: Dict) -> Tuple[str, bool]:
        delta_nll_minus = deltas['delta_nll_minus']
        delta_nll_plus = deltas['delta_nll_plus']
        delta_es_minus = deltas['delta_es_minus']
        delta_es_plus = deltas['delta_es_plus']
        
        # Compute Slope Relative Differences (SRD)
        srd_nll = float('inf')
        srd_es = float('inf')
        
        if abs(delta_nll_plus) > 1e-10:
            srd_nll = abs((self.prop_scale * abs(delta_nll_minus) - abs(delta_nll_plus)) / abs(delta_nll_plus))
        
        if abs(delta_es_plus) > 1e-10:
            srd_es = abs((self.prop_scale * abs(delta_es_minus) - abs(delta_es_plus)) / abs(delta_es_plus))
        
        return srd_nll, srd_es

    def run_algorithm(self, true_states: np.ndarray, estimated_states: np.ndarray,
                     claimed_covariances: np.ndarray) -> AlgorithmResult:
        """
        Run the complete NCI, NLL, and ES algorithm.
        
        Args:
            true_states: True state vectors (N x d)
            estimated_states: Estimated state vectors (N x d)
            claimed_covariances: Claimed covariance matrices (N x d x d)
            
        Returns:
            AlgorithmResult object containing all results
        """
        

        # Compute ELT based on raw series
        elt, p_elt = self.compute_elt(true_states, estimated_states, claimed_covariances)
        # if elt !=0:
        #     return 'Bias', True

        #--------------raw series results--------------#
        # compute NCI 
        nees_ks_pvalue, nci = self.compute_nees_nci(true_states, estimated_states, claimed_covariances)
        
        # Compute directional probes
        nll_values, es_values, deltas = self.compute_directional_probes(
            true_states, estimated_states, claimed_covariances
        )
        srd_nll, srd_es = self.calculate_probe( nees_ks_pvalue,nci, deltas)

        # #--------------block averaged results (useless)--------------#

        #  # compute block means
        # estimated_states_means, claimed_covariances_means, _ = BlockAveragerEffective(T_batch=self.block_window_s, f_hz=1.0).process(true_states-estimated_states, claimed_covariances)
        # true_states_means = np.zeros_like(estimated_states_means)

        # # compute NCI 
        # nees_ks_pvalue_block, nci_block = self.compute_nees_nci(true_states_means, estimated_states_means, claimed_covariances_means)
        
        # # Compute directional probes
        # nll_values_block, es_values_block, deltas_block = self.compute_directional_probes(
        #     true_states_means, estimated_states_means, claimed_covariances_means
        # )
        # srd_nll_block, srd_es_block = self.calculate_probe(nees_ks_pvalue_block,nci_block, deltas_block)
        
        # Organize results into DataFrame
        results_dict = {
            # ELT results
            'elt': elt,
            'p_elt': p_elt,
            
            # Raw series results
            'nees_ks_pvalue': nees_ks_pvalue,
            'nci': nci,
            'srd_nll': srd_nll,
            'srd_es': srd_es,
            
            # # Block averaged results
            # 'nees_ks_pvalue_block': nees_ks_pvalue_block,
            # 'nci_block': nci_block,
            # 'srd_nll_block': srd_nll_block,
            # 'srd_es_block': srd_es_block,
        }
        
        # Create DataFrame
        results_df = pd.DataFrame([results_dict])
        
        return results_df

