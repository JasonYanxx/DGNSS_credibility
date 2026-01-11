import numpy as np
from typing import Dict, Tuple
from scipy import stats
from scipy.stats import chi2, multivariate_normal
from dataclasses import dataclass


@dataclass
class AlgorithmResult:
    """Container for algorithm results"""
    nci: float
    nll_values: Dict[float, float]  # k -> NLL(k)
    es_values: Dict[float, float]   # k -> ES(k)
    delta_nll_minus: float
    delta_nll_plus: float
    delta_es_minus: float
    delta_es_plus: float
    elt: int
    p_elt: float
    classification: str

class NCI_NLL_ES_Algorithm:
    """
    Implementation of the NCI, NLL, and ES algorithm for uncertainty quantification evaluation.
    
    This algorithm combines:
    1. NCI (Normalized Covariance Index) for scale signal
    2. NLL (Negative Log-Likelihood) for optimism-sided sensitivity  
    3. ES (Energy Score) for pessimism-sided sensitivity
    
    Based on the document: "Algorithm Development and Evaluation Using NCI, NLL, and ES"
    """
    
    def __init__(self, state_dim: int = 2, n_samples: int = 1000):
        """
        Initialize the algorithm.
        
        Args:
            state_dim: Dimension of the state vector
            n_samples: Number of Monte Carlo samples for ES calculation
        """
        self.state_dim = state_dim
        self.n_samples = n_samples
        
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
        ks_stat, nees_ks_pvalue = stats.kstest(eps_arr, lambda x: chi2.cdf(x, df=self.state_dim))

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

    
    def classify_uncertainty(self, elt: int, nees_ks_pvalue: float, nci: float, deltas: Dict) -> Tuple[str, bool]:
        """
        Classify uncertainty using ELT, NCI, and SRD per the document.
        
        Args:
            nci: NCI value
            deltas: Dictionary of directional differences
            
        Returns:
            Tuple of (classification, end_of_judgement)
        """
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
        
        # Branch on ELT (bias test)
        if elt == 0:
            # No bias
            if nees_ks_pvalue > self.nees_chi2_alpha or abs(nci) <=self.tau_nci:
                return "Calibrated", True
            else:
                if nci < -self.tau_nci:
                    return "Pessimistic", True
                elif nci > self.tau_nci:
                    return "Optimistic", True
                else:
                    # Should not happen if abs(nci) > tau
                    return "Calibrated", True
        else:
            return "Further Justification", False
            
        return "Unknown", True


                
        # # Branch on ELT (bias test)
        # if elt == 0:
        #     # No bias
        #     if nees_ks_pvalue > self.nees_chi2_alpha:
        #         if (delta_nll_minus > 0 and delta_nll_plus > 0 and 
        #             delta_es_minus > 0 and delta_es_plus > 0):
        #             return "Calibrated", True
        #         else:
        #             return "Bias", True
        #     else:
        #         if nci < -self.tau_nci:
        #             return "Pessimistic", True
        #         elif nci > self.tau_nci:
        #             return "Optimistic", True
        #         else:
        #             return "Calibrated", True
        # else:
        #     return "Further Justification", False
    
    def classify_uncertainty_remove_bias(self, nci: float, deltas: Dict) -> Tuple[str, bool]:
        """
        Classify uncertainty using ELT, NCI, and SRD per the document.
        
        Args:
            nci: NCI value
            deltas: Dictionary of directional differences
            
        Returns:
            Tuple of (classification, end_of_judgement)
        """
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
        

        # bias
        if nci < -self.tau_nci:
            return "Pessimistic + Bias", True
        elif nci > self.tau_nci:
            # Maybe also optimistic
            if (delta_nll_minus > 0 and delta_nll_plus > 0 and 
                delta_es_minus > 0 and delta_es_plus > 0):
                return "Bias", True
            elif srd_nll > srd_es:
                return "Optimistic + Bias", True
            else:
                return "Bias", True
        else:
            return "Bias", True 


        # # bias
        # if nci < -self.tau_nci:
        #     return "Pessimistic + Bias", True
        # elif nci > self.tau_nci:
        #     # Maybe also optimistic or small pessimistic
        #     if (delta_nll_minus > 0 and delta_nll_plus > 0 and 
        #         delta_es_minus > 0 and delta_es_plus > 0):
        #         return "Bias", True
        #     elif srd_nll > srd_es:
        #         return "Optimistic + Bias", True
        #     elif srd_es > srd_nll:
        #         return "Pessimistic + Bias", True
        #     else:
        #         return "Bias", True
        # else:
        #     return "Bias", True 

    

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
        # Step 1: Compute ELT and NCI
        elt, p_elt = self.compute_elt(true_states, estimated_states, claimed_covariances)
        nees_ks_pvalue, nci = self.compute_nees_nci(true_states, estimated_states, claimed_covariances)
        
        # Step 2: Compute directional probes
        nll_values, es_values, deltas = self.compute_directional_probes(
            true_states, estimated_states, claimed_covariances
        )
        
        # Step 3: Classify uncertainty
        classification, is_end = self.classify_uncertainty(elt, nees_ks_pvalue,nci, deltas)
        if is_end == False:
            # remove bias effect
            bias_estiamte = np.mean(estimated_states-true_states,axis=0)
            # Compute NCI
            nees_ks_pvalue, nci = self.compute_nees_nci(true_states, estimated_states-bias_estiamte, claimed_covariances)
            # Compute directional probes
            nll_values, es_values, deltas = self.compute_directional_probes(
                true_states, estimated_states-bias_estiamte, claimed_covariances
            )
            classification, is_end = self.classify_uncertainty_remove_bias(nci, deltas)
            
        
        return AlgorithmResult(
            nci=nci,
            nll_values=nll_values,
            es_values=es_values,
            delta_nll_minus=deltas['delta_nll_minus'],
            delta_nll_plus=deltas['delta_nll_plus'],
            delta_es_minus=deltas['delta_es_minus'],
            delta_es_plus=deltas['delta_es_plus'],
            elt=elt,
            p_elt=p_elt,
            classification=classification,
        )

class PureNLLClassifier:
    """Pure Negative Log-Likelihood based classifier"""
    
    def __init__(self, state_dim: int = 2):
        self.state_dim = state_dim
        
    def classify(self, true_states: np.ndarray, estimated_states: np.ndarray, 
                claimed_covariances: np.ndarray) -> str:
        """Classify based on NLL values"""
        N = len(true_states)
        nll_values = []
        
        for k in range(N):
            e_k = true_states[k] - estimated_states[k]
            Sigma_k = claimed_covariances[k]
            
            # Add regularization
            Sigma_k_reg = Sigma_k + np.eye(self.state_dim) * 1e-8
            
            try:
                nll = -multivariate_normal.logpdf(e_k, mean=np.zeros(self.state_dim), cov=Sigma_k_reg)
                nll_values.append(nll)
            except:
                nll_values.append(1000.0)
        
        # Use mean NLL for classification
        mean_nll = np.mean(nll_values)
        
        # Thresholds based on chi2 distribution
        expected_nll = self.state_dim / 2  # Expected NLL for chi2 with df=state_dim
        
        if mean_nll < expected_nll * 0.8:
            return 'Optimistic'
        elif mean_nll > expected_nll * 1.2:
            return 'Pessimistic'
        else:
            return 'Calibrated'


class PureESClassifier:
    """Pure Energy Score based classifier"""
    
    def __init__(self, state_dim: int = 2, n_samples: int = 1000):
        self.state_dim = state_dim
        self.n_samples = n_samples
        
    def calculate_energy_score(self, true_state: np.ndarray, estimated_state: np.ndarray,
                              covariance: np.ndarray) -> float:
        """Calculate Energy Score"""
        try:
            samples = np.random.multivariate_normal(estimated_state, covariance, self.n_samples)
        except:
            samples = np.random.multivariate_normal(estimated_state, np.eye(len(estimated_state)), self.n_samples)
        
        # First term: E[||X - x_true||]
        distances_to_true = np.linalg.norm(samples - true_state, axis=1)
        term1 = np.mean(distances_to_true)
        
        # Second term: 0.5 * E[||X - X'||]
        n_pairs = min(200, self.n_samples // 2)
        idx1 = np.random.choice(self.n_samples, n_pairs, replace=False)
        idx2 = np.random.choice(self.n_samples, n_pairs, replace=False)
        pairwise_distances = np.linalg.norm(samples[idx1] - samples[idx2], axis=1)
        term2 = 0.5 * np.mean(pairwise_distances)
        
        return term1 - term2
    
    def classify(self, true_states: np.ndarray, estimated_states: np.ndarray, 
                claimed_covariances: np.ndarray) -> str:
        """Classify based on Energy Score values"""
        N = len(true_states)
        es_values = []
        
        for k in range(N):
            es = self.calculate_energy_score(true_states[k], estimated_states[k], claimed_covariances[k])
            es_values.append(es)
        
        # Use mean ES for classification
        mean_es = np.mean(es_values)
        
        # Thresholds (empirical)
        if mean_es < 0.5:
            return 'Optimistic'
        elif mean_es > 2.0:
            return 'Pessimistic'
        else:
            return 'Calibrated'


class NEESChiSquaredClassifier:
    """NEES Chi-squared test based classifier"""
    
    def __init__(self, state_dim: int = 2, alpha: float = 0.05):
        self.state_dim = state_dim
        self.alpha = alpha
        
    def classify(self, true_states: np.ndarray, estimated_states: np.ndarray, 
                claimed_covariances: np.ndarray) -> str:
        """Classify based on NEES chi-squared test"""
        N = len(true_states)
        nees_values = []
        
        for k in range(N):
            e_k = true_states[k] - estimated_states[k]
            Sigma_k = claimed_covariances[k]
            
            # Add regularization
            Sigma_k_reg = Sigma_k + np.eye(self.state_dim) * 1e-8
            
            try:
                nees = e_k.T @ np.linalg.inv(Sigma_k_reg) @ e_k
                nees_values.append(nees)
            except:
                nees_values.append(1000.0)
        
        # Chi-squared goodness-of-fit test
        nees_array = np.array(nees_values)
        expected_nees = self.state_dim
        
        # Use chi-squared test statistic
        chi2_stat = np.sum((nees_array - expected_nees)**2 / expected_nees)
        p_value = 1 - chi2.cdf(chi2_stat, df=N-1)
        
        if p_value < self.alpha:
            if np.mean(nees_array) < expected_nees:
                return 'Optimistic'
            else:
                return 'Pessimistic'
        else:
            return 'Calibrated'


class PureNCIClassifier:
    """Pure NCI based classifier"""
    
    def __init__(self, state_dim: int = 2):
        self.state_dim = state_dim
        
    def classify(self, true_states: np.ndarray, estimated_states: np.ndarray, 
                claimed_covariances: np.ndarray) -> str:
        """Classify based on NCI values"""
        N = len(true_states)
        nees_values = []
        
        for k in range(N):
            e_k = true_states[k] - estimated_states[k]
            Sigma_k = claimed_covariances[k]
            
            # Add regularization
            Sigma_k_reg = Sigma_k + np.eye(self.state_dim) * 1e-8
            
            try:
                nees = e_k.T @ np.linalg.inv(Sigma_k_reg) @ e_k
                nees_values.append(nees)
            except:
                nees_values.append(1000.0)
        
        # Calculate NCI
        log_nees = np.log10(nees_values)
        mean_log_nees = np.mean(log_nees)
        
        # Expected value for chi2 with df=state_dim
        expected_log_nees = np.log10(self.state_dim)
        nci = 10 * (mean_log_nees - expected_log_nees)
        
        # Classification based on NCI thresholds
        if nci < -0.5:  # dB
            return 'Optimistic'
        elif nci > 0.5:  # dB
            return 'Pessimistic'
        else:
            return 'Calibrated'
