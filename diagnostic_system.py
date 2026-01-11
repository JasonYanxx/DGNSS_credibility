"""
Diagnostic System for DGNSS Monitoring Network (Commissioning Protocol)
Strictly implements the 'Block Averaging -> TIM Framework' pipeline.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from lib.algorithm_comparison import NCI_NLL_ES_Algorithm, AlgorithmResult

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


class DiagnosticSystem:
    """
    Implements the full diagnosis pipeline defined in research plan.md:
    1. Block Averaging (Time Domain Decoupling)
    2. TIM Manuscript Methodology (ELT + Directional Probing on Block Means)
    """
    def __init__(self, state_dim: int = 3):
        self.averager = BlockAverager()
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
