"""
Test Script for DiaNew Algorithm Performance Under Thermal Noise Only (No Multipath)

This script tests the DiaNew diagnostic algorithm in 6 scenarios with only thermal noise
(no multipath), to validate the algorithm's ability to distinguish:
1. Credible (calibrated)
2. Optimism (underestimated covariance)
3. Pessimism (overestimated covariance)
4. SMM (systematic bias)
5. Optimism + SMM
6. Pessimism + SMM

Author: Auto-generated
Date: 2026-01-17
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from skyfield.api import utc

from geometry_engine import generate_obs
from error_injection import inject_errors
from dgnss_solver import solve_dgnss
from diagnostic_system import DiaNew
from lib.algorithm_comparison import PureNCIClassifier, NEESChiSquaredClassifier,NCI_NLL_ES_Algorithm


def run_diagnosis_thermal_only(name, mode, inject_nmm, block_window_s, 
                                obs_base, obs_rover, sat_info,
                                base_loc, rover_loc):
    """
    Run diagnosis with thermal noise only (no multipath).
    
    Parameters:
    -----------
    name : str
        Scenario name
    mode : str
        Error injection mode ('healthy_no_multipath' or 'system_fault_smm')
    inject_nmm : float
        Covariance scaling factor
    block_window_s : float
        Block averaging window in seconds
    obs_base, obs_rover, sat_info : DataFrames
        Observation data
    base_loc, rover_loc : tuples
        Station locations
        
    Returns:
    --------
    results_df : pandas.DataFrame
        Diagnostic results
    """
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"  Mode: {mode}")
    print(f"  NMM Scale Factor: {inject_nmm:.2f}")
    print(f"{'='*80}")
    
    print("Injecting errors (thermal noise only)...")
    obs_base_c, obs_rover_c, sigma_dd = inject_errors(
        obs_base, obs_rover, 1.0,  # baseline_km
        base_loc=base_loc, rover_loc=rover_loc,
        sample_interval=1.0,
        use_real_models=False,
        scenario_mode=mode,
        multipath_config=None  # No multipath configuration
    )
    
    print("Solving DGNSS...")
    _, _, _, _, _, _, Q_xyz, error_vecs = solve_dgnss(
        obs_base_c, obs_rover_c, sat_info, base_loc, rover_loc,
        stochastic_model='equal_weight', sigma_dd=sigma_dd, decimation_factor=1
    )
    
    if inject_nmm != 1.0:
        print(f"Scaling covariance by {inject_nmm}...")
        Q_xyz = [cov * inject_nmm for cov in Q_xyz]
    
    print("Running diagnosis...")
    
    # Pure NEES and Pure NCI classifiers
    all_zero_states = np.zeros_like(error_vecs)
    classifier_nees = NEESChiSquaredClassifier(state_dim=3)
    classification_nees = classifier_nees.classify(all_zero_states, error_vecs, Q_xyz)
    classifier_nci = PureNCIClassifier(state_dim=3)
    classification_nci = classifier_nci.classify(all_zero_states, error_vecs, Q_xyz)
    
    # NCI_NLL_ES algorithm
    classifier = NCI_NLL_ES_Algorithm(state_dim=3, n_samples=1000)
    res_new = classifier.run_algorithm(all_zero_states, error_vecs, Q_xyz)
    
    # Convert AlgorithmResult to DataFrame
    df_new = pd.DataFrame([{
        'elt': res_new.elt,
        'p_elt': res_new.p_elt,
        'nci': res_new.nci,
        'delta_nll_minus': res_new.delta_nll_minus,
        'delta_nll_plus': res_new.delta_nll_plus,
        'delta_es_minus': res_new.delta_es_minus,
        'delta_es_plus': res_new.delta_es_plus,
        'classification': res_new.classification,
        'pure_nees_classification': classification_nees,
        'pure_nci_classification': classification_nci,
        'inject_nmm': inject_nmm,
        'error_mode': mode
    }])
    
    # Compute SRD metrics for consistency with other tests
    prop_scale = 2.0  # Default proportional scaling factor
    if abs(res_new.delta_nll_plus) > 1e-10:
        srd_nll = abs((prop_scale * abs(res_new.delta_nll_minus) - abs(res_new.delta_nll_plus)) / abs(res_new.delta_nll_plus))
    else:
        srd_nll = float('inf')
    
    if abs(res_new.delta_es_plus) > 1e-10:
        srd_es = abs((prop_scale * abs(res_new.delta_es_minus) - abs(res_new.delta_es_plus)) / abs(res_new.delta_es_plus))
    else:
        srd_es = float('inf')
    
    df_new['srd_nll'] = srd_nll
    df_new['srd_es'] = srd_es
    
    print(f"  Pure NEES: {classification_nees}")
    print(f"  Pure NCI: {classification_nci}")
    print(f"  NCI_NLL_ES ELT: {res_new.elt}, p_elt: {res_new.p_elt:.4f}")
    print(f"  NCI_NLL_ES NCI: {res_new.nci:.4f} dB")
    print(f"  NCI_NLL_ES Classification: {res_new.classification}")
    
    return df_new


def test_thermal_only_scenarios(
    baseline_km=1.0,
    duration_hours=4.0,
    block_window_s=900.0,
    nmm_scale=2.0,
    output_dir='test_results'
):
    """
    Main test function for thermal-only (no multipath) scenarios.
    
    Tests 6 scenarios:
    1. Credible (calibrated)
    2. Optimism (underestimated covariance)
    3. Pessimism (overestimated covariance)
    4. SMM (systematic bias)
    5. Optimism + SMM
    6. Pessimism + SMM
    
    Parameters:
    -----------
    baseline_km : float
        Baseline length in km (default: 1.0)
    duration_hours : float
        Simulation duration in hours (default: 4.0)
    block_window_s : float
        Block averaging window in seconds (default: 900.0)
    nmm_scale : float
        Scaling factor for NMM scenarios (default: 2.0)
        - Optimism uses 1/nmm_scale
        - Pessimism uses nmm_scale
    output_dir : str
        Output directory for results (default: 'test_results')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("THERMAL NOISE ONLY TEST (No Multipath)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Baseline: {baseline_km} km")
    print(f"  Duration: {duration_hours} hours")
    print(f"  Block Window: {block_window_s} s ({block_window_s/60:.1f} minutes)")
    print(f"  NMM Scale: {nmm_scale} (Optimism: {1/nmm_scale:.2f}, Pessimism: {nmm_scale:.2f})")
    print("="*80)
    
    # Common configuration
    sample_interval = 1.0  # 1 Hz
    base_loc = (42.3601, -71.0589, 0.0)  # Boston area
    mask_angle_deg = 5.0
    stochastic_model = 'equal_weight'
    
    # Calculate rover location
    rover_lat = base_loc[0] + (baseline_km / 111.0)
    rover_loc = (rover_lat, base_loc[1], base_loc[2])
    
    # Start time
    start_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=utc)
    
    # Generate geometry once (shared across all tests)
    print("\nGenerating geometry...")
    obs_base, obs_rover, _, sat_info, _ = generate_obs(
        start_time, duration_hours, sample_interval, base_loc, baseline_km,
        mask_angle_deg=mask_angle_deg
    )
    print(f"Generated {len(obs_base)} epochs of observations")
    
    # Define scenarios
    # Note: scenario_mode options from error_injection.py:
    #   - 'healthy_no_multipath': Only thermal noise
    #   - 'system_fault_smm': Constant bias (SMM)
    scenarios = [
        {
            'name': 'credible',
            'mode': 'healthy_no_multipath',
            'inject_nmm': 1.0,
            'description': 'Calibrated (thermal only, correct covariance)'
        },
        {
            'name': 'optimism',
            'mode': 'healthy_no_multipath',
            'inject_nmm': 1.0 / nmm_scale,
            'description': 'Optimism (thermal only, underestimated covariance)'
        },
        {
            'name': 'pessimism',
            'mode': 'healthy_no_multipath',
            'inject_nmm': nmm_scale,
            'description': 'Pessimism (thermal only, overestimated covariance)'
        },
        {
            'name': 'smm',
            'mode': 'system_fault_smm',
            'inject_nmm': 1.0,
            'description': 'SMM (systematic bias, correct covariance)'
        },
        {
            'name': 'smm_optimism',
            'mode': 'system_fault_smm',
            'inject_nmm': 1.0 / nmm_scale,
            'description': 'SMM + Optimism (bias + underestimated covariance)'
        },
        {
            'name': 'smm_pessimism',
            'mode': 'system_fault_smm',
            'inject_nmm': nmm_scale,
            'description': 'SMM + Pessimism (bias + overestimated covariance)'
        },
    ]
    
    # Collect all results
    all_results = []
    
    # Test each scenario
    for scenario in scenarios:
        print(f"\n\n{'#'*80}")
        print(f"# SCENARIO: {scenario['name'].upper()}")
        print(f"# {scenario['description']}")
        print(f"{'#'*80}")
        
        try:
            df_result = run_diagnosis_thermal_only(
                name=scenario['name'],
                mode=scenario['mode'],
                inject_nmm=scenario['inject_nmm'],
                block_window_s=block_window_s,
                obs_base=obs_base,
                obs_rover=obs_rover,
                sat_info=sat_info,
                base_loc=base_loc,
                rover_loc=rover_loc
            )
            
            # Add scenario name and description
            df_result.insert(0, 'scenario', scenario['name'])
            df_result.insert(1, 'description', scenario['description'])
            
            all_results.append(df_result)
            
        except Exception as e:
            print(f"ERROR in {scenario['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f'thermal_only_test_{timestamp}_d{duration_hours}h_scale{nmm_scale}.csv')
        
        # Export to CSV
        combined_df.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print(f"All results exported to: {output_file}")
        print(f"Total test cases: {len(combined_df)}")
        print(f"Columns: {list(combined_df.columns)}")
        print(f"{'='*80}")
        
        return combined_df
    else:
        print("\nERROR: No results collected!")
        return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DiaNew algorithm with thermal noise only (no multipath)')
    parser.add_argument('--duration', type=float, default=0.25,
                        help='Simulation duration in hours (default: 4.0)')
    parser.add_argument('--nmm-scale', type=float, default=5.0,
                        help='NMM scaling factor (default: 2.0)')
    parser.add_argument('--output-dir', type=str, default='test_results',
                        help='Output directory for results (default: test_results)')
    args = parser.parse_args()
    
    test_thermal_only_scenarios(
        baseline_km=1.0,
        duration_hours=args.duration,
        block_window_s=900.0,
        nmm_scale=args.nmm_scale,
        output_dir=args.output_dir
    )
