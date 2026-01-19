"""
Test Script for DiaNew Algorithm Performance Under Different Multipath Configurations

This script systematically tests the DiaNew diagnostic algorithm across 6 scenarios
with varying multipath parameters to study how multipath characteristics affect
diagnostic performance.

Author: Auto-generated
Date: 2024
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
from lib.algorithm_comparison import PureNCIClassifier, NEESChiSquaredClassifier


def run_diagnosis_with_multipath_config(name, mode, inject_nmm, block_window_s, 
                                       obs_base, obs_rover, sat_info,
                                       multipath_config, base_loc, rover_loc):
    """
    Run diagnosis with specific multipath configuration.
    
    Parameters:
    -----------
    name : str
        Scenario name
    mode : str
        Error injection mode
    inject_nmm : float
        Covariance scaling factor
    block_window_s : float
        Block averaging window in seconds
    obs_base, obs_rover, sat_info : DataFrames
        Observation data
    multipath_config : dict
        Multipath parameters: {
            'specular_amp_base': float,
            'specular_amp_rover': float,
            'specular_freq': float,  # Hz (period = 1/freq in seconds)
            'phi_diffuse': float,  # AR(1) correlation (0.8-0.995)
            'sigma_diffuse_base': float,
            'sigma_diffuse_rover': float
        }
    base_loc, rover_loc : tuples
        Station locations
        
    Returns:
    --------
    results_df : pandas.DataFrame
        Diagnostic results
    """
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    spec_freq = multipath_config.get('specular_freq', 1.0/200.0)
    spec_period = 1.0 / spec_freq
    phi = multipath_config.get('phi_diffuse', 0.99)
    # Correlation time: tau = -dt / ln(phi) ≈ dt / (1-phi) for phi close to 1
    # For dt=1s: tau ≈ 1/(1-phi) seconds
    tau_approx = 1.0 / (1.0 - phi) if phi < 1.0 else float('inf')
    print(f"Multipath Config: spec_amp_base={multipath_config['specular_amp_base']:.3f}, "
          f"spec_amp_rover={multipath_config['specular_amp_rover']:.3f}, "
          f"spec_freq={spec_freq:.6f} Hz (period={spec_period:.1f}s), "
          f"phi={phi:.3f} (tau≈{tau_approx:.1f}s), "
          f"sigma_diff_base={multipath_config['sigma_diffuse_base']:.3f}, "
          f"sigma_diff_rover={multipath_config['sigma_diffuse_rover']:.3f}")
    print(f"{'='*80}")
    
    print("Injecting errors with multipath configuration...")
    obs_base_c, obs_rover_c, sigma_dd = inject_errors(
        obs_base, obs_rover, 1.0,  # baseline_km
        base_loc=base_loc, rover_loc=rover_loc,
        sample_interval=1.0,
        use_real_models=False,
        scenario_mode=mode,
        multipath_config=multipath_config  # Pass multipath config
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
    
    # DiaNew algorithm
    diag_new = DiaNew(state_dim=3, n_samples=1000, block_window_s=block_window_s)
    df_new = diag_new.run_algorithm(all_zero_states, error_vecs, Q_xyz)
    
    # Add pure NEES and pure NCI results
    df_new['pure_nees_classification'] = classification_nees
    df_new['pure_nci_classification'] = classification_nci
    
    # Add multipath configuration parameters to DataFrame
    df_new['multipath_spec_amp_base'] = multipath_config['specular_amp_base']
    df_new['multipath_spec_amp_rover'] = multipath_config['specular_amp_rover']
    df_new['multipath_spec_freq'] = multipath_config.get('specular_freq', 1.0/200.0)
    df_new['multipath_spec_period'] = 1.0 / df_new['multipath_spec_freq']  # Period in seconds
    df_new['multipath_phi_diffuse'] = multipath_config['phi_diffuse']
    df_new['multipath_sigma_diffuse_base'] = multipath_config['sigma_diffuse_base']
    df_new['multipath_sigma_diffuse_rover'] = multipath_config['sigma_diffuse_rover']
    
    print(f"  Pure NEES: {classification_nees}")
    print(f"  Pure NCI: {classification_nci}")
    print(f"  DiaNew ELT: {df_new['elt'].iloc[0]}, p_elt: {df_new['p_elt'].iloc[0]:.4f}")
    print(f"  DiaNew NCI: {df_new['nci'].iloc[0]:.4f}")
    
    return df_new


def test_dianew_multipath_sensitivity(
    baseline_km=1.0,
    duration_hours=4.0,
    block_window_s=900.0,
    nmm_scale=2.0,
    output_dir='test_results'
):
    """
    Main test function for DiaNew multipath sensitivity analysis.
    
    Tests 6 scenarios with multiple multipath configurations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("DIANEW MULTIPATH SENSITIVITY TEST")
    print("="*80)
    print(f"Configuration:")
    print(f"  Baseline: {baseline_km} km")
    print(f"  Duration: {duration_hours} hours")
    print(f"  Block Window: {block_window_s} s ({block_window_s/60:.1f} minutes)")
    print(f"  NMM Scale: {nmm_scale}")
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
    
    # Define multipath configurations to test
    # Strategy: Use default parameters as baseline, vary one parameter at a time
    # Default parameters (from error_injection.py):
    #   specular_amp_base = 0.2 m, specular_amp_rover = 0.2 m
    #   specular_freq = 1.0/200.0 Hz (200s period)
    #   phi_diffuse = 0.99
    #   sigma_diffuse_base = 0.05 m, sigma_diffuse_rover = 0.1 m
    
    # Baseline configuration (all defaults)
    baseline_config = {
        'specular_amp_base': 0.2,
        'specular_amp_rover': 0.2,
        'specular_freq': 1.0 / 200.0,  # 200s period (default)
        'phi_diffuse': 0.99,  # Default high correlation
        'sigma_diffuse_base': 0.05,
        'sigma_diffuse_rover': 0.1,
    }
    
    multipath_configs = [
        {
            'name': 'baseline',
            **baseline_config,  # All default parameters
        },
        # Vary specular amplitude (weak and slightly strong)
        {
            'name': 'specular_amp_weak',
            **{**baseline_config, 'specular_amp_base': 0.1, 'specular_amp_rover': 0.1},  # Only change amplitude
        },
        {
            'name': 'specular_amp_slightly_strong',
            **{**baseline_config, 'specular_amp_base': 0.4, 'specular_amp_rover': 0.4},  # Only change amplitude
        },
        # Vary specular frequency (fast and slow)
        {
            'name': 'specular_freq_fast',
            **{**baseline_config, 'specular_freq': 1.0 / 100.0},  # 100s period (fast, only change freq)
        },
        {
            'name': 'specular_freq_slow',
            **{**baseline_config, 'specular_freq': 1.0 / 300.0},  # 300s period (slow, only change freq)
        },
        # Vary phi_diffuse (low, medium, very high correlation)
        {
            'name': 'phi_diffuse_low',
            **{**baseline_config, 'phi_diffuse': 0.8},  # Low correlation (tau ~ 5s, only change phi)
        },
        {
            'name': 'phi_diffuse_medium',
            **{**baseline_config, 'phi_diffuse': 0.9},  # Medium correlation (tau ~ 10s, only change phi)
        },
        {
            'name': 'phi_diffuse_veryhigh',
            **{**baseline_config, 'phi_diffuse': 0.995},  # Very high correlation (tau ~ 200s, only change phi)
        },
    ]
    
    # Define scenarios
    scenarios = [
        {
            'name': 'multipath_thermal',
            'mode': 'multipath_dominant',
            'inject_nmm': 1.0
        },
        {
            'name': 'multipath_smm_thermal',
            'mode': 'mixed',
            'inject_nmm': 1.0
        },
        {
            'name': 'multipath_optimistic',
            'mode': 'multipath_dominant',
            'inject_nmm': 1/nmm_scale
        },
        {
            'name': 'multipath_permissive',
            'mode': 'multipath_dominant',
            'inject_nmm': nmm_scale
        },
        {
            'name': 'multipath_smm_optimistic',
            'mode': 'mixed',
            'inject_nmm': 1/nmm_scale
        },
        {
            'name': 'multipath_smm_permissive',
            'mode': 'mixed',
            'inject_nmm': nmm_scale
        },
    ]
    
    # Collect all results
    all_results = []
    
    # Test each multipath configuration
    for mp_config_dict in multipath_configs:
        mp_config_name = mp_config_dict.pop('name')
        mp_config = mp_config_dict
        
        print(f"\n\n{'#'*80}")
        print(f"# MULTIPATH CONFIGURATION: {mp_config_name.upper()}")
        print(f"{'#'*80}")
        
        # Test each scenario with this multipath configuration
        for scenario in scenarios:
            try:
                df_result = run_diagnosis_with_multipath_config(
                    name=f"{scenario['name']}_{mp_config_name}",
                    mode=scenario['mode'],
                    inject_nmm=scenario['inject_nmm'],
                    block_window_s=block_window_s,
                    obs_base=obs_base,
                    obs_rover=obs_rover,
                    sat_info=sat_info,
                    multipath_config=mp_config,
                    base_loc=base_loc,
                    rover_loc=rover_loc
                )
                
                # Add scenario and multipath config name
                df_result.insert(0, 'scenario', scenario['name'])
                df_result.insert(1, 'multipath_config', mp_config_name)
                
                all_results.append(df_result)
                
            except Exception as e:
                print(f"ERROR in {scenario['name']} with {mp_config_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f'dianew_multipath_test_{timestamp}_d{duration_hours}h_scale_{nmm_scale}.csv')
        
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
    
    parser = argparse.ArgumentParser(description='Test DiaNew algorithm with different multipath configurations')
    parser.add_argument('--duration', type=float, default=4.0,
                        help='Simulation duration in hours (default: 4.0)')
    parser.add_argument('--nmm-scale', type=float, default=4.0,
                        help='NMM scale factor (default: 2.0)')
    parser.add_argument('--output-dir', type=str, default='test_results',
                        help='Output directory for results (default: test_results)')
    args = parser.parse_args()
    
    test_dianew_multipath_sensitivity(
        baseline_km=1.0,
        duration_hours=args.duration,
        block_window_s=900.0,
        nmm_scale=args.nmm_scale,
        output_dir=args.output_dir
    )
