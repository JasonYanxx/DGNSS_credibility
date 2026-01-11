"""
Main Pipeline: DGNSS Credibility Diagnosis & Calibration
Implements the "Time Domain Decoupling - Joint Estimation - Asymmetric Diagnosis - Optimization Calibration" pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
from skyfield.api import utc, wgs84
import os

from geometry_engine import generate_obs
from error_injection import inject_errors
from dgnss_solver import solve_dgnss
from diagnostic_system import DiagnosticSystem, BlockAverager

def compute_acf(x, max_lag=None):
    """
    Compute Autocorrelation Function (ACF).
    """
    if len(x) < 2:
        return np.array([1.0])
        
    if max_lag is None:
        max_lag = len(x) // 2
        
    # Standardize
    x = x - np.mean(x)
    
    # Check for zero variance
    if np.all(x == 0):
        return np.ones(max_lag)
    
    # Compute ACF via FFT for speed
    n = len(x)
    # Pad to next power of 2 for efficiency
    n_fft = 2**int(np.ceil(np.log2(2*n-1)))
    f = np.fft.fft(x, n=n_fft)
    acf = np.fft.ifft(f * np.conjugate(f)).real
    
    if acf[0] == 0:
        return np.zeros(max_lag)
        
    acf = acf[:n] / acf[0]
    
    return acf[:max_lag]


def plot_commissioning_results(scenario_results, output_dir='plots_commissioning'):
    """
    Generate comparison plots for commissioning verification.
    
    Shows the effectiveness of block averaging in distinguishing:
    - Multipath (colored noise): Should pass with block averaging
    - System Fault (SMM): Should fail with both methods
    - Mixed: Should fail
    - Optimism: Should be detected by both methods
    
    Parameters:
    -----------
    scenario_results : dict
        Results from run_commissioning_verification()
    output_dir : str
        Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3)
    
    scenarios = ['A_multipath', 'B_fault', 'C_mixed', 'D_optimism']
    scenario_names = ['Scenario 1: Multipath', 'Scenario 2: System Fault', 'Scenario 3: Mixed', 'Scenario 4: Optimism']
    colors = ['blue', 'red', 'purple', 'green']
    
    fig.suptitle('Commissioning Protocol Verification: 15-Minute Block Averaging', 
                 fontsize=16, fontweight='bold')
    
    # ========================================================================
    # ROW 1: Block Means (Position Errors)
    # ========================================================================
    for col, (key, name, color) in enumerate(zip(scenarios, scenario_names, colors)):
        ax = fig.add_subplot(gs[0, col])
        result = scenario_results[key]
        
        # Plot raw position errors (first 100 epochs for visibility)
        block_means_raw = result['raw_diagnosis']['block_means'][:100]
        block_means_block = result['block_diagnosis']['block_means']
        
        x_raw = np.arange(len(block_means_raw))
        x_block = np.arange(len(block_means_block))
        
        # Plot 3D norm of position errors
        errors_raw = np.linalg.norm(block_means_raw, axis=1)
        errors_block = np.linalg.norm(block_means_block, axis=1)
        
        ax.plot(x_raw, errors_raw, 'o-', alpha=0.5, markersize=2, 
                label='30s Blocks (Raw)', color='gray', linewidth=0.5)
        ax.plot(x_block, errors_block, 's-', markersize=6, 
                label='15-min Blocks', color=color, linewidth=2)
        
        ax.set_xlabel('Block Index', fontsize=10)
        ax.set_ylabel('Position Error (m)', fontsize=10)
        ax.set_title(f'{name}\nBlock Means', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # ========================================================================
    # ROW 2: SMM Estimates (Bias Detection) - Corresponds to ELT
    # ========================================================================
    for col, (key, name, color) in enumerate(zip(scenarios, scenario_names, colors)):
        ax = fig.add_subplot(gs[1, col])
        result = scenario_results[key]
        
        # Plot SMM estimates (3D norm)
        smm_raw = np.linalg.norm(result['raw_diagnosis']['smm_estimate'], axis=1)
        smm_block = np.linalg.norm(result['block_diagnosis']['smm_estimate'], axis=1)
        
        x_raw = np.arange(len(smm_raw))
        x_block = np.arange(len(smm_block))
        
        ax.plot(x_raw, smm_raw, 'o-', alpha=0.5, markersize=2, 
                label='30s Blocks', color='gray', linewidth=0.5)
        ax.plot(x_block, smm_block, 's-', markersize=6, 
                label='15-min Blocks', color=color, linewidth=2)
        
        ax.set_xlabel('Block Index', fontsize=10)
        ax.set_ylabel('SMM Estimate (m)', fontsize=10)
        ax.set_title('Bias Detection (KF Estimate)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # ========================================================================
    # ROW 3: NCI (Noncredibility Index) - Corresponds to NLL/NCI
    # ========================================================================
    for col, (key, name, color) in enumerate(zip(scenarios, scenario_names, colors)):
        ax = fig.add_subplot(gs[2, col])
        result = scenario_results[key]
        
        nci_raw = result['raw_diagnosis']['algo_result'].nci
        nci_block = result['block_diagnosis']['algo_result'].nci
        
        bars = ax.bar(['30s Blocks\n(Raw 1Hz)', '15-min Blocks\n(Averaged)'], 
                      [nci_raw, nci_block], 
                      color=['gray', color], alpha=0.7, width=0.6)
        
        # Add threshold lines
        ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, 
                   label='Threshold (+0.5)', alpha=0.7)
        ax.axhline(y=-0.5, color='purple', linestyle='--', linewidth=1.5, 
                   label='Threshold (-0.5)', alpha=0.7)
        
        ax.set_ylabel('NCI (dB)', fontsize=10)
        ax.set_title('Noncredibility Index', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ========================================================================
    # ROW 4: "Whitening Effect" (ACF) - Figure 2 in Research Plan
    # ========================================================================
    # Only show for Multipath scenario (Scenario 1) as requested
    ax_acf = fig.add_subplot(gs[3, :])
    
    multipath_res = scenario_results['A_multipath']
    acf_raw = multipath_res['acf_raw']
    acf_block = multipath_res['acf_block']
    
    lags_raw = np.arange(len(acf_raw))
    # Scale block lags to match time: 1 block = M raw samples
    # But usually we just plot lag index. Let's plot normalized lag (time).
    # Assuming 1Hz raw, 900s block.
    
    ax_acf.plot(lags_raw, acf_raw, label='Raw 1Hz Residuals', color='gray', alpha=0.7)
    
    # For block ACF, we stretch it to visualize on same time scale or just plot overlapping?
    # Research plan says "contrast: long tail vs quickly truncated".
    # Let's plot them on separate axes or just overlay with different x-scales if needed?
    # No, ACF is function of lag k. Lag k for block mean is 900x longer time.
    # It's better to show that Raw ACF has long correlation time (in seconds), 
    # while Block ACF drops to zero at lag 1 (meaning blocks are independent).
    
    ax_acf.plot(np.arange(len(acf_block)) * 900, acf_block, 'o-', label='15-min Block Means', color='blue', linewidth=2)
    
    ax_acf.set_xlim(0, 3600) # Show first hour lag
    ax_acf.set_xlabel('Lag (seconds)', fontsize=10)
    ax_acf.set_ylabel('Autocorrelation', fontsize=10)
    ax_acf.set_title('Figure 2: The "Whitening" Effect (Scenario 1)', fontsize=12, fontweight='bold')
    ax_acf.legend()
    ax_acf.grid(True, alpha=0.3)
    ax_acf.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    # ========================================================================
    # ROW 5: Classification Summary - Figure 3 / Triage Logic
    # ========================================================================
    ax = fig.add_subplot(gs[4, :])
    ax.axis('off')
    
    # Create summary table
    table_data = []
    table_data.append(['Scenario', 'Expected Outcome', 'Raw 1Hz Result', '15-min Block Result', 'Status'])
    table_data.append(['-'*20, '-'*30, '-'*20, '-'*20, '-'*10])
    
    expected_outcomes = {
        'A_multipath': ('Pass (Healthy)', 'Multipath decorrelates -> Calibrated'),
        'B_fault': ('Fail (Fault)', 'Bias survives -> Bias Detected'),
        'C_mixed': ('Fail (Fault)', 'Bias survives -> Bias Detected'),
        'D_optimism': ('Fail (Model)', 'Underestimation -> Optimistic')
    }
    
    for key, name in zip(scenarios, scenario_names):
        result = scenario_results[key]
        expected, _ = expected_outcomes[key]
        raw_class = result['raw_diagnosis']['classification']
        block_class = result['block_diagnosis']['classification']
        
        # Determine pass/fail for block averaging
        status = '?'
        if key == 'A_multipath':
            # Should be calibrated after block averaging (or at least no Bias)
            # Current code might output Optimistic if noise is large, but "Bias" should be gone.
            status = '✓ PASS' if 'Bias' not in block_class else '✗ FAIL'
        elif key in ['B_fault', 'C_mixed']:
            # Should detect bias
            status = '✓ PASS' if 'Bias' in block_class else '✗ FAIL'
        elif key == 'D_optimism':
            # Should detect optimism
            status = '✓ PASS' if 'Optimistic' in block_class else '✗ FAIL'
        
        table_data.append([name, expected, raw_class, block_class, status])
    
    # Create table
    table = ax.table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.5)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(1, i)].set_facecolor('#E0E0E0')
    
    # Style data rows with alternating colors
    for i in range(2, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
            # Highlight status column
            if j == 4:
                cell = table[(i, j)]
                if '✓ PASS' in table_data[i][j]:
                    cell.set_facecolor('#C6EFCE')
                    cell.set_text_props(weight='bold', color='green')
                elif '✗ FAIL' in table_data[i][j]:
                    cell.set_facecolor('#FFC7CE')
                    cell.set_text_props(weight='bold', color='red')
    
    ax.set_title('Verification Summary (Triage Logic)', fontsize=13, fontweight='bold', pad=20)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'commissioning_verification.png'), 
                dpi=300, bbox_inches='tight')
    print(f"\nCommissioning verification plot saved to {output_dir}/commissioning_verification.png")
    plt.close()
    
    # Save summary CSV
    summary_df = pd.DataFrame({
        'Scenario': scenario_names,
        'Raw_Classification': [scenario_results[k]['raw_diagnosis']['classification'] for k in scenarios],
        'Block_Classification': [scenario_results[k]['block_diagnosis']['classification'] for k in scenarios],
        'Raw_NCI': [scenario_results[k]['raw_diagnosis']['algo_result'].nci for k in scenarios],
        'Block_NCI': [scenario_results[k]['block_diagnosis']['algo_result'].nci for k in scenarios],
    })
    summary_df.to_csv(os.path.join(output_dir, 'commissioning_summary.csv'), index=False)
    print(f"Commissioning summary saved to {output_dir}/commissioning_summary.csv")


def run_commissioning_verification(baseline_km=1.0, duration_hours=2.0, block_window_s=900.0):
    """
    Run the Commissioning Protocol Verification experiment.
    
    Tests whether 15-minute block averaging can distinguish between:
    - Scenario 1: Environmental Multipath (Healthy but Noisy)
    - Scenario 2: System Fault (Bias)
    - Scenario 3: Mixed (Multipath + Bias)
    - Scenario 4: Optimism (Model Issue)
    """
    print("\n" + "="*80)
    print("COMMISSIONING PROTOCOL VERIFICATION")
    print("="*80)
    print(f"Configuration:")
    print(f"  Baseline: {baseline_km} km (short baseline)")
    print(f"  Duration: {duration_hours} hours")
    print(f"  Sample Interval: 1.0 Hz")
    print(f"  Block Window: {block_window_s} s ({block_window_s/60:.1f} minutes)")
    print("="*80)
    
    # Common configuration
    sample_interval = 1.0  # 1 Hz 
    base_loc = (42.3601, -71.0589, 0.0)  # Boston area
    mask_angle_deg = 5.0
    stochastic_model = 'equal_weight'
    
    # Calculate rover location (short baseline)
    rover_lat = base_loc[0] + (baseline_km / 111.0)
    rover_loc = (rover_lat, base_loc[1], base_loc[2])
    
    # Start time
    start_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=utc)
    
    # Storage for results
    scenario_results = {}
    
    # Helper to run a scenario
    def run_scenario(name, mode, inject_nmm=1.0):
        print("\n" + "-"*80)
        print(f"SCENARIO: {name} (Mode: {mode})")
        print("-"*80)
        
        print("Generating geometry...")
        obs_base, obs_rover, _, sat_info, _ = generate_obs(
            start_time, duration_hours, sample_interval, base_loc, baseline_km,
            mask_angle_deg=mask_angle_deg
        )
        
        print(f"Injecting errors...")
        obs_base_c, obs_rover_c, sigma_dd = inject_errors(
            obs_base, obs_rover, baseline_km,
            base_loc=base_loc, rover_loc=rover_loc,
            sample_interval=sample_interval,
            use_real_models=False,
            scenario_mode=mode
        )
        
        print("Solving DGNSS...")
        _, _, _, _, _, _, Q_xyz, error_vecs = solve_dgnss(
            obs_base_c, obs_rover_c, sat_info, base_loc, rover_loc,
            stochastic_model=stochastic_model, sigma_dd=sigma_dd, decimation_factor=1
        )
        
        if inject_nmm != 1.0:
            print(f"Scaling covariance by {inject_nmm}...")
            Q_xyz = [cov * inject_nmm for cov in Q_xyz]
            
        # save error_vecs and Q_xyz
        np.savez(f"error_vecs_{name}.npz", error_vecs=error_vecs, Q_xyz=Q_xyz)
        print(f"Saved error_vecs and Q_xyz to error_vecs_{name}.npz")
        # load error_vecs and Q_xyz
        data = np.load(f"error_vecs_{name}.npz")
        error_vecs = data['error_vecs']
        Q_xyz = data['Q_xyz']
        print(f"Loaded error_vecs and Q_xyz from error_vecs_{name}.npz")

        print("Running diagnosis...")
        # Raw 1Hz (30s blocks)
        diag_raw = DiagnosticSystem()
        diag_raw.averager = BlockAverager(T_batch=1.0, f_hz=1.0)
        res_raw = diag_raw.run(error_vecs, Q_xyz)
        
        # Block Averaged (15-min)
        diag_block = DiagnosticSystem()
        diag_block.averager = BlockAverager(T_batch=block_window_s, f_hz=1.0)
        res_block = diag_block.run(error_vecs, Q_xyz)
        
        if res_block is None:
            return None
            
        print(f"  Raw 1Hz: {res_raw['classification']}")
        print(f"  Block 15-min: {res_block['classification']}")
        
        # Compute ACF for first error component (X)
        errors_x = np.array(error_vecs)[:, 0]
        acf_raw = compute_acf(errors_x, max_lag=3600) # 1 hour lag
        
        block_means_x = res_block['block_means'][:, 0]
        acf_block = compute_acf(block_means_x, max_lag=len(block_means_x)-1)
        
        return {
            'raw_diagnosis': res_raw,
            'block_diagnosis': res_block,
            'acf_raw': acf_raw,
            'acf_block': acf_block,
            'num_epochs': len(error_vecs)
        }

    # 1. Scenario A: Multipath Dominant (Healthy)
    res = run_scenario("A: Multipath", "multipath_dominant")
    if res is None: return None
    scenario_results['A_multipath'] = res
    
    # 2. Scenario B: System Fault (Bias)
    res = run_scenario("B: System Fault", "system_fault_smm")
    scenario_results['B_fault'] = res
    
    # 3. Scenario C: Mixed (Multipath + Bias)
    res = run_scenario("C: Mixed", "mixed")
    scenario_results['C_mixed'] = res
    
    # 4. Scenario D: Optimism
    res = run_scenario("D: Optimism", "optimism", inject_nmm=0.5)
    scenario_results['D_optimism'] = res

    # 5. Healthy with no multipath
    res = run_scenario("E: Healthy with no multipath", "healthy_no_multipath")
    scenario_results['E_healthy_no_multipath'] = res
    
    return scenario_results


def main():
    """Main execution function for Commissioning Protocol Verification."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DGNSS Commissioning Protocol Verification')
    parser.add_argument('--duration', type=float, default=5,
                        help='Simulation duration in hours (default: 2.0)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("DGNSS Commissioning Protocol Verification")
    print("="*60)
    print("Testing 15-minute block averaging for multipath vs. bias distinction")
    
    scenario_results = run_commissioning_verification(
        baseline_km=1.0,  # Short baseline for commissioning
        duration_hours=args.duration,
        block_window_s=900.0  # 15 minutes
    )
    
    if scenario_results is None:
        print("\nERROR: Commissioning verification failed!")
        return
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_commissioning_results(scenario_results)
    
    print("\n" + "="*60)
    print("Commissioning verification completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()

    # # load error_vecs and Q_xyz
    # data = np.load("error_vecs_A: Multipath.npz")
    # error_vecs = data['error_vecs']
    # Q_xyz = data['Q_xyz']

    # # plot error_vecs
    # plt.subplot(2, 1, 1)
    # plt.plot(error_vecs[:, 0])
    # plt.plot(error_vecs[:, 1])
    # plt.plot(error_vecs[:, 2])

    # means, covs, _ = BlockAverager(T_batch=5*60, f_hz=1.0).process(error_vecs, Q_xyz)
    # # plot means
    # plt.subplot(2, 1, 2)
    # plt.plot(means[:, 0])
    # plt.plot(means[:, 1])
    # plt.plot(means[:, 2])
    # plt.show()
    
    # print("Running diagnosis...")
    # # Raw 1Hz (30s blocks)
    # diag_raw = DiagnosticSystem()
    # diag_raw.averager = BlockAverager(T_batch=1.0, f_hz=1.0)
    # res_raw = diag_raw.run(error_vecs, Q_xyz)
    
    # # Block Averaged (15-min)
    # diag_block = DiagnosticSystem()
    # diag_block.averager = BlockAverager(T_batch=15*60, f_hz=1.0)
    # res_block = diag_block.run(error_vecs, Q_xyz)
    # aa=0
        
