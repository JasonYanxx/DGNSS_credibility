"""
Stage 2: Error Injection Engine
Injects realistic errors with spatial correlation based on baseline length.
Uses real atmospheric delay models (Saastamoinen troposphere, Klobuchar ionosphere).
"""

import numpy as np
import pandas as pd
from atmospheric_models import (
    calculate_atmospheric_delays,
    generate_multipath_error,
    generate_thermal_noise_error
)
from datetime import datetime


def inject_errors(obs_base, obs_rover, baseline_length_km, base_loc=None, rover_loc=None, 
                   L_corr_km=30.0, sample_interval=1.0, use_real_models=False, scenario_mode=None,
                   multipath_config=None):
    """
    Inject spatially correlated errors into GNSS observations.
    Uses real atmospheric delay models (Saastamoinen troposphere, Klobuchar ionosphere).
    
    Parameters:
    -----------
    obs_base : pandas.DataFrame
        Base station observations (must contain: Time, ElevationAngle, SatelliteID, Pseudorange)
    obs_rover : pandas.DataFrame
        Rover station observations (must contain: Time, ElevationAngle, SatelliteID, Pseudorange)
    baseline_length_km : float
        Baseline length in km
    base_loc : tuple, optional
        (latitude_deg, longitude_deg, height_m) base station location.
        If None, estimated from observations (default: Boston area).
    rover_loc : tuple, optional
        (latitude_deg, longitude_deg, height_m) rover station location.
        If None, estimated from baseline_length_km (default: North of base).
    L_corr_km : float
        Correlation length scale in km (default 30 km)
    use_real_models : bool
        If True, use real atmospheric models. If False, use simplified Gauss-Markov process.
    scenario_mode : str, optional
        Commissioning scenario mode. Options:
        - None: Normal mode (baseline-dependent spatial correlation)
        - "multipath_dominant": Inject colored noise (Gauss-Markov) to simulate multipath
        - "system_fault_smm": Inject constant bias to simulate system fault
        - "optimism": Return underestimated covariance (no error injection change)
    
    Returns:
    --------
    obs_base_corrupted : pandas.DataFrame
        Corrupted base observations
    obs_rover_corrupted : pandas.DataFrame
        Corrupted rover observations
    sigma_dd : float
        Standard deviation of double difference errors
    """
    obs_base_corrupted = obs_base.copy()
    obs_rover_corrupted = obs_rover.copy()
    
    base_lat, base_lon, base_height = base_loc
    rover_lat, rover_lon, rover_height = rover_loc
    
    # Error parameters (for simplified model only)
    sigma_thermal = 0.10  # 10 cm thermal noise (simplified)
    sigma_tropo = 2.0  # 2 m troposphere error (1-sigma, simplified)
    sigma_iono = 1.5  # 1.5 m ionosphere error (1-sigma, simplified)
    
    # Spatial correlation coefficient
    if baseline_length_km == 0:
        rho = 1.0  # Perfect correlation at zero baseline
    else:
        rho = np.exp(-baseline_length_km / L_corr_km)
    
    # Initialize error state storage (per satellite ID)
    err_state = {}
    
    # Multipath parameters (from Proposal: Hybrid Model)
    # 1. Specular reflection (Main Sine Component)
    # Base station (better environment, weaker multipath)
    # specular_amp_range_base = (0.5, 1.0) 
    if multipath_config is not None:
        specular_amp_base = multipath_config.get('specular_amp_base', 0.2)  # m
        specular_amp_rover = multipath_config.get('specular_amp_rover', 0.2)  # m
        specular_freq = multipath_config.get('specular_freq', 1.0 / 200.0)  # Hz
        phi_diffuse = multipath_config.get('phi_diffuse', 0.99)
        sigma_diffuse_base = multipath_config.get('sigma_diffuse_base', 0.05)  # m
        sigma_diffuse_rover = multipath_config.get('sigma_diffuse_rover', 0.1)  # m
    else:
        # Default values
        specular_amp_base = 0.2  # m
        specular_amp_rover = 0.2  # m
        specular_freq = 1.0 / 200.0  # Hz (default: 200s period)
        phi_diffuse = 0.99
        sigma_diffuse_base = 0.05  # m
        sigma_diffuse_rover = 0.1  # m
    
    # 2. Diffuse reflection (Background AR(1) Noise)
    # Weakly correlated AR(1) process: eta_k = phi * eta_{k-1} + w_k
    # Base station (weaker diffuse multipath)
    sigma_diffuse_driving_base = sigma_diffuse_base * np.sqrt(1 - phi_diffuse**2) # Driving noise std
    # Rover station (stronger diffuse multipath)
    sigma_diffuse_driving_rover = sigma_diffuse_rover * np.sqrt(1 - phi_diffuse**2) # Driving noise std
    
    # 3. Total multipath error sigma
    sigma_multipat_base = np.sqrt(specular_amp_base**2/2 + sigma_diffuse_base**2)
    sigma_multipat_rover = np.sqrt(specular_amp_rover**2/2 + sigma_diffuse_rover**2)
    
    # Sample interval (seconds) - assume 30s if not available
    dt = sample_interval
    
    # Constant bias in meters (for SMM)
    constant_bias = 10

    # Initialize faulty satellite ID for consistent bias injection
    faulty_sat_id = None

    # Process each epoch
    for i in range(len(obs_base_corrupted)):
        base_row = obs_base_corrupted.iloc[i]
        rover_row = obs_rover_corrupted.iloc[i]
        
        sat_ids_base = np.array(base_row['SatelliteID'])
        sat_ids_rover = np.array(rover_row['SatelliteID'])
        pr_base = np.array(base_row['Pseudorange'])
        pr_rover = np.array(rover_row['Pseudorange'])
        elev_base = np.array(base_row['ElevationAngle'])
        elev_rover = np.array(rover_row['ElevationAngle'])
        
        # Get CN0 
        cn0_base = np.array(base_row['CN0'])
        cn0_rover = np.array(rover_row['CN0'])
        # Get azimuth
        az_base = np.array(base_row['AzimuthAngle'])
        az_rover = np.array(rover_row['AzimuthAngle'])
        # Get time for atmospheric models
        epoch_time = base_row['Time']
        gps_time = 0.0
        if isinstance(epoch_time, datetime):
            doy = epoch_time.timetuple().tm_yday
            gps_time = epoch_time.hour * 3600 + epoch_time.minute * 60 + epoch_time.second + epoch_time.microsecond / 1e6
        else:
            doy = 180  # Default to mid-year
            gps_time = float(i * dt)
        
        # Find common satellites
        common_sats = np.intersect1d(sat_ids_base, sat_ids_rover)
        if len(common_sats) == 0:
            continue
        
        # Get indices of common satellites
        base_idx = np.array([np.where(sat_ids_base == sat)[0][0] for sat in common_sats])
        rover_idx = np.array([np.where(sat_ids_rover == sat)[0][0] for sat in common_sats])
        
        num_sats = len(common_sats)
        
        # Initialize errors for this epoch
        err_tropo_base = np.zeros(num_sats)
        err_tropo_rover = np.zeros(num_sats)
        err_iono_base = np.zeros(num_sats)
        err_iono_rover = np.zeros(num_sats)
        
        # Generate errors per satellite
        for s, sat_id in enumerate(common_sats):
            sat_id_str = f'SAT{sat_id}'
            
            # Initialize state if new satellite
            if sat_id_str not in err_state:
                # Assign random specular amplitude for this satellite
                # Base: weaker (0.5-1.0m)
                # specular_amp_base = np.random.uniform(specular_amp_range_base[0], specular_amp_range_base[1])
                # Rover: stronger (1.0-2.0m)
                # specular_amp_rover = np.random.uniform(specular_amp_range_rover[0], specular_amp_range_rover[1])
                
                err_state[sat_id_str] = {
                    'tropo_base': 0.0,
                    'tropo_rover': 0.0,
                    'iono_base': 0.0,
                    'iono_rover': 0.0,
                    'diffuse_mp_base': 0.0, # Diffuse multipath state (AR process)
                    'diffuse_mp_rover': 0.0,
                    'specular_amp_base': specular_amp_base,
                    'specular_amp_rover': specular_amp_rover
                }
            
            if use_real_models:
                # Use real atmospheric delay models
                # Calculate real atmospheric delays for base station
                tropo_base_real, iono_base_real = calculate_atmospheric_delays(
                    elevation_deg=elev_base[base_idx[s]],
                    azimuth_deg=az_base[base_idx[s]],
                    lat_deg=base_lat,
                    lon_deg=base_lon,
                    height_m=base_height,
                    gps_time=gps_time,
                    doy=doy
                )
                
                # Calculate real atmospheric delays for rover station
                tropo_rover_real, iono_rover_real = calculate_atmospheric_delays(
                    elevation_deg=elev_rover[rover_idx[s]],
                    azimuth_deg=az_rover[rover_idx[s]],
                    lat_deg=rover_lat,
                    lon_deg=rover_lon,
                    height_m=rover_height,
                    gps_time=gps_time,
                    doy=doy
                )
                
                # Apply spatial correlation to real delays
                # The correlation models how similar the delays are at both stations
                err_tropo_base[s] = tropo_base_real
                err_tropo_rover[s] = tropo_rover_real
                
                err_iono_base[s] = iono_base_real
                err_iono_rover[s] = iono_rover_real
            else:
                # just for testing
                err_tropo_base[s] = 0
                err_tropo_rover[s] = 0
                err_iono_base[s] = 0
                err_iono_rover[s] = 0

                # # Use simplified Gauss-Markov process (original method)
                # # Generate base station errors (Gauss-Markov process)
                # # Troposphere: slow-varying (time constant ~1 hour)
                # tau_tropo = 3600.0  # seconds
                # alpha_tropo = np.exp(-dt / tau_tropo)
                # err_state[sat_id_str]['tropo_base'] = (
                #     alpha_tropo * err_state[sat_id_str]['tropo_base'] +
                #     np.sqrt(1 - alpha_tropo**2) * np.random.randn() * sigma_tropo
                # )
                # err_tropo_base[s] = err_state[sat_id_str]['tropo_base']
                
                # # Ionosphere: faster-varying (time constant ~15 minutes)
                # tau_iono = 900.0  # seconds
                # alpha_iono = np.exp(-dt / tau_iono)
                # err_state[sat_id_str]['iono_base'] = (
                #     alpha_iono * err_state[sat_id_str]['iono_base'] +
                #     np.sqrt(1 - alpha_iono**2) * np.random.randn() * sigma_iono
                # )
                # err_iono_base[s] = err_state[sat_id_str]['iono_base']
                
                # # Generate correlated rover errors
                # # E_rover = rho * E_base + sqrt(1 - rho^2) * Independent_Noise
                # err_tropo_rover[s] = (
                #     rho * err_tropo_base[s] +
                #     np.sqrt(1 - rho**2) * np.random.randn() * sigma_tropo
                # )
                # err_state[sat_id_str]['tropo_rover'] = err_tropo_rover[s]
                
                # err_iono_rover[s] = (
                #     rho * err_iono_base[s] +
                #     np.sqrt(1 - rho**2) * np.random.randn() * sigma_iono
                # )
                # err_state[sat_id_str]['iono_rover'] = err_iono_rover[s]
        
                # err_iono_rover[s] = err_state[sat_id_str]['iono_rover'] = err_iono_rover[s]
        
        # --- Multipath Error Injection: Hybrid Model (Specular + Diffuse) ---
        # Initialize multipath error arrays for this epoch for both base and rover
        err_multipath_base = np.zeros(num_sats)
        err_multipath_rover = np.zeros(num_sats)
        
        # Compute the sine component phase for this epoch
        # Note: We use satellite-specific amplitude, so we compute sine value per sat
        sine_phase = 2 * np.pi * specular_freq * gps_time
        
        # Loop over each common satellite in this epoch
        for s, sat_id in enumerate(common_sats):
            sat_id_str = f'SAT{sat_id}'
            
            # Retrieve amplitudes for this satellite
            amp_base = err_state[sat_id_str]['specular_amp_base']
            amp_rover = err_state[sat_id_str]['specular_amp_rover']
            
            # 1. Specular Reflection (Deterministic Sine)
            specular_val_base = amp_base * np.sin(sine_phase)
            specular_val_rover = amp_rover * np.sin(sine_phase)
            
            # 2. Diffuse Reflection (Stochastic AR(1))
            
            # --- Base Diffuse (Weaker) ---
            current_diffuse_base = err_state[sat_id_str]['diffuse_mp_base']
            noise_base = np.random.randn() * sigma_diffuse_driving_base
            new_diffuse_base = phi_diffuse * current_diffuse_base + noise_base
            err_state[sat_id_str]['diffuse_mp_base'] = new_diffuse_base
            
            # --- Rover Diffuse (Stronger, Independent) ---
            current_diffuse_rover = err_state[sat_id_str]['diffuse_mp_rover']
            noise_rover = np.random.randn() * sigma_diffuse_driving_rover
            new_diffuse_rover = phi_diffuse * current_diffuse_rover + noise_rover
            err_state[sat_id_str]['diffuse_mp_rover'] = new_diffuse_rover
            
            # Total Multipath = Specular + Diffuse
            err_multipath_base[s] = specular_val_base + new_diffuse_base
            err_multipath_rover[s] = specular_val_rover + new_diffuse_rover
        
        # # Thermal noise: C/N0-dependent, independent at each station
        # if use_real_models:
        #     # Use C/N0-dependent thermal noise model
        #     err_thermal_base = np.array([
        #         generate_thermal_noise_error(cn0_base[base_idx[s]], code_type='C/A')
        #         for s in range(num_sats)
        #     ])
        #     err_thermal_rover = np.array([
        #         generate_thermal_noise_error(cn0_rover[rover_idx[s]], code_type='C/A')
        #         for s in range(num_sats)
        #     ])
        # else:
        #     # Simplified: constant variance
        #     err_thermal_base = np.random.randn(num_sats) * sigma_thermal
        #     err_thermal_rover = np.random.randn(num_sats) * sigma_thermal

        # Thermal noise Simplified: constant variance
        err_thermal_base = np.random.randn(num_sats) * sigma_thermal
        err_thermal_rover = np.random.randn(num_sats) * sigma_thermal
        
        # ========================================================================
        # COMMISSIONING SCENARIO MODE: Override error sources if scenario is set
        # ========================================================================
        if scenario_mode is not None:
            if scenario_mode == "multipath_dominant":
                # Scenario A: Inject colored noise (Gauss-Markov) to simulate multipath
                # Use strong multipath on rover, minimal on base (short baseline)
                # Multipath is already computed above, so we just enable it
                err_total_base = err_multipath_base + err_thermal_base
                err_total_rover = err_multipath_rover + err_thermal_rover
                
            elif scenario_mode == "system_fault_smm":
                # Scenario B: Inject constant bias (System Model Mismatch)
                # Add a constant bias to ONE rover measurement (randomly selected but consistent)
                # Determine faulty satellite (once)
                if faulty_sat_id is None and len(common_sats) > 0:
                    # Pick one visible satellite to be faulty
                    faulty_sat_id = np.random.choice(common_sats)

                bias_vector = np.zeros(num_sats)
                if faulty_sat_id is not None and faulty_sat_id in common_sats:
                    idx = np.where(common_sats == faulty_sat_id)[0][0]
                    bias_vector[idx] = constant_bias

                err_total_base = err_thermal_base
                err_total_rover = err_thermal_rover + bias_vector
                
            elif scenario_mode == "mixed":
                # Scenario C: Mixed (Multipath + Bias)
                # Combine multipath and constant bias
                # Determine faulty satellite (once) - share logic
                if faulty_sat_id is None and len(common_sats) > 0:
                    faulty_sat_id = np.random.choice(common_sats)

                bias_vector = np.zeros(num_sats)
                if faulty_sat_id is not None and faulty_sat_id in common_sats:
                    idx = np.where(common_sats == faulty_sat_id)[0][0]
                    bias_vector[idx] = constant_bias

                err_total_base = err_multipath_base + err_thermal_base
                err_total_rover = err_multipath_rover + err_thermal_rover + bias_vector
                
            elif scenario_mode == "healthy_no_multipath":
                # Scenario E: Healthy with no multipath
                # Just use thermal noise (no additional error sources)
                err_total_base = err_thermal_base
                err_total_rover = err_thermal_rover
            else:
                raise ValueError(f"Unknown scenario_mode: {scenario_mode}")
        else:
            # Normal mode: Use baseline-dependent error model
            use_atomospher_error = True
            use_multipath_error = False
            use_thermal_noise_error = False
            err_total_base = 0
            err_total_rover = 0
            if use_atomospher_error:
                err_total_base = err_total_base + err_tropo_base + err_iono_base
                err_total_rover = err_total_rover + err_tropo_rover + err_iono_rover
            if use_multipath_error:
                err_total_base = err_total_base + err_multipath_base
                err_total_rover = err_total_rover + err_multipath_rover
            if use_thermal_noise_error:
                err_total_base = err_total_base + err_thermal_base
                err_total_rover = err_total_rover + err_thermal_rover
        
        # Apply errors to pseudoranges
        pr_base_corr = pr_base.copy()
        pr_rover_corr = pr_rover.copy()
        
        pr_base_corr[base_idx] = pr_base[base_idx] + err_total_base
        pr_rover_corr[rover_idx] = pr_rover[rover_idx] + err_total_rover
        
        # Store back
        obs_base_corrupted.at[i, 'Pseudorange'] = pr_base_corr.tolist()
        obs_rover_corrupted.at[i, 'Pseudorange'] = pr_rover_corr.tolist()
    
    # std of double difference
    if scenario_mode == "multipath_dominant":
        # Include multipath in sigma_dd
        sigma_dd = np.sqrt(sigma_thermal**2*4 + sigma_multipat_base**2*2 + sigma_multipat_rover**2*2)
    elif scenario_mode == "system_fault_smm":
        # Bias doesn't affect sigma, only thermal noise
        sigma_dd = np.sqrt(sigma_thermal**2*4)
    elif scenario_mode == "mixed":
        # Include multipath in sigma_dd, bias is mean offset
        sigma_dd = np.sqrt(sigma_thermal**2*4 + sigma_multipat_base**2*2 + sigma_multipat_rover**2*2)
    elif scenario_mode == "healthy_no_multipath":
        # Healthy with no multipath
        sigma_dd = np.sqrt(sigma_thermal**2*4)
    else:
        # Normal mode
        sigma_dd = np.sqrt(sigma_thermal**2*4 + sigma_multipat_base**2*2 + sigma_multipat_rover**2*2)
    
    return obs_base_corrupted, obs_rover_corrupted, sigma_dd

