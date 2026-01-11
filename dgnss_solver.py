"""
Stage 3: DGNSS Solver
Implements Double-Difference (DD) Weighted Least Squares solver.
"""

import numpy as np
import pandas as pd
from skyfield.api import wgs84


def solve_dgnss(obs_base, obs_rover, sat_info, base_loc, rover_loc,
                stochastic_model='elevation', sigma_dd=0.2, decimation_factor=1):
    """
    Solve DGNSS using Double Difference Weighted Least Squares.
    
    Parameters:
    -----------
    obs_base : pandas.DataFrame
        Base station observations with columns: Time, SatelliteID, Pseudorange, 
        ElevationAngle, CN0, EarthRotationCorrection
    obs_rover : pandas.DataFrame
        Rover station observations (same structure as obs_base)
    sat_info : pandas.DataFrame
        Satellite positions with columns: Time, SatelliteID, SatPos (ECEF [x, y, z])
    base_loc : tuple
        (latitude_deg, longitude_deg, height_m) base station location
    rover_loc : numpy.ndarray or tuple
        [latitude_deg, longitude_deg, height_m] rover station location
    stochastic_model : str
        Weighting strategy: 'equal_weight', 'elevation', 'cn0'
    sigma_dd : float
        Standard deviation of double difference errors (meters)
    decimation_factor : int
        Decimation factor for solution epochs (1 = no decimation)
    
    Returns:
    --------
    epochs_dec : list
        List of datetime objects for each solution epoch
    residuals : list
        List of residual vectors (one per epoch)
    covariance : list
        List of covariance matrices for residuals
    elevation : list
        List of elevation angles (average of base and rover)
    snr : list
        List of CN0 values
    xt_hat : list
        List of estimated rover positions (ECEF, 3D)
    Q_xyz : list
        List of position covariance matrices (3x3)
    error_vecs : list
        List of position error vectors (3D, relative to true position)
    """
    # Convert locations to ECEF
    base_lat, base_lon, base_height = base_loc
    if isinstance(rover_loc, np.ndarray):
        rover_lat, rover_lon, rover_height = rover_loc[0], rover_loc[1], rover_loc[2]
    else:
        rover_lat, rover_lon, rover_height = rover_loc
    
    base_pos_wgs = wgs84.latlon(base_lat, base_lon, elevation_m=base_height)
    base_pos_ecef = base_pos_wgs.itrs_xyz.m  # ECEF [x, y, z] in meters
    
    rover_pos_wgs = wgs84.latlon(rover_lat, rover_lon, elevation_m=rover_height)
    rover_pos_ecef = rover_pos_wgs.itrs_xyz.m  # True rover position (ECEF)
    
    # Storage for results
    epochs_dec = []
    residuals = []
    covariance = []
    elevation = []
    snr = []
    xt_hat = []
    Q_xyz = []
    error_vecs = []
    
    # Process each epoch
    epoch_counter = 0
    for idx, (base_row, rover_row) in enumerate(zip(obs_base.itertuples(), obs_rover.itertuples())):
        # Apply decimation
        if idx % decimation_factor != 0:
            continue
        
        epoch_dt = base_row.Time
        
        # Extract data for this epoch
        sat_ids = base_row.SatelliteID  # List of satellite IDs
        pr_base = np.array(base_row.Pseudorange)  # Base pseudoranges
        pr_rover = np.array(rover_row.Pseudorange)  # Rover pseudoranges
        er_base = np.array(base_row.EarthRotationCorrection)  # Earth rotation correction
        er_rover = np.array(rover_row.EarthRotationCorrection)
        elev_base = np.array(base_row.ElevationAngle)  # Base elevations
        elev_rover = np.array(rover_row.ElevationAngle)  # Rover elevations
        cn0_base = np.array(base_row.CN0)  # Base CN0
        cn0_rover = np.array(rover_row.CN0)  # Rover CN0
        
        # Get satellite positions for this epoch from sat_info
        epoch_sat_info = sat_info[pd.to_datetime(sat_info['Time']) == pd.to_datetime(epoch_dt)]
        
        if len(epoch_sat_info) == 0:
            continue  # Skip if no satellite info available
        
        # Ensure satellite IDs match
        sat_info_ids = epoch_sat_info['SatelliteID'].tolist()
        if len(sat_ids) != len(sat_info_ids) or sat_ids != sat_info_ids:
            continue  # Skip if mismatch
        
        # Extract satellite positions (ECEF)
        sat_positions = []
        for sat_id in sat_ids:
            sat_row = epoch_sat_info[epoch_sat_info['SatelliteID'] == sat_id]
            if len(sat_row) == 0:
                break
            sat_positions.append(np.array(sat_row.iloc[0]['SatPos']))
        
        if len(sat_positions) != len(sat_ids):
            continue  # Skip if positions don't match
        
        sat_positions = np.array(sat_positions)  # Shape: (N_sat, 3)
        
        # Need at least 4 satellites for 3D positioning
        if len(sat_ids) < 4:
            continue
        
        # Select reference satellite (highest elevation at base station)
        ref_idx = np.argmax(elev_base)
        
        # Form double differences
        # Correct for Earth rotation
        pr_base_corr = pr_base - er_base
        pr_rover_corr = pr_rover - er_rover
        
        # Single differences: SD = PR_rover - PR_base
        sd = pr_rover_corr - pr_base_corr
        
        # Double differences: DD_i = SD_i - SD_ref
        dd_meas = sd - sd[ref_idx]
        
        # Remove reference satellite from all arrays
        mask = np.arange(len(sat_ids)) != ref_idx
        dd_meas = dd_meas[mask]
        sat_pos_nonref = sat_positions[mask]
        sat_pos_ref = sat_positions[ref_idx]
        elev_avg = (elev_base[mask] + elev_rover[mask]) / 2.0
        cn0_avg = (cn0_base[mask] + cn0_rover[mask]) / 2.0
        
        n_dd = len(dd_meas)
        
        # Initial guess for rover position (use true position + small random offset)
        x_init = rover_pos_ecef + np.random.randn(3) * 1.0  # 1m initial offset
        
        # Iterative Least Squares (Gauss-Newton)
        x_est = x_init.copy()
        max_iter = 10
        convergence_threshold = 1e-4
        
        for iteration in range(max_iter):
            # Compute geometric ranges from current estimate
            range_base_ref = np.linalg.norm(sat_pos_ref - base_pos_ecef)
            range_rover_ref_est = np.linalg.norm(sat_pos_ref - x_est)
            
            range_base_nonref = np.linalg.norm(sat_pos_nonref - base_pos_ecef, axis=1)
            range_rover_nonref_est = np.linalg.norm(sat_pos_nonref - x_est, axis=1)
            
            # Predicted double differences
            dd_pred = (range_rover_nonref_est - range_base_nonref) - (range_rover_ref_est - range_base_ref)
            
            # Residuals
            res = dd_meas - dd_pred  # (n_dd,)
            
            # Geometry matrix (Jacobian)
            # H_i = (u_rover_i - u_rover_ref)
            # where u = (sat_pos - rover_pos) / ||sat_pos - rover_pos||
            u_rover_ref = (sat_pos_ref - x_est) / range_rover_ref_est
            u_rover_nonref = (sat_pos_nonref - x_est) / range_rover_nonref_est[:, np.newaxis]
            
            H = u_rover_ref - u_rover_nonref  # Shape: (n_dd, 3)
            
            # Weighting matrix
            if stochastic_model == 'equal_weight':
                # Account for mathematical correlation in DDs (shared reference satellite)
                # print("DEBUG: Using Corrected W matrix")
                # Covariance R has sigma_dd^2 on diagonal and sigma_dd^2/2 on off-diagonal
                # R = (sigma_dd^2/2) * (I + J)
                # W = R^-1 = (2/sigma_dd^2) * (I - 1/(n_dd+1) * J)
                factor = 2.0 / (sigma_dd**2)
                J_mat = np.ones((n_dd, n_dd))
                I_mat = np.eye(n_dd)
                W = factor * (I_mat - (1.0 / (n_dd + 1)) * J_mat)
            elif stochastic_model == 'elevation':
                # Elevation-based weighting: weight = sin(elevation)^2
                sin_elev = np.sin(np.deg2rad(elev_avg))
                weights = sin_elev**2
                W = np.diag(weights) / (sigma_dd**2)
            elif stochastic_model == 'cn0':
                # CN0-based weighting
                weights = (cn0_avg / 45.0)**2  # Normalize to typical CN0
                W = np.diag(weights) / (sigma_dd**2)
            else:
                W = np.eye(n_dd) / (sigma_dd**2)
            
            # Weighted Least Squares update
            try:
                N = H.T @ W @ H  # Normal matrix (3x3)
                b = H.T @ W @ res  # Right-hand side (3,)
                dx = np.linalg.solve(N, b)
            except np.linalg.LinAlgError:
                # Singular matrix, skip this epoch
                break
            
            # Update estimate
            x_est = x_est + dx
            
            # Check convergence
            if np.linalg.norm(dx) < convergence_threshold:
                break
        
        # Final residuals and covariance
        range_base_ref = np.linalg.norm(sat_pos_ref - base_pos_ecef)
        range_rover_ref_est = np.linalg.norm(sat_pos_ref - x_est)
        range_base_nonref = np.linalg.norm(sat_pos_nonref - base_pos_ecef, axis=1)
        range_rover_nonref_est = np.linalg.norm(sat_pos_nonref - x_est, axis=1)
        dd_pred = (range_rover_nonref_est - range_base_nonref) - (range_rover_ref_est - range_base_ref)
        res_final = dd_meas - dd_pred
        
        # Geometry matrix (final)
        u_rover_ref = (sat_pos_ref - x_est) / range_rover_ref_est
        u_rover_nonref = (sat_pos_nonref - x_est) / range_rover_nonref_est[:, np.newaxis]
        H_final = u_rover_ref - u_rover_nonref
        
        # Weighting matrix (final)
        if stochastic_model == 'equal_weight':
            # Account for mathematical correlation in DDs
            factor = 2.0 / (sigma_dd**2)
            J_mat = np.ones((n_dd, n_dd))
            I_mat = np.eye(n_dd)
            W_final = factor * (I_mat - (1.0 / (n_dd + 1)) * J_mat)
        elif stochastic_model == 'elevation':
            sin_elev = np.sin(np.deg2rad(elev_avg))
            weights = sin_elev**2
            W_final = np.diag(weights) / (sigma_dd**2)
        elif stochastic_model == 'cn0':
            weights = (cn0_avg / 45.0)**2
            W_final = np.diag(weights) / (sigma_dd**2)
        else:
            W_final = np.eye(n_dd) / (sigma_dd**2)
        
        # Position covariance
        try:
            N_final = H_final.T @ W_final @ H_final
            Q_xyz_k = np.linalg.inv(N_final)  # 3x3 covariance matrix
        except np.linalg.LinAlgError:
            Q_xyz_k = np.eye(3) * 100.0  # Large uncertainty if singular
        
        # Residual covariance
        R_res = np.linalg.inv(W_final)  # (n_dd, n_dd)
        
        # Position error (relative to true position)
        error_vec = x_est - rover_pos_ecef
        
        # Store results
        epochs_dec.append(epoch_dt)
        residuals.append(res_final)
        covariance.append(R_res)
        elevation.append(elev_avg)
        snr.append(cn0_avg)
        xt_hat.append(x_est)
        Q_xyz.append(Q_xyz_k)
        error_vecs.append(error_vec)
        
        epoch_counter += 1
    
    return epochs_dec, residuals, covariance, elevation, snr, xt_hat, Q_xyz, error_vecs
