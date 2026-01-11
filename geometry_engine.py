"""
Stage 1: Geometry Engine
Generates true satellite positions and geometric ranges for Base and Rover stations.
"""

import numpy as np
import pandas as pd
from skyfield.api import load, wgs84, EarthSatellite, utc
from datetime import datetime, timedelta


def generate_obs(start_time, duration_hours, sample_interval, base_loc, baseline_length_km, mask_angle_deg=0.0):
    """
    Generate clean GNSS observations with true geometric ranges.
    
    Parameters:
    -----------
    start_time : datetime
        Simulation start time
    duration_hours : float
        Simulation duration in hours
    sample_interval : float
        Sample interval in seconds
    base_loc : tuple
        (latitude_deg, longitude_deg, height_m) base station location
    baseline_length_km : float
        Distance from base to rover in km (moves North)
    mask_angle_deg : float, optional
        Elevation mask angle in degrees (default: 0.0). Satellites below this 
        elevation will be excluded. Typical values: 5-15 degrees.
    
    Returns:
    --------
    obs_base : pandas.DataFrame
        Base station observations with columns: Time, SatelliteID, Pseudorange, ElevationAngle, CN0
    obs_rover : pandas.DataFrame
        Rover station observations
    rover_loc : numpy.ndarray 
        length 3 array of [latitude_deg, longitude_deg, height_m] rover station location
    sat_info : pandas.DataFrame
        Satellite positions with columns: Time, SatelliteID, SatPos (ECEF [x, y, z] in meters)
    epochs : numpy.ndarray
        Array of datetime objects
    """
    ts = load.timescale()
    
    # Convert start_time to skyfield time
    if isinstance(start_time, datetime):
        # Ensure timezone-aware datetime
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=utc)
        t_start = ts.from_datetime(start_time)
    else:
        t_start = ts.now()
    
    # Calculate end time
    duration_seconds = duration_hours * 3600
    num_epochs = int(duration_seconds / sample_interval)
    
    # Create base and rover locations
    base_lat, base_lon, base_height = base_loc
    rover_lat = base_lat + (baseline_length_km / 111.0)  # Approx 111 km per degree latitude
    rover_lon = base_lon
    rover_height = base_height
    rover_loc = np.array([rover_lat, rover_lon, rover_height])
    
    base = wgs84.latlon(base_lat, base_lon, elevation_m=base_height)
    rover = wgs84.latlon(rover_lat, rover_lon, elevation_m=rover_height)
    
    # Create topocentric positions
    base_topos = base
    rover_topos = rover
    
    # Load GPS satellites
    # stations_url = 'http://celestrak.com/NORAD/elements/gps-ops.txt'
    # satellites = load.tle_file(stations_url, reload=True)
    satellites = load.tle_file('gps-ops.txt', reload=True)
    
    # Initialize data storage
    base_data = []
    rover_data = []
    sat_pos_data = []  # For sat_info DataFrame
    epochs_list = []
    
    # Generate observations for each epoch
    for i in range(num_epochs):
        t_epoch = t_start + (i * sample_interval) / 86400.0  # Convert seconds to days
        epoch_dt = t_epoch.utc_datetime()
        epochs_list.append(epoch_dt)
        
        # Get visible satellites at this epoch
        visible_sats = []
        for sat in satellites:
            # Compute topocentric position from base station
            # (Topocentric = satellite position relative to a ground station)
            difference = sat - base_topos
            topocentric = difference.at(t_epoch)
            # Get satellite elevation, azimuth, and distance
            alt, az, distance = topocentric.altaz()
            
            # Only include satellites above mask angle
            if alt.degrees >= mask_angle_deg:
                visible_sats.append({
                    'satellite': sat,
                    'base_elevation': alt.degrees,
                    'base_azimuth': az.degrees,
                    'base_distance': distance.m
                })
        
        if len(visible_sats) < 4:
            # Need at least 4 satellites for positioning
            continue
        
        # Process each visible satellite
        for sat_info in visible_sats:
            sat = sat_info['satellite']
            # Try to get satellite ID from various attributes
            sat_id = None
            if hasattr(sat, 'model') and hasattr(sat.model, 'satnum'):
                sat_id = sat.model.satnum
            elif hasattr(sat, 'name'):
                # Extract PRN from name if available (e.g., "GPS BIIF-12 (PRN 12)")
                name_str = str(sat.name)
                import re
                match = re.search(r'PRN\s*(\d+)', name_str)
                if match:
                    sat_id = int(match.group(1))
            
            if sat_id is None:
                sat_id = hash(str(sat)) % 32 + 1  # Fallback PRN
            
            # Base station observations
            difference_base = sat - base_topos
            topocentric_base = difference_base.at(t_epoch)
            base_elevation, base_azimuth, base_distance = topocentric_base.altaz()
            base_pr = base_distance.m
            
            # Calculate Earth rotation correction
            # Topocentric distance accounts for Earth rotation during signal propagation,
            # while simple geometric ECEF distance does not. The correction is the difference.
            base_ecef = base_topos.itrs_xyz.m
            sat_ecef = sat.at(t_epoch).position.m
            base_pr_geometric = np.linalg.norm(base_ecef - sat_ecef)
            base_earth_rotation_correction = base_pr - base_pr_geometric

            # Rover station observations
            difference_rover = sat - rover_topos
            topocentric_rover = difference_rover.at(t_epoch)
            rover_elevation, rover_azimuth, rover_distance = topocentric_rover.altaz()
            rover_pr = rover_distance.m
            
            # Calculate Earth rotation correction for rover
            rover_ecef = rover_topos.itrs_xyz.m
            rover_pr_geometric = np.linalg.norm(rover_ecef - sat_ecef)
            rover_earth_rotation_correction = rover_pr - rover_pr_geometric

            # Also check rover elevation mask (satellite must be visible from both stations)
            if rover_elevation.degrees < mask_angle_deg:
                continue
            
            # Generate synthetic CN0 (signal strength) based on elevation
            base_cn0 = 35 + (base_elevation.degrees / 90.0) * 15  # 35-50 dB-Hz
            rover_cn0 = 35 + (rover_elevation.degrees / 90.0) * 15
            
            base_data.append({
                'Time': epoch_dt,
                'SatelliteID': sat_id,
                'Pseudorange': base_pr,
                'ElevationAngle': base_elevation.degrees,
                'AzimuthAngle': base_azimuth.degrees,
                'CN0': base_cn0,
                'EarthRotationCorrection': base_earth_rotation_correction
            })
            
            rover_data.append({
                'Time': epoch_dt,
                'SatelliteID': sat_id,
                'Pseudorange': rover_pr,
                'ElevationAngle': rover_elevation.degrees,
                'AzimuthAngle': rover_azimuth.degrees,
                'CN0': rover_cn0,
                'EarthRotationCorrection': rover_earth_rotation_correction
            })
            
            # Store satellite position (AFTER rover mask check)
            sat_pos_ecef = sat.at(t_epoch).position.m  # Returns [x, y, z] in meters
            sat_pos_data.append({
                'Time': epoch_dt,
                'SatelliteID': sat_id,
                'SatPos': sat_pos_ecef.tolist()  # Convert numpy array to list for DataFrame
            })
    
    # Convert to DataFrames
    obs_base = pd.DataFrame(base_data)
    obs_rover = pd.DataFrame(rover_data)
    sat_info = pd.DataFrame(sat_pos_data)
    
    epochs = np.array(epochs_list[:len(obs_base)])

    # Group observations by Time for easier processing
    obs_base = obs_base.groupby('Time', sort=False).agg({
        'SatelliteID': lambda x: x.tolist(),
        'Pseudorange': lambda x: x.tolist(),
        'ElevationAngle': lambda x: x.tolist(),
        'AzimuthAngle': lambda x: x.tolist(),
        'CN0': lambda x: x.tolist(),
        'EarthRotationCorrection': lambda x: x.tolist()
    }).reset_index()

    obs_rover = obs_rover.groupby('Time', sort=False).agg({
        'SatelliteID': lambda x: x.tolist(),
        'Pseudorange': lambda x: x.tolist(),
        'ElevationAngle': lambda x: x.tolist(),
        'AzimuthAngle': lambda x: x.tolist(),
        'CN0': lambda x: x.tolist(),
        'EarthRotationCorrection': lambda x: x.tolist()
    }).reset_index()


    
    return obs_base, obs_rover,rover_loc, sat_info, epochs
