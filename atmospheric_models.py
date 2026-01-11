"""
Atmospheric delay models for GNSS.
Provides real tropospheric and ionospheric delay calculations.
"""

import numpy as np
from datetime import datetime, timezone


def saastamoinen_tropo_delay(elevation_deg, height_m, lat_deg, doy, humi=0.7):
    """
    Calculate tropospheric delay using Saastamoinen model (RTKLIB version).
    
    This is the full Saastamoinen model as implemented in RTKLIB, using standard
    atmosphere to compute pressure, temperature, and water vapor pressure.
    
    Parameters:
    -----------
    elevation_deg : float
        Satellite elevation angle in degrees
    height_m : float
        Receiver height above sea level in meters
    lat_deg : float
        Receiver latitude in degrees
    doy : int
        Day of year (1-365) - not used in RTKLIB version but kept for compatibility
    humi : float, optional
        Relative humidity (0.0-1.0). Default: 0.7 (70%)
    
    Returns:
    --------
    tropo_delay : float
        Tropospheric delay in meters
    
    Reference:
    ----------
    RTKLIB tropmodel() function
    Saastamoinen, J. (1972). "Atmospheric correction for the troposphere and 
    stratosphere in radio ranging of satellites."
    """
    # Constants
    temp0 = 15.0  # Temperature at sea level (Celsius)
    PI = np.pi
    
    # Check input validity
    if height_m < -100.0 or height_m > 1e4 or elevation_deg <= 0:
        return 0.0
    
    # Standard atmosphere model
    # Height (ensure non-negative)
    hgt = max(0.0, height_m)
    
    # Pressure (hPa) using standard atmosphere
    # pres = 1013.25 * (1.0 - 2.2557E-5 * hgt)^5.2568
    pres = 1013.25 * np.power(1.0 - 2.2557e-5 * hgt, 5.2568)
    
    # Temperature (Kelvin)
    # temp = temp0 - 6.5E-3 * hgt + 273.16
    temp = temp0 - 6.5e-3 * hgt + 273.16
    
    # Water vapor pressure (hPa)
    # e = 6.108 * humi * exp((17.15 * temp - 4684.0) / (temp - 38.45))
    e = 6.108 * humi * np.exp((17.15 * temp - 4684.0) / (temp - 38.45))
    
    # Saastamoinen model
    # Convert latitude to radians
    lat_rad = np.deg2rad(lat_deg)
    
    # Zenith angle (radians)
    el_rad = np.deg2rad(elevation_deg)
    z = PI / 2.0 - el_rad
    
    # Dry component (hydrostatic)
    # trph = 0.0022768 * pres / (1.0 - 0.00266 * cos(2.0 * lat) - 0.00028 * hgt/1E3) / cos(z)
    trph = 0.0022768 * pres / (1.0 - 0.00266 * np.cos(2.0 * lat_rad) - 0.00028 * hgt / 1e3) / np.cos(z)
    
    # Wet component (non-hydrostatic)
    # trpw = 0.002277 * (1255.0/temp + 0.05) * e / cos(z)
    trpw = 0.002277 * (1255.0 / temp + 0.05) * e / np.cos(z)
    
    # Total tropospheric delay
    tropo_delay = trph + trpw
    
    return tropo_delay


def klobuchar_iono_delay(elevation_deg, azimuth_deg, lat_deg, lon_deg, 
                         gps_time, alpha=None, beta=None):
    """
    Calculate ionospheric delay using Klobuchar model (RTKLIB version).
    
    This is the exact implementation of RTKLIB's ionmodel() function.
    
    Parameters:
    -----------
    elevation_deg : float
        Satellite elevation angle in degrees
    azimuth_deg : float
        Satellite azimuth angle in degrees
    lat_deg : float
        Receiver latitude in degrees
    lon_deg : float
        Receiver longitude in degrees
    gps_time : float or datetime
        GPS time (seconds of week) or datetime object
    alpha : array-like, optional
        Klobuchar alpha parameters (4 values: a0, a1, a2, a3). 
        If None, uses RTKLIB default values.
    beta : array-like, optional
        Klobuchar beta parameters (4 values: b0, b1, b2, b3).
        If None, uses RTKLIB default values.
    
    Returns:
    --------
    iono_delay : float
        Ionospheric delay in meters (L1 frequency)
    
    Reference:
    ----------
    RTKLIB ionmodel() function
    Klobuchar, J. A. (1987). "Ionospheric time-delay algorithm for single-frequency GPS users."
    """
    from datetime import datetime
    from skyfield.api import utc
    
    # RTKLIB default ionospheric parameters (2004/1/1)
    ion_default = np.array([
        0.1118e-07, -0.7451e-08, -0.5961e-07, 0.1192e-06,  # alpha[0-3] (a0, a1, a2, a3)
        0.1167e+06, -0.2294e+06, -0.1311e+06, 0.1049e+07  # beta[0-3] (b0, b1, b2, b3)
    ])
    
    # Combine alpha and beta into single ion array (RTKLIB format: {a0,a1,a2,a3,b0,b1,b2,b3})
    if alpha is None and beta is None:
        ion = ion_default.copy()
    else:
        # Use provided values or defaults
        if alpha is None:
            alpha = ion_default[0:4]
        else:
            alpha = np.array(alpha)
        if beta is None:
            beta = ion_default[4:8]
        else:
            beta = np.array(beta)
        ion = np.concatenate([alpha, beta])
    
    # Check if ion parameters are valid (non-zero norm)
    # RTKLIB: if (norm(ion,8)<=0.0) ion=ion_default;
    if np.linalg.norm(ion) <= 0.0:
        ion = ion_default.copy()
    
    # Convert angles to radians
    PI = np.pi
    el_rad = np.deg2rad(elevation_deg)
    az_rad = np.deg2rad(azimuth_deg)
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    
    # Input validation
    # RTKLIB: if (pos[2]<-1E3||azel[1]<=0) return 0.0;
    # Note: height check would need height_m parameter, but RTKLIB checks pos[2]
    # We'll skip height check as height is not used in the model
    if elevation_deg <= 0:
        return 0.0
    
    # Earth centered angle (semi-circle)
    # RTKLIB: psi = 0.0137 / (azel[1]/PI + 0.11) - 0.022;
    psi = 0.0137 / (el_rad / PI + 0.11) - 0.022
    
    # Subionospheric latitude/longitude (semi-circle)
    # RTKLIB: phi = pos[0]/PI + psi*cos(azel[0]);
    phi = lat_rad / PI + psi * np.cos(az_rad)
    
    # Clamp latitude
    # RTKLIB: if (phi> 0.416) phi= 0.416; else if (phi<-0.416) phi=-0.416;
    if phi > 0.416:
        phi = 0.416
    elif phi < -0.416:
        phi = -0.416
    
    # RTKLIB: lam = pos[1]/PI + psi*sin(azel[0])/cos(phi*PI);
    lam = lon_rad / PI + psi * np.sin(az_rad) / np.cos(phi * PI)
    
    # Geomagnetic latitude (semi-circle)
    # RTKLIB: phi += 0.064*cos((lam-1.617)*PI);
    phi += 0.064 * np.cos((lam - 1.617) * PI)
    
    # Local time (seconds)
    # RTKLIB: tt = 43200.0*lam + time2gpst(t, &week);
    # Convert gps_time to seconds of week
    if isinstance(gps_time, datetime):
        # Convert datetime to GPS seconds of week
        # GPS epoch: 1980-01-06 00:00:00 UTC
        gps_epoch = datetime(1980, 1, 6, 0, 0, 0, tzinfo=utc)
        if gps_time.tzinfo is None:
            # Assume UTC if timezone-naive
            gps_time = gps_time.replace(tzinfo=utc)
        delta = gps_time - gps_epoch
        gps_seconds = delta.total_seconds()
        # Convert to seconds of week (GPS weeks started at GPS epoch)
        gps_sow = gps_seconds % 604800.0  # 604800 = 7 days in seconds
    else:
        # Assume gps_time is already seconds of week
        gps_sow = float(gps_time) % 604800.0
    
    # RTKLIB: tt = 43200.0*lam + time2gpst(t, &week);
    tt = 43200.0 * lam + gps_sow
    
    # Normalize to 0 <= tt < 86400
    # RTKLIB: tt -= floor(tt/86400.0)*86400.0; /* 0<=tt<86400 */
    tt = tt - np.floor(tt / 86400.0) * 86400.0
    
    # Slant factor
    # RTKLIB: f = 1.0 + 16.0*pow(0.53-azel[1]/PI, 3.0);
    f = 1.0 + 16.0 * np.power(0.53 - el_rad / PI, 3.0)
    
    # Ionospheric delay
    # RTKLIB uses Horner's method:
    # amp = ion[0] + phi*(ion[1] + phi*(ion[2] + phi*ion[3]));
    # per = ion[4] + phi*(ion[5] + phi*(ion[6] + phi*ion[7]));
    amp = ion[0] + phi * (ion[1] + phi * (ion[2] + phi * ion[3]))
    per = ion[4] + phi * (ion[5] + phi * (ion[6] + phi * ion[7]))
    
    # Clamp values
    # RTKLIB: amp = amp<0.0?0.0:amp; per = per<72000.0?72000.0:per;
    if amp < 0.0:
        amp = 0.0
    if per < 72000.0:
        per = 72000.0
    
    # Phase
    # RTKLIB: x = 2.0*PI*(tt-50400.0)/per;
    x = 2.0 * PI * (tt - 50400.0) / per
    
    # Ionospheric delay calculation
    # RTKLIB: return CLIGHT*f*(fabs(x)<1.57?5E-9+amp*(1.0+x*x*(-0.5+x*x/24.0)):5E-9);
    CLIGHT = 299792458.0  # Speed of light in m/s
    
    if np.abs(x) < 1.57:
        # RTKLIB formula: 5E-9 + amp*(1.0 + x*x*(-0.5 + x*x/24.0))
        # This expands to: 5E-9 + amp*(1.0 - 0.5*x^2 + x^4/24.0)
        iono_delay = CLIGHT * f * (5e-9 + amp * (1.0 + x * x * (-0.5 + x * x / 24.0)))
    else:
        iono_delay = CLIGHT * f * 5e-9
    
    return iono_delay


def calculate_atmospheric_delays(elevation_deg, azimuth_deg, lat_deg, lon_deg, 
                                 height_m, gps_time, doy=None):
    """
    Calculate both tropospheric and ionospheric delays.
    
    Parameters:
    -----------
    elevation_deg : float
        Satellite elevation angle in degrees
    azimuth_deg : float
        Satellite azimuth angle in degrees
    lat_deg : float
        Receiver latitude in degrees
    lon_deg : float
        Receiver longitude in degrees
    height_m : float
        Receiver height above sea level in meters
    gps_time : float or datetime
        GPS time or datetime object
    doy : int, optional
        Day of year (1-365). If None, calculated from gps_time.
    
    Returns:
    --------
    tropo_delay : float
        Tropospheric delay in meters
    iono_delay : float
        Ionospheric delay in meters
    """
    # Calculate day of year if not provided
    if doy is None:
        if isinstance(gps_time, datetime):
            doy = gps_time.timetuple().tm_yday
        else:
            doy = 180  # Default to mid-year
    
    # Calculate delays
    tropo_delay = saastamoinen_tropo_delay(elevation_deg, height_m, lat_deg, doy)
    iono_delay = klobuchar_iono_delay(elevation_deg, azimuth_deg, lat_deg, lon_deg, gps_time)
    
    return tropo_delay, iono_delay


def multipath_error(elevation_deg, multipath_type='code'):
    """
    Calculate multipath error based on elevation angle.
    
    Multipath errors are stronger at low elevations due to signal reflections
    from ground surfaces. This model is based on standard GNSS multipath models.
    
    Parameters:
    -----------
    elevation_deg : float
        Satellite elevation angle in degrees
    multipath_type : str
        'code' for code multipath (pseudorange), 'carrier' for carrier phase multipath
    
    Returns:
    --------
    multipath_error : float
        Multipath error in meters (RMS)
    """
    el_rad = np.deg2rad(elevation_deg)
    
    if multipath_type == 'code':
        # Code multipath model (pseudorange)
        # Standard model: stronger at low elevations
        # Typical range: 0.1-1.5 m depending on environment
        # Model: σ_mp = a * exp(-b * sin(el)) + c
        
        # Urban/suburban environment (moderate multipath)
        a = 0.5  # meters
        b = 2.0  # decay constant
        c = 0.1  # floor value (meters)
        
        multipath_rms = a * np.exp(-b * np.sin(el_rad)) + c
        
    else:  # carrier phase multipath
        # Carrier phase multipath (much smaller)
        # Typical range: 0.01-0.05 m
        a = 0.03
        b = 2.5
        c = 0.01
        multipath_rms = a * np.exp(-b * np.sin(el_rad)) + c
    
    return multipath_rms


def thermal_noise_error(cn0_db_hz, code_type='C/A'):
    """
    Calculate thermal noise (code noise) error based on signal strength.
    
    Thermal noise is inversely proportional to signal strength (C/N0).
    
    Theoretical Foundation:
    -----------------------
    For code measurements, the theoretical precision is given by:
        σ_code = chip_width / (2 * sqrt(C/N0 * T))
    
    where:
        - chip_width: Code chip width in meters (C/A: ~293 m, P: ~29.3 m)
        - C/N0: Carrier-to-noise density ratio (Hz)
        - T: Integration time (seconds)
    
    This formula comes from the Cramér-Rao Lower Bound (CRLB) for code phase
    estimation in the presence of white Gaussian noise.
    
    For C/A code:
    - Uses an empirical model because actual receiver performance deviates
      from theoretical due to:
      * Early-late correlator spacing
      * DLL (Delay Lock Loop) bandwidth
      * Quantization effects
      * Non-ideal correlation functions
    - Empirical formula: σ = a * exp(-b * (CN0 - c)) + d
      Matches typical GPS receiver code noise characteristics
    
    For P-code:
    - Uses theoretical formula directly because:
      * Higher chipping rate (10.23 MHz vs 1.023 MHz) → better precision
      * Less affected by receiver implementation details
      * Closer to theoretical performance
    
    References:
    -----------
    1. Van Dierendonck, A. J., et al. (1992). "Theory and Performance of Narrow 
       Correlator Spacing in a GPS Receiver." Navigation, 39(3), 265-283.
    
    2. Kaplan, E. D., & Hegarty, C. J. (2017). "Understanding GPS/GNSS: 
       Principles and Applications" (3rd ed.). Artech House.
       - Chapter 7: GPS Signal Structure
       - Chapter 8: GPS Receiver Design
    
    3. Misra, P., & Enge, P. (2011). "Global Positioning System: Signals, 
       Measurements, and Performance" (2nd ed.). Ganga-Jamuna Press.
       - Section 5.3: Code Tracking Loop
    
    4. Betz, J. W. (2000). "Design and Performance of Code Tracking for the 
       GPS M Code Signal." Proceedings of ION GPS 2000.
    
    Parameters:
    -----------
    cn0_db_hz : float
        Carrier-to-noise density ratio in dB-Hz
    code_type : str
        Code type: 'C/A' for GPS L1 C/A code, 'P' for P-code
    
    Returns:
    --------
    thermal_noise : float
        Thermal noise error in meters (RMS)
    """
    # Convert dB-Hz to linear
    cn0_linear = 10.0 ** (cn0_db_hz / 10.0)  # Hz
    
    # Integration time (typical): 20 ms
    T = 0.02  # seconds
    
    if code_type == 'C/A':
        # GPS L1 C/A code thermal noise model
        # Empirical model based on standard GPS receiver performance
        # Typical values: 0.1-0.3 m for CN0 40-50 dB-Hz, 0.3-0.5 m for CN0 35 dB-Hz
        
        # Empirical formula: σ = a * exp(-b * (CN0 - c)) + d
        # Matches typical GPS receiver code noise characteristics
        # This deviates from pure theory due to receiver implementation effects
        a = 0.4  # meters (amplitude)
        b = 0.15  # decay rate per dB
        c = 35.0  # reference CN0 (dB-Hz)
        d = 0.1  # floor value (meters)
        
        thermal_noise_rms = a * np.exp(-b * (cn0_db_hz - c)) + d
        
        # Ensure reasonable range
        thermal_noise_rms = max(0.05, min(thermal_noise_rms, 0.5))
        
    else:  # P-code
        # P-code theoretical model
        # Based on CRLB: σ = chip_width / (2 * sqrt(C/N0 * T))
        # P-code chip width: c / (10.23 MHz) ≈ 29.3 meters
        # where c = 299792458 m/s (speed of light)
        chip_width = 29.3  # meters (P-code chip width)
        thermal_noise_rms = chip_width / (2.0 * np.sqrt(cn0_linear * T))
        thermal_noise_rms = np.clip(thermal_noise_rms, 0.005, 0.1)
    
    return thermal_noise_rms

import numpy as np

def thermal_noise_error_corrected(cn0_db_hz, code_type='C/A', bandwidth_hz=2.0, correlator_spacing=0.5):
    """
    Calculate DLL thermal noise error (in meters) using the Kaplan/Betz closed-loop model.
    
    Theoretical Basis:
    ------------------
    Based on Kaplan & Hegarty (3rd Ed), Eq 8.16.
    sigma_code = lambda_chip * sqrt( (B_L * d) / (2 * cn0_linear) * (1 + 1/(T*cn0_linear)) )
    
    Parameters:
    -----------
    cn0_db_hz : float
        C/N0 in dB-Hz
    code_type : str
        'C/A' (293m chip) or 'P' (29.3m chip)
    bandwidth_hz : float
        DLL Loop Bandwidth (B_L). Standard is 0.5 - 2.0 Hz.
    correlator_spacing : float
        Early-Late spacing in chips (d). 
        Standard = 0.5 to 1.0. Narrow Correlator = 0.1.
        
    Returns:
    --------
    sigma : float (meters)
    """
    # 1. Constants
    c = 299792458.0
    
    if code_type == 'C/A':
        chip_rate = 1.023e6
        # C/A code usually uses 20ms integration (data bit duration)
        T = 0.02 
    elif code_type == 'P':
        chip_rate = 10.23e6
        # P-code often uses shorter integration or assumes semi-codeless handling
        # But for theoretical P-code tracking, we can assume 20ms or 10ms
        T = 0.02
    else:
        raise ValueError("Unknown code type")

    lambda_chip = c / chip_rate  # ~293m for C/A, ~29.3m for P
    
    # 2. Conversion
    cn0_linear = 10.0**(cn0_db_hz / 10.0) # Convert dB-Hz to linear ratio (Hz)
    
    # 3. Validation
    # At extremely low C/N0, the loop unlocks. We return a high ceiling.
    if cn0_db_hz < 20:
        return 100.0
        
    # 4. The Standard Closed-Loop Formula (Betz/Kaplan)
    # Factor 1: Geometry of the chip
    # Factor 2: DLL filtering (B_L) and Correlator slope (d)
    # Factor 3: Squaring loss (1 + 1/SNR_in)
    
    term1 = (bandwidth_hz * correlator_spacing) / (2 * cn0_linear)
    term2 = 1 + (1 / (T * cn0_linear))
    
    sigma_in_chips = np.sqrt(term1 * term2)
    
    # Convert chips to meters
    sigma_meters = lambda_chip * sigma_in_chips
    
    return sigma_meters


def generate_multipath_error(elevation_deg, multipath_type='code'):
    """
    Generate a random multipath error sample.
    
    Parameters:
    -----------
    elevation_deg : float
        Satellite elevation angle in degrees
    multipath_type : str
        'code' or 'carrier'
    
    Returns:
    --------
    multipath_error : float
        Random multipath error sample in meters
    """
    rms = multipath_error(elevation_deg, multipath_type)
    # Multipath has both random and systematic components
    # Use Rayleigh distribution for magnitude (positive bias)
    error = np.random.rayleigh(rms / np.sqrt(2))
    # Random sign
    error = error * np.sign(np.random.randn())
    return error


def generate_thermal_noise_error(cn0_db_hz, code_type='C/A'):
    """
    Generate a random thermal noise error sample.
    
    Parameters:
    -----------
    cn0_db_hz : float
        Carrier-to-noise density ratio in dB-Hz
    code_type : str
        'C/A' or 'P'
    
    Returns:
    --------
    thermal_noise_error : float
        Random thermal noise error sample in meters
    """
    rms = thermal_noise_error(cn0_db_hz, code_type)
    rms_corrected = thermal_noise_error_corrected(cn0_db_hz, code_type)
    # Thermal noise is Gaussian (white noise)
    error = np.random.randn() * rms
    return error

