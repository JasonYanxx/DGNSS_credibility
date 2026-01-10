function [sc, gnssBase, gnssRover, base, rover] = generate_scenario(startTime, durationHours, sampleRate, baseLoc, baselineLengthKm)
% GENERATE_SCENARIO Creates satellite scenario and GNSS measurement generators
%
% Purpose: Sets up the simulation environment per Research Plan Section 3, Phase A
%          - Creates satellite scenario with GPS satellites
%          - Places base and rover stations
%          - Configures GNSS generators for clean pseudorange measurements
%
% Inputs:
%   startTime        - datetime object for simulation start (UTC)
%   durationHours    - Simulation duration in hours
%   sampleRate       - Sample rate in Hz (e.g., 1/30 for 30-second intervals)
%   baseLoc          - Base station location [lat, lon, alt] (degrees, degrees, meters)
%   baselineLengthKm - Distance from base to rover in km (0 for zero baseline)
%
% Outputs:
%   sc       - satelliteScenario object
%   gnssBase - gnssMeasurementGenerator for base station
%   gnssRover- gnssMeasurementGenerator for rover station
%   base     - groundStation object for base
%   rover    - groundStation object for rover

% Calculate stop time
stopTime = startTime + hours(durationHours);

% Initialize satellite scenario
% Sample time = 1/sampleRate (e.g., 30 seconds for sampleRate = 1/30 Hz)
sc = satelliteScenario(startTime, stopTime, 1/sampleRate,"AutoSimulate",false);

% Add GPS satellites from TLE file
% Per Research Plan: Use real GPS orbits from almanac file
if isfile("gpsAlmanac.txt")
    sats = satellite(sc, "gpsAlmanac.txt");
else
    % Fallback: Use default TLE file if almanac not available
    warning('gpsAlmanac.txt not found. Using default TLE file.');
    
    % Create default TLE file with 6 GPS satellites in different orbital planes
    tleData = [ ...
        "GPS BIIF-10 (PRN 08)", ...
        "1 40730U 15036A   23324.50000000  .00000000  00000-0  00000-0 0  9999", ...
        "2 40730  55.0000   0.0000 0010000   0.0000   0.0000  2.00560000    00", ...
        "GPS BIIF-11 (PRN 10)", ...
        "1 41019U 15062A   23324.50000000  .00000000  00000-0  00000-0 0  9999", ...
        "2 41019  55.0000  60.0000 0010000   0.0000   0.0000  2.00560000    00", ...
        "GPS BIIF-12 (PRN 32)", ...
        "1 41328U 16007A   23324.50000000  .00000000  00000-0  00000-0 0  9999", ...
        "2 41328  55.0000 120.0000 0010000   0.0000   0.0000  2.00560000    00", ...
        "GPS BIII-01 (PRN 04)", ...
        "1 43873U 18109A   23324.50000000  .00000000  00000-0  00000-0 0  9999", ...
        "2 43873  55.0000 180.0000 0010000   0.0000   0.0000  2.00560000    00", ...
        "GPS BIII-02 (PRN 18)", ...
        "1 44506U 19054A   23324.50000000  .00000000  00000-0  00000-0 0  9999", ...
        "2 44506  55.0000 240.0000 0010000   0.0000   0.0000  2.00560000    00", ...
        "GPS BIII-03 (PRN 23)", ...
        "1 45854U 20041A   23324.50000000  .00000000  00000-0  00000-0 0  9999", ...
        "2 45854  55.0000 300.0000 0010000   0.0000   0.0000  2.00560000    00" ...
        ];
    
    fid = fopen("gps_default.tle", "w");
    fprintf(fid, "%s\n", tleData);
    fclose(fid);
    
    sats = satellite(sc, "gps_default.tle");
end

% Create base station at specified location
base = groundStation(sc, baseLoc(1), baseLoc(2), "Name", "Base");

% Create rover station
% Per Research Plan: For zero baseline (L=0), rover is at same location as base
if baselineLengthKm == 0
    roverLoc = baseLoc; % Zero baseline: perfect error cancellation
else
    % Move rover North by baselineLengthKm
    % Approximation: 1 degree latitude ≈ 111.1329 km
    delta_lat = baselineLengthKm / 111.1329;
    roverLoc = [baseLoc(1) + delta_lat, baseLoc(2), baseLoc(3)];
end

rover = groundStation(sc, roverLoc(1), roverLoc(2), "Name", "Rover");

% Create GNSS measurement generators
% Per Research Plan Section 3, Phase A: Generate clean pseudoranges (zero noise)
% R2025a API: gnssMeasurementGenerator requires InitialTime, ReferenceLocation, SampleRate
% RangeAccuracy=0 gives clean measurements (no noise added by generator)
gnssBase = gnssMeasurementGenerator('InitialTime', startTime, ...
    'ReferenceLocation', baseLoc, ...
    'SampleRate', sampleRate, ...
    'RangeAccuracy', 0); 

gnssRover = gnssMeasurementGenerator('InitialTime', startTime, ...
    'ReferenceLocation', roverLoc, ...
    'SampleRate', sampleRate, ...
    'RangeAccuracy', 0);

end
