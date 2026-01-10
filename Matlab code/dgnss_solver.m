function [posEst, residuals, H, W] = dgnss_solver(obsBase, obsRover, basePosTrue, weightMode)
% DGNSS_SOLVER Computes Double-Difference Least Squares solution
%
% Purpose: Implements Research Plan Section 3, Phase C - DGNSS Processing
%          Solves for rover position using double-difference (DD) least squares
%          Double differences remove clock biases and common errors
%
% DGNSS Principle:
%   - Single Difference (SD): (Rover - Base) removes receiver clock bias
%   - Double Difference (DD): SD_satellite - SD_reference removes satellite clock bias
%   - At zero baseline: Common errors cancel perfectly
%   - At long baseline: Spatial decorrelation causes residual errors
%
% Inputs:
%   obsBase     - Table of base observations (must contain Pseudorange, SatellitePosition, SatelliteID)
%   obsRover    - Table of rover observations
%   basePosTrue - Known base position [x, y, z] in ECEF (meters)
%   weightMode  - 'elevation' for Model A (1/sin^2(El)) or 'snr' for Model B
%
% Outputs:
%   posEst      - Estimated rover position [x, y, z] in ECEF (meters)
%   residuals   - Post-fit DD residuals (meters)
%   H           - Design matrix (Double Difference)
%   W           - Weight matrix (inverse of observation covariance)

if nargin < 4
    weightMode = 'elevation';
end

% Initialize outputs
posEst = [NaN, NaN, NaN];
residuals = [];
H = [];
W = [];

% Find common satellites (required for double differencing)
[commonSats, idxBase, idxRover] = intersect(obsBase.SatelliteID, obsRover.SatelliteID);

if length(commonSats) < 4
    return; % Need at least 4 satellites for 3D position solution
end

% Extract measurements for common satellites
baseMeas = obsBase(idxBase, :);
roverMeas = obsRover(idxRover, :);

% Get satellite positions (ECEF)
satPos = baseMeas.SatellitePosition;
if iscell(satPos)
    satPos = cell2mat(satPos);
end
if size(satPos, 2) ~= 3 && size(satPos, 1) == 3
    satPos = satPos';
end

% Calculate line-of-sight vectors and elevations
% Use base position as initial guess for rover (for elevation calculation)
approxRoverPos = basePosTrue;
uVec = satPos - approxRoverPos;  % Vector from rover to satellite
range = sqrt(sum(uVec.^2, 2));
uVec = uVec ./ range;  % Unit vectors

% Convert to LLA for elevation calculation
if exist('ecef2lla', 'file') == 2
    [lat, lon, h] = ecef2lla(approxRoverPos);
else
    % Manual ECEF to LLA conversion (WGS84)
    x = approxRoverPos(1); y = approxRoverPos(2); z = approxRoverPos(3);
    a = 6378137.0; f = 1/298.257223563; e2 = 2*f - f*f;
    lon = atan2(y, x);
    p = sqrt(x^2 + y^2);
    lat = atan2(z, p * (1 - e2));
    h = 0;
    for i = 1:10
        N = a / sqrt(1 - e2 * sin(lat)^2);
        h = p / cos(lat) - N;
        lat = atan2(z, p * (1 - e2 * N / (N + h)));
        if abs(h - (p / cos(lat) - a / sqrt(1 - e2 * sin(lat)^2))) < 1e-3
            break;
        end
    end
    lat = rad2deg(lat);
    lon = rad2deg(lon);
end

% Transform to ENU (East-North-Up) frame for elevation calculation
slat = sin(deg2rad(lat)); clat = cos(deg2rad(lat));
slon = sin(deg2rad(lon)); clon = cos(deg2rad(lon));
R_ECEF_to_ENU = [-slon, clon, 0;
                 -slat*clon, -slat*slon, clat;
                 clat*clon,  clat*slon, slat];
enu = (R_ECEF_to_ENU * uVec')';
el = asin(enu(:, 3));  % Elevation angle (radians)

% Select reference satellite (highest elevation)
[~, refIdx] = max(el);

% Form Double Differences
% Single Difference: SD = Pseudorange_Rover - Pseudorange_Base
sd = roverMeas.Pseudorange - baseMeas.Pseudorange;

% Double Difference: DD = SD_satellite - SD_reference
% This removes both receiver and satellite clock biases
otherIdx = setdiff(1:length(commonSats), refIdx);
ddObs = sd(otherIdx) - sd(refIdx);

% Form Design Matrix
% H_sd = -uVec (unit vector from receiver to satellite)
% H_dd = H_sd(other) - H_sd(ref)
H_sd = -uVec;
H = H_sd(otherIdx, :) - H_sd(refIdx, :);

% Form Weight Matrix
% Model A (elevation): w = 1/sin^2(El) - higher elevation = higher weight
% Model B (SNR): w = f(C/N0) - placeholder implementation
if strcmpi(weightMode, 'elevation')
    w_diag = 1 ./ (sin(el).^2);
else
    w_diag = ones(size(el));  % Uniform weights for SNR model (placeholder)
end

% Construct Double-Difference Covariance Matrix
% Variance of SD: Var(SD) = Var(Base) + Var(Rover) = 2/w (assuming equal quality)
var_sd = (1./w_diag) + (1./w_diag);

% Covariance of DD: Cov(DD_i, DD_j) = Cov(SD_i - SD_ref, SD_j - SD_ref)
% Since DD_i = SD_i - SD_ref:
%   - If i=j: Var(DD_i) = Var(SD_i) + Var(SD_ref)
%   - If i≠j: Cov(DD_i, DD_j) = Var(SD_ref)
nDD = length(otherIdx);
Cov_DD = zeros(nDD, nDD);
var_ref = var_sd(refIdx);

for i = 1:nDD
    idx_i = otherIdx(i);
    for j = 1:nDD
        idx_j = otherIdx(j);
        if i == j
            Cov_DD(i,j) = var_sd(idx_i) + var_ref;
        else
            Cov_DD(i,j) = var_ref;
        end
    end
end

% Weight matrix is inverse of covariance
W = pinv(Cov_DD);
W = (W + W') / 2;  % Ensure symmetry

% Solve Least Squares: dx = (H'*W*H)^(-1) * H'*W * ddObs
% dx is the baseline vector from base to rover
if size(ddObs, 2) > size(ddObs, 1)
    ddObs = ddObs';
end

HWH = H' * W * H;
dx = HWH \ (H' * W * ddObs);

if size(dx, 2) == 1
    dx = dx';
end

posEst = basePosTrue + dx;

% Compute residuals: v = ddObs - H*dx
residuals = ddObs - H * dx';
if size(residuals, 2) > size(residuals, 1)
    residuals = residuals';
end

end
