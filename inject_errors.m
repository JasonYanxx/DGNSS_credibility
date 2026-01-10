function [obsBaseOut, obsRoverOut] = inject_errors(obsBase, obsRover, baselineLengthKm, correlationLengthKm)
% INJECT_ERRORS Adds thermal noise and spatially correlated tropospheric errors
%
% Purpose: Implements Research Plan Section 3, Phase B - Error Injection Engine
%          Adds realistic errors to clean pseudorange measurements:
%          1. Thermal noise (independent white noise)
%          2. Spatially correlated tropospheric errors (key contribution)
%
% Error Model per Research Plan:
%   rho_obs = rho_clean + delta_tropo + delta_thermal
%
% Spatial Correlation Formula:
%   epsilon_r = epsilon_b * e^(-L/L_corr) + eta * sqrt(1 - e^(-2L/L_corr))
%   where:
%     - epsilon_b: Base station tropospheric error
%     - epsilon_r: Rover station tropospheric error
%     - L: Baseline length (km)
%     - L_corr: Correlation length (km)
%     - eta: Independent noise with same variance as epsilon_b
%
% Key Behavior:
%   - At L=0: rho=1, errors identical → perfect cancellation in DGNSS
%   - At L>>L_corr: rho→0, errors independent → DGNSS assumption fails
%
% Inputs:
%   obsBase           - Table of base observations (must contain Pseudorange, SatelliteID)
%   obsRover          - Table of rover observations
%   baselineLengthKm  - Distance between base and rover (L) in km
%   correlationLengthKm - Correlation length parameter (L_corr) in km
%
% Outputs:
%   obsBaseOut        - Base observations with injected errors
%   obsRoverOut       - Rover observations with injected errors

obsBaseOut = obsBase;
obsRoverOut = obsRover;

% Error Parameters per Research Plan Section 3, Phase B
sigma_thermal = 0.20; % 20 cm thermal noise standard deviation
sigma_tropo = 2.0;    % 2 m tropospheric error standard deviation

%% Step 1: Add Thermal Noise (Independent White Noise)
% Thermal noise is uncorrelated between base and rover
nBase = height(obsBase);
nRover = height(obsRover);

if nBase > 0
    obsBaseOut.Pseudorange = obsBaseOut.Pseudorange + randn(nBase, 1) * sigma_thermal;
end

if nRover > 0
    obsRoverOut.Pseudorange = obsRoverOut.Pseudorange + randn(nRover, 1) * sigma_thermal;
end

%% Step 2: Add Spatially Correlated Tropospheric Error
% This is the key contribution: errors are correlated based on baseline length
% Per Research Plan: epsilon_r = epsilon_b * e^(-L/L_corr) + eta * sqrt(1 - e^(-2L/L_corr))

if nBase > 0 && nRover > 0
    % Find common satellites (errors must be applied to same physical source)
    [commonSats, idxBase, idxRover] = intersect(obsBase.SatelliteID, obsRover.SatelliteID);

    if ~isempty(commonSats)
        % Generate base station tropospheric error for common satellites
        % Note: Using independent noise per epoch (could be replaced with Gauss-Markov)
        err_tropo_base = randn(length(commonSats), 1) * sigma_tropo;

        % Calculate spatial correlation coefficient
        % rho = e^(-L / L_corr)
        % At L=0: rho=1 (perfect correlation, errors identical)
        % At L=L_corr: rho=e^(-1)≈0.37 (moderate correlation)
        % At L>>L_corr: rho→0 (independent errors)
        rho = exp(-baselineLengthKm / correlationLengthKm);

        % Generate correlated rover error
        % Formula: epsilon_r = rho * epsilon_b + sqrt(1 - rho^2) * eta
        % This ensures:
        %   - Variance of epsilon_r = Variance of epsilon_b = sigma_tropo^2
        %   - Correlation between epsilon_r and epsilon_b = rho
        eta = randn(length(commonSats), 1) * sigma_tropo;
        err_tropo_rover = (rho * err_tropo_base) + (sqrt(1 - rho^2) * eta);

        % Apply errors to matched satellite indices
        obsBaseOut.Pseudorange(idxBase) = obsBaseOut.Pseudorange(idxBase) + err_tropo_base;
        obsRoverOut.Pseudorange(idxRover) = obsRoverOut.Pseudorange(idxRover) + err_tropo_rover;
    end

    % Handle satellites visible only at one station (independent errors)
    [~, idxBaseOnly] = setdiff(obsBase.SatelliteID, commonSats);
    if ~isempty(idxBaseOnly)
        obsBaseOut.Pseudorange(idxBaseOnly) = obsBaseOut.Pseudorange(idxBaseOnly) + ...
            randn(length(idxBaseOnly), 1) * sigma_tropo;
    end

    [~, idxRoverOnly] = setdiff(obsRover.SatelliteID, commonSats);
    if ~isempty(idxRoverOnly)
        obsRoverOut.Pseudorange(idxRoverOnly) = obsRoverOut.Pseudorange(idxRoverOnly) + ...
            randn(length(idxRoverOnly), 1) * sigma_tropo;
    end
end

end
