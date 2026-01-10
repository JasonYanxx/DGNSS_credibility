function [obsBaseOut, obsRoverOut, covarianceScale] = inject_errors(obsBase, obsRover, scenarioType, varargin)
% INJECT_ERRORS Adds scenario-specific errors for Commissioning Protocol validation
%
% Purpose: Implements Design Doc Section 3.1 - Error Injection Engine
%          Generates three specific scenarios to validate the 15-minute Block Averaging 
%          Commissioning Protocol:
%          1. Multipath: Time-correlated errors (Gauss-Markov colored noise)
%          2. SMM: Constant bias + white noise
%          3. Optimism: White noise with underestimated reported covariance
%
% Design Doc Reference: "Refactoring DGNSS Simulation for Commissioning Protocol 
%                        Verification (Block Averaging & Multipath Diagnosis)"
%
% Inputs:
%   obsBase       - Table of base observations (must contain Pseudorange, SatelliteID)
%   obsRover      - Table of rover observations
%   scenarioType  - String: 'Multipath', 'SMM', or 'Optimism'
%   varargin      - Optional parameters:
%                   'dt' (time step in seconds, default = 1/60 for 60Hz)
%                   'prevState' (previous Gauss-Markov state for Multipath)
%
% Outputs:
%   obsBaseOut        - Base observations with injected errors
%   obsRoverOut       - Rover observations with injected errors
%   covarianceScale   - Scaling factor for reported covariance (1.0 except for Optimism)
%
% Scenario Details per Design Doc:
%   'Multipath': Gauss-Markov process with time constant τ = 300s (5 min)
%                Formula: x[k+1] = exp(-dt/τ) * x[k] + σ * sqrt(1 - exp(-2*dt/τ)) * w[k]
%                Target: ELT Pass after block averaging (colored noise averages down)
%
%   'SMM':       Constant bias (0.5m) + white noise (0.2m)
%                Target: ELT Fail after block averaging (bias persists)
%
%   'Optimism':  White noise (σ = 0.2m), but report covariance as σ² = 0.1² (scaled by 0.25)
%                Target: ELT Pass, but NLL/NCI shows Optimism (underestimated uncertainty)

% Parse optional inputs
p = inputParser;
addParameter(p, 'dt', 1/60, @isnumeric);  % Default: 60Hz sampling
addParameter(p, 'prevState', [], @(x) isnumeric(x) || isempty(x));
parse(p, varargin{:});

dt = p.Results.dt;
prevState = p.Results.prevState;

% Initialize outputs
obsBaseOut = obsBase;
obsRoverOut = obsRover;
covarianceScale = 1.0;  % Default: no scaling (modified for Optimism scenario)

% Validate scenario type
validScenarios = {'Multipath', 'SMM', 'Optimism'};
if ~ismember(scenarioType, validScenarios)
    error('Invalid scenarioType. Must be one of: %s', strjoin(validScenarios, ', '));
end

%% Error Parameters per Design Doc
sigma_thermal = 0.20;  % 20 cm thermal noise (white noise baseline)
sigma_mp = 2.0;        % 2 m multipath error standard deviation
tau_mp = 300;          % 5 minutes time constant for Gauss-Markov (multipath correlation)
bias_smm = 0.5;        % 50 cm constant bias for SMM scenario
optimism_factor = 0.25; % Report covariance scaled by 0.25 (variance ratio, i.e., σ_reported = 0.5 * σ_actual)

nBase = height(obsBase);
nRover = height(obsRover);

%% Scenario-Specific Error Injection

switch scenarioType
    case 'Multipath'
        %% Multipath Scenario: Gauss-Markov Time-Correlated Colored Noise
        % Per Design Doc Section 3.1:
        % - Inject Gauss-Markov process with time constant τ = 300s (5 min)
        % - Formula: x[k+1] = φ * x[k] + σ * sqrt(1 - φ²) * w[k]
        %   where φ = exp(-dt/τ)
        % - Target: ELT Pass after block averaging (colored noise averages down)
        
        phi = exp(-dt / tau_mp);  % Gauss-Markov coefficient
        noise_std = sigma_mp * sqrt(1 - phi^2);  % Process noise standard deviation
        
        % Generate Gauss-Markov errors for base station
        if nBase > 0
            if isempty(prevState) || ~isfield(prevState, 'base')
                % Initialize state with zero or steady-state distribution
                err_base = randn(nBase, 1) * sigma_mp;  % Steady-state initialization
            else
                % Continue from previous state
                err_base = phi * prevState.base + randn(nBase, 1) * noise_std;
            end
            obsBaseOut.Pseudorange = obsBaseOut.Pseudorange + err_base;
        end
        
        % Generate correlated Gauss-Markov errors for rover station
        % For simplicity, assume same temporal correlation but independent realizations
        if nRover > 0
            if isempty(prevState) || ~isfield(prevState, 'rover')
                err_rover = randn(nRover, 1) * sigma_mp;
            else
                err_rover = phi * prevState.rover + randn(nRover, 1) * noise_std;
            end
            obsRoverOut.Pseudorange = obsRoverOut.Pseudorange + err_rover;
        end
        
        % Store state for next epoch (caller should pass this back as prevState)
        % Note: This is a simplified implementation; full implementation would
        % track state per satellite
        
    case 'SMM'
        %% SMM Scenario: Constant Bias + White Noise
        % Per Design Doc Section 3.1:
        % - Inject constant bias (0.5m) + white noise (0.2m)
        % - Target: ELT Fail after block averaging (bias persists)
        
        % Add constant bias + white noise to base station
        if nBase > 0
            err_base = bias_smm + randn(nBase, 1) * sigma_thermal;
            obsBaseOut.Pseudorange = obsBaseOut.Pseudorange + err_base;
        end
        
        % Add constant bias + white noise to rover station
        if nRover > 0
            err_rover = bias_smm + randn(nRover, 1) * sigma_thermal;
            obsRoverOut.Pseudorange = obsRoverOut.Pseudorange + err_rover;
        end
        
    case 'Optimism'
        %% Optimism Scenario: White Noise with Underestimated Reported Covariance
        % Per Design Doc Section 3.1:
        % - Inject white noise (σ = 0.2m)
        % - Report covariance scaled by factor 0.25 (i.e., σ_reported² = 0.25 * σ_actual²)
        % - Target: ELT Pass, but NLL/NCI shows Optimism
        
        % Add white noise to base station
        if nBase > 0
            err_base = randn(nBase, 1) * sigma_thermal;
            obsBaseOut.Pseudorange = obsBaseOut.Pseudorange + err_base;
        end
        
        % Add white noise to rover station
        if nRover > 0
            err_rover = randn(nRover, 1) * sigma_thermal;
            obsRoverOut.Pseudorange = obsRoverOut.Pseudorange + err_rover;
        end
        
        % Set covariance scale factor (solver should multiply reported covariance by this)
        covarianceScale = optimism_factor;
end

end
