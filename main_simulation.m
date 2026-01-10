% main_simulation.m
% DGNSS Credibility Diagnosis Simulation
% 
% Purpose: Implements the Research Plan for diagnosing credibility collapse in DGNSS
%          as baseline length increases. Demonstrates how spatial decorrelation of
%          atmospheric errors causes model misspecification that standard tests miss.
%
% Research Plan Reference: "Credibility Diagnosis of Stochastic Models in DGNSS:
%                          Unmasking the Impact of Spatial Decorrelation via a 
%                          Multi-Metric Framework"
%
% MATLAB Version: R2025a
% Required Toolboxes: Navigation Toolbox, Satellite Communications Toolbox
%
% Output: 
%   - dgnss_simulation_results.mat: Structure array with metrics for each scenario
%   - nci_vs_baseline.png: Plot showing NCI vs baseline length
%   - all_metrics_vs_baseline.png: Comprehensive metrics comparison
%
% Expected Results (per Research Plan):
%   - L=0km: NCI≈1 (Credible - errors cancel perfectly)
%   - L=50km: NCI>>1 (Pessimism + SMM - errors decorrelate, model fails)

clear; clc; close all;

%% ========================================================================
% CONFIGURATION: Experimental Grid per Research Plan Section 4
% ========================================================================

% Baseline Lengths: Test from zero baseline to long baseline
% Per Research Plan: {0km, 1km, 10km, 30km, 50km}
baselineLengths = [0, 1, 10, 30, 50]; % km

% Stochastic Models: Model A (elevation weighting) and Model B (SNR-based)
% Per Research Plan Section 4
stochasticModels = {'elevation', 'snr'};

% Atmosphere Conditions: Quiet (L_corr=50km) and Active (L_corr=10km)
% Per Research Plan Section 4
atmosphereConditions = struct('name', {'Quiet', 'Active'}, ...
    'L_corr', {50, 10});

% Simulation Parameters
startTime = datetime('now', 'TimeZone', 'UTC');
durationHours = 24; % 24 hours per Research Plan (to capture full geometry changes)
sampleRate = 1/30; % 30 second interval (Hz) per Research Plan
decimationInterval = 5 * 60; % 5 minutes in seconds (for statistical independence)
baseLoc = [42.3601, -71.0589, 0]; % Base station location [lat, lon, alt] - Boston

% Results Storage
results = [];

%% ========================================================================
% MAIN SIMULATION LOOP: Grid Search over Experimental Parameters
% ========================================================================

fprintf('=== DGNSS Credibility Diagnosis Simulation ===\n');
fprintf('Per Research Plan: Testing credibility collapse with baseline length\n\n');

% Loop over all combinations: Baseline Length × Stochastic Model × Atmosphere
for iL = 1:length(baselineLengths)
    L = baselineLengths(iL);
    if L == 0
        fprintf('Baseline: 0 km (Zero Baseline)\n');
    else
        fprintf('Baseline: %.1f km\n', L);
    end

    for iStoch = 1:length(stochasticModels)
        weightMode = stochasticModels{iStoch};
        fprintf('  Stochastic Model: %s\n', weightMode);
        
        for iAtm = 1:length(atmosphereConditions)
            atm = atmosphereConditions(iAtm);
            L_corr = atm.L_corr;
            fprintf('    Atmosphere: %s (L_corr=%d km)\n', atm.name, L_corr);

            %% Phase A: Scenario Generation (Research Plan Section 3, Phase A)
            % Create satellite scenario and GNSS measurement generators
            [sc, gnssBase, gnssRover, baseObj, roverObj] = ...
                generate_scenario(startTime, durationHours, sampleRate, baseLoc, L);
            
            sats = sc.Satellites;
            if isempty(sats)
                warning('No satellites available. Skipping scenario.');
                continue;
            end

            %% Phase B: Error Injection and Processing (Research Plan Section 3, Phase B & C)
            metricHistory = [];
            epochCount = 0;
            
            % Calculate decimation: Process 1 epoch every 5 minutes
            % This ensures statistical independence per Research Plan Section 3, Phase C
            decimationEpochs = max(1, round(decimationInterval * sampleRate));
            expectedEpochs = round(durationHours * 3600 * sampleRate);
            
            fprintf('      Processing epochs (decimation: every %d epochs = %.1f min)\n', ...
                decimationEpochs, decimationInterval/60);

            % Process epochs: Advance scenario and compute metrics
            while epochCount < expectedEpochs * 2 && advance(sc)
                epochCount = epochCount + 1;
                
                % Decimation: Only process epochs at decimation intervals
                if mod(epochCount - 1, decimationEpochs) ~= 0
                    continue;
                end

                % Get satellite and receiver states (ECEF coordinates)
                satStates = states(sats, "CoordinateFrame", "ecef"); % satStates: Nx1x6
                satStates = squeeze(satStates); % Now Nx6
                satPos = satStates(:,1:3);  % Nx3 matrix of satellite positions
                satVel = satStates(:,4:6);  % Nx3 matrix of satellite velocities
                
                baseState = states(baseObj, "CoordinateFrame", "ecef");
                basePos = baseState(1:3)';  % Base position [x, y, z] ECEF
                baseVel = baseState(4:6)';  % Base velocity [vx, vy, vz] ECEF
                
                roverState = states(roverObj, "CoordinateFrame", "ecef");
                roverPos = roverState(1:3)';  % Rover position [x, y, z] ECEF
                roverVel = roverState(4:6)';  % Rover velocity [vx, vy, vz] ECEF

                % Generate clean pseudorange measurements (zero noise per Research Plan)
                obsBase = step(gnssBase, basePos, baseVel, satPos, satVel);
                obsRover = step(gnssRover, roverPos, roverVel, satPos, satVel);

                % Validate measurements: Need at least 4 satellites for DGNSS solution
                if height(obsBase) < 4 || height(obsRover) < 4 || ...
                   ~ismember('Pseudorange', obsBase.Properties.VariableNames)
                    continue;
                end

                % Ensure SatelliteID and SatellitePosition columns exist
                if ~ismember('SatelliteID', obsBase.Properties.VariableNames)
                    obsBase.SatelliteID = (1:height(obsBase))';
                end
                if ~ismember('SatelliteID', obsRover.Properties.VariableNames)
                    obsRover.SatelliteID = (1:height(obsRover))';
                end
                if ~ismember('SatellitePosition', obsBase.Properties.VariableNames)
                    obsBase.SatellitePosition = satPos(1:height(obsBase), :);
                end
                if ~ismember('SatellitePosition', obsRover.Properties.VariableNames)
                    obsRover.SatellitePosition = satPos(1:height(obsRover), :);
                end

                % Inject errors: Add thermal noise and spatially correlated tropospheric errors
                % Per Research Plan Section 3, Phase B
                % Formula: epsilon_r = epsilon_b * e^(-L/L_corr) + eta * sqrt(1 - e^(-2L/L_corr))
                % At L=0: errors identical (perfect cancellation)
                % At L=50km: errors decorrelate (DGNSS assumption fails)
                [obsBaseErr, obsRoverErr] = inject_errors(obsBase, obsRover, L, L_corr);

                % DGNSS Processing: Double-Difference Least Squares solver
                % Per Research Plan Section 3, Phase C
                [posEst, residuals, H, W] = dgnss_solver(obsBaseErr, obsRoverErr, basePos, weightMode);

                % Calculate Credibility Metrics: ELT, NCI, NLL, ES
                % Per Research Plan Section 3, Phase C
                if ~isempty(residuals)
                    metrics = calculate_metrics(residuals, H, W, 3);
                    metrics.Epoch = epochCount;
                    metrics.Baseline = L;
                    metrics.StochasticModel = weightMode;
                    metrics.Atmosphere = atm.name;
                    
                    % Append to history
                    if isempty(metricHistory)
                        metricHistory = metrics;
                    else
                        metricHistory(end+1) = metrics;
                    end
                end
            end

            %% Aggregate Results for This Scenario
            if ~isempty(metricHistory)
                % Compute mean metrics (excluding NaN values)
                meanELT = mean([metricHistory.ELT], 'omitnan');
                meanNCI = mean([metricHistory.NCI], 'omitnan');
                meanNLL = mean([metricHistory.NLL], 'omitnan');
                meanES = mean([metricHistory.ES], 'omitnan');

                fprintf('      Results: ELT=%.2f, NCI=%.2f, NLL=%.2f, ES=%.2f\n', ...
                    meanELT, meanNCI, meanNLL, meanES);

                % Store scenario results
                scenarioResult.Baseline = L;
                scenarioResult.StochasticModel = weightMode;
                scenarioResult.Atmosphere = atm.name;
                scenarioResult.Metrics = metricHistory;
                scenarioResult.Summary = struct('ELT', meanELT, 'NCI', meanNCI, ...
                    'NLL', meanNLL, 'ES', meanES);

                if isempty(results)
                    results = scenarioResult;
                else
                    results(end+1) = scenarioResult;
                end
            end
        end
    end
end

%% ========================================================================
% SAVE RESULTS
% ========================================================================

if ~isempty(results)
    save('dgnss_simulation_results.mat', 'results');
    fprintf('\n=== Simulation Complete ===\n');
    fprintf('Results saved to: dgnss_simulation_results.mat\n');
    fprintf('Total scenarios completed: %d\n', length(results));
else
    warning('No results to save. Check simulation parameters.');
end

%% ========================================================================
% ANALYSIS AND VISUALIZATION
% Per Research Plan: Demonstrate credibility collapse with baseline length
% ========================================================================

if ~isempty(results)
    % Extract data
    baselines = [results.Baseline];
    ncis = arrayfun(@(x) x.Summary.NCI, results);
    elts = arrayfun(@(x) x.Summary.ELT, results);
    nlls = arrayfun(@(x) x.Summary.NLL, results);
    ess = arrayfun(@(x) x.Summary.ES, results);
    atmospheres = {results.Atmosphere};
    
    idxQuiet = strcmp(atmospheres, 'Quiet');
    idxActive = strcmp(atmospheres, 'Active');
    
    % Plot 1: NCI vs Baseline (Main credibility metric)
    % Expected: NCI≈1 at L=0 (Credible), NCI>>1 at L=50km (Pessimism + SMM)
    figure('Name', 'NCI vs Baseline', 'Position', [100, 100, 800, 600]);
    hold on;
    
    if any(idxQuiet & ~isnan(ncis))
        plot(baselines(idxQuiet), ncis(idxQuiet), 'b-o', ...
            'DisplayName', 'Quiet (L_{corr}=50km)', 'LineWidth', 2, 'MarkerSize', 8);
    end
    if any(idxActive & ~isnan(ncis))
        plot(baselines(idxActive), ncis(idxActive), 'r-x', ...
            'DisplayName', 'Active (L_{corr}=10km)', 'LineWidth', 2, 'MarkerSize', 8);
    end
    
    yline(1.0, 'k--', 'Expected (NCI=1)', 'LineWidth', 1.5);
    xlabel('Baseline Length (km)');
    ylabel('NCI (Normalized Consistency Index)');
    title('DGNSS Credibility: NCI vs Baseline Length');
    subtitle('Expected: NCI≈1 at L=0 (Credible), NCI>>1 at L=50km (Pessimism + SMM)');
    legend('Location', 'best');
    grid on;
    set(gca, 'YScale', 'log');
    saveas(gcf, 'nci_vs_baseline.png');
    
    % Plot 2: All Metrics Comparison
    figure('Name', 'All Metrics vs Baseline', 'Position', [150, 150, 1200, 800]);
    
    subplot(2,2,1);
    if any(idxQuiet & ~isnan(ncis)), plot(baselines(idxQuiet), ncis(idxQuiet), 'b-o', 'DisplayName', 'Quiet'); hold on; end
    if any(idxActive & ~isnan(ncis)), plot(baselines(idxActive), ncis(idxActive), 'r-x', 'DisplayName', 'Active'); end
    yline(1.0, 'k--'); xlabel('Baseline (km)'); ylabel('NCI'); title('NCI'); legend; grid on;
    
    subplot(2,2,2);
    if any(idxQuiet & ~isnan(elts)), plot(baselines(idxQuiet), elts(idxQuiet), 'b-o', 'DisplayName', 'Quiet'); hold on; end
    if any(idxActive & ~isnan(elts)), plot(baselines(idxActive), elts(idxActive), 'r-x', 'DisplayName', 'Active'); end
    xlabel('Baseline (km)'); ylabel('ELT'); title('ELT'); legend; grid on;
    
    subplot(2,2,3);
    if any(idxQuiet & ~isnan(nlls)), plot(baselines(idxQuiet), nlls(idxQuiet), 'b-o', 'DisplayName', 'Quiet'); hold on; end
    if any(idxActive & ~isnan(nlls)), plot(baselines(idxActive), nlls(idxActive), 'r-x', 'DisplayName', 'Active'); end
    xlabel('Baseline (km)'); ylabel('NLL'); title('NLL'); legend; grid on;
    
    subplot(2,2,4);
    if any(idxQuiet & ~isnan(ess)), plot(baselines(idxQuiet), ess(idxQuiet), 'b-o', 'DisplayName', 'Quiet'); hold on; end
    if any(idxActive & ~isnan(ess)), plot(baselines(idxActive), ess(idxActive), 'r-x', 'DisplayName', 'Active'); end
    xlabel('Baseline (km)'); ylabel('ES'); title('ES'); legend; grid on;
    
    sgtitle('DGNSS Credibility Metrics vs Baseline Length');
    saveas(gcf, 'all_metrics_vs_baseline.png');
    
    % Print Credibility Analysis Summary
    fprintf('\n=== Credibility Analysis Summary ===\n');
    fprintf('Per Research Plan: At L=0 errors cancel (Credible), at L=50km errors decorrelate (Pessimism + SMM)\n\n');
    
    for L = unique(baselines)
        idxL = baselines == L;
        nciL = ncis(idxL);
        validNCI = nciL(~isnan(nciL));
        if ~isempty(validNCI)
            meanNCI = mean(validNCI);
            fprintf('Baseline L=%.1f km: Mean NCI=%.3f', L, meanNCI);
            if meanNCI > 1.5
                fprintf(' -> Pessimism + SMM\n');
            elseif meanNCI < 0.5
                fprintf(' -> Optimism\n');
            else
                fprintf(' -> Credible\n');
            end
        end
    end
end
