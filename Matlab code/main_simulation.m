% main_simulation.m
% DGNSS Commissioning Protocol Validation via Block Averaging
% 
% Purpose: Implements the Design Doc for validating the "15-minute Block Averaging 
%          Commissioning Protocol". Tests whether block averaging can distinguish 
%          between three error scenarios: Multipath, SMM, and Optimism.
%
% Design Doc Reference: "Refactoring DGNSS Simulation for Commissioning Protocol 
%                        Verification (Block Averaging & Multipath Diagnosis)"
%
% MATLAB Version: R2025a
% Required Toolboxes: Navigation Toolbox, Satellite Communications Toolbox
%
% Output: 
%   - commissioning_protocol_results.mat: Structure array with metrics for each scenario
%   - elt_by_scenario.png: ELT results for each scenario
%   - nci_by_scenario.png: NCI results showing optimism detection
%
% Expected Results (per Design Doc):
%   - Multipath: ELT Pass after averaging (colored noise averages down)
%   - SMM: ELT Fail after averaging (constant bias persists)
%   - Optimism: ELT Pass, but NLL/NCI shows Optimism (underestimated uncertainty)

clear; clc; close all;

%% ========================================================================
% CONFIGURATION: Commissioning Protocol Parameters per Design Doc Section 3.2
% ========================================================================

% Test Scenarios: Replace baseline length loop with scenario types
% Per Design Doc Section 3.1
testScenarios = {'Multipath', 'SMM', 'Optimism'};

% Stochastic Models: Model A (elevation weighting)
% Per Design Doc: Use single model for clarity
stochasticModel = 'elevation';

% Block Averaging Parameters per Design Doc Section 3.2
% 15-minute blocks at 60 Hz sampling rate
blockDurationMinutes = 15;
sampleRateHz = 60;  % 60 Hz sampling (1 sample per second)
blockSize = blockDurationMinutes * 60 * sampleRateHz;  % 900 epochs per block (15 min * 60 sec/min * 1 Hz)

% Note: Design Doc specifies "e.g., 15 mins * 60Hz = 900 samples"
% If actual sampling is 1Hz (1 sample per second), then 15 min * 60 sec/min * 1 Hz = 900 samples
% Adjusting to match: 15 min * 60 sec/min * 1 Hz = 900
sampleRateHz = 1;  % 1 Hz sampling (1 sample per second)
blockSize = blockDurationMinutes * 60 * sampleRateHz;  % 900 epochs per block

% Simulation Parameters
startTime = datetime('now', 'TimeZone', 'UTC');
durationHours = 4;  % 4 hours to collect multiple blocks (4 hours * 60 min/hr / 15 min/block = 16 blocks)
sampleRate = 1 / sampleRateHz;  % Convert Hz to interval in seconds
baseLoc = [42.3601, -71.0589, 0];  % Base station location [lat, lon, alt] - Boston
baselineLength = 0.1;  % Short baseline (100m = 0.1 km)

% Results Storage
results = [];

%% ========================================================================
% MAIN SIMULATION LOOP: Test Each Scenario with Block Averaging
% ========================================================================

fprintf('=== DGNSS Commissioning Protocol Validation ===\n');
fprintf('Per Design Doc: Testing 15-minute Block Averaging Protocol\n');
fprintf('Block Size: %d epochs (%.1f minutes at %.0f Hz)\n\n', blockSize, blockDurationMinutes, sampleRateHz);

for iScenario = 1:length(testScenarios)
    scenarioType = testScenarios{iScenario};
    fprintf('Scenario: %s\n', scenarioType);
    
    %% Phase A: Scenario Generation (Design Doc Section 3.2)
    % Create satellite scenario and GNSS measurement generators
    [sc, gnssBase, gnssRover, baseObj, roverObj] = ...
        generate_scenario(startTime, durationHours, sampleRate, baseLoc, baselineLength);
    
    sats = sc.Satellites;
    if isempty(sats)
        warning('No satellites available. Skipping scenario.');
        continue;
    end
    
    %% Phase B: Block Buffering and Processing (Design Doc Section 3.2, Change 2)
    % Accumulate blockSize epochs, then compute blockMean and blockVariance
    
    blockBuffer = [];  % Buffer to accumulate residuals for current block
    blockMeans = [];   % Store mean of each block
    blockVariances = []; % Store variance of each block
    blockMetrics = [];  % Store raw metrics before averaging
    blockCount = 0;
    epochCount = 0;
    
    expectedEpochs = round(durationHours * 3600 * sampleRateHz);
    dt = 1 / sampleRateHz;  % Time step in seconds
    prevState = [];  % For Gauss-Markov state continuity
    
    fprintf('  Processing epochs (collecting blocks of %d epochs)...\n', blockSize);
    
    % Process epochs: Advance scenario and accumulate into blocks
    while epochCount < expectedEpochs * 2 && advance(sc)
        epochCount = epochCount + 1;
        
        % Get satellite and receiver states (ECEF coordinates)
        satStates = states(sats, "CoordinateFrame", "ecef");
        satStates = squeeze(satStates);
        satPos = satStates(:,1:3);
        satVel = satStates(:,4:6);
        
        baseState = states(baseObj, "CoordinateFrame", "ecef");
        basePos = baseState(1:3)';
        baseVel = baseState(4:6)';
        
        roverState = states(roverObj, "CoordinateFrame", "ecef");
        roverPos = roverState(1:3)';
        roverVel = roverState(4:6)';
        
        % Generate clean pseudorange measurements
        obsBase = step(gnssBase, basePos, baseVel, satPos, satVel);
        obsRover = step(gnssRover, roverPos, roverVel, satPos, satVel);
        
        % Validate measurements: Need at least 4 satellites
        if height(obsBase) < 4 || height(obsRover) < 4 || ...
           ~ismember('Pseudorange', obsBase.Properties.VariableNames)
            continue;
        end
        
        % Ensure required columns exist
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
        
        % Inject errors based on scenario type (Design Doc Section 3.1)
        [obsBaseErr, obsRoverErr, covarianceScale] = ...
            inject_errors(obsBase, obsRover, scenarioType, 'dt', dt, 'prevState', prevState);
        
        % Update Gauss-Markov state for Multipath scenario
        % (Simplified: full implementation would track per-satellite states)
        if strcmp(scenarioType, 'Multipath')
            prevState.base = zeros(height(obsBase), 1);  % Placeholder
            prevState.rover = zeros(height(obsRover), 1);
        end
        
        % DGNSS Processing: Double-Difference Least Squares solver
        [posEst, residuals, H, W] = dgnss_solver(obsBaseErr, obsRoverErr, basePos, stochasticModel);
        
        % Apply covariance scaling for Optimism scenario
        % Per Design Doc Section 3.1: Scale reported covariance
        if covarianceScale ~= 1.0
            W = W / covarianceScale;  % Scale weight matrix (W = inv(Cov))
        end
        
        % Accumulate residuals into block buffer
        if ~isempty(residuals)
            % Store residuals for this epoch
            epochData.residuals = residuals;
            epochData.H = H;
            epochData.W = W;
            
            if isempty(blockBuffer)
                blockBuffer = epochData;
            else
                blockBuffer(end+1) = epochData;
            end
        end
        
        % Check if block is complete
        if length(blockBuffer) >= blockSize
            blockCount = blockCount + 1;
            
            % Compute block mean and variance (Design Doc Section 3.2, Change 2)
            % Extract all residuals in this block
            allResiduals = [];
            for k = 1:length(blockBuffer)
                allResiduals = [allResiduals; blockBuffer(k).residuals];
            end
            
            % Block statistics
            blockMean = mean(allResiduals);
            blockVariance = var(allResiduals);
            
            blockMeans = [blockMeans; blockMean];
            blockVariances = [blockVariances; blockVariance];
            
            % Calculate metrics on the averaged block (use last epoch's H and W as representative)
            % Note: This is a simplification; full implementation would aggregate H and W properly
            lastH = blockBuffer(end).H;
            lastW = blockBuffer(end).W;
            blockResidual = mean(allResiduals);  % Single mean residual for the block
            
            if ~isempty(blockResidual) && ~isnan(blockResidual)
                % Calculate metrics for this block mean
                % Note: Metrics expect vector residuals; here we have scalar block mean
                % For proper implementation, collect residuals per double-difference, then average
                metrics.BlockMean = blockMean;
                metrics.BlockVariance = blockVariance;
                metrics.BlockCount = blockCount;
                metrics.Scenario = scenarioType;
                
                if isempty(blockMetrics)
                    blockMetrics = metrics;
                else
                    blockMetrics(end+1) = metrics;
                end
            end
            
            % Clear buffer for next block
            blockBuffer = [];
            
            fprintf('    Block %d complete: Mean=%.4f m, Variance=%.4f m²\n', ...
                blockCount, blockMean, blockVariance);
        end
    end
    
    %% Phase C: Diagnosis Pipeline (Design Doc Section 3.2, Change 3)
    % Step A: Compute ELT on the sequence of blockMeans
    % Step B: Pre-whitening - subtract mean from blockMeans
    % Step C: Compute NCI/NLL on residuals to detect Optimism
    
    if ~isempty(blockMeans) && length(blockMeans) > 1
        fprintf('  Diagnosis Pipeline:\n');
        
        % Step A: Compute ELT on sequence of block means
        % ELT = sum of squared standardized block means
        % Assuming blockMeans ~ N(0, σ²/blockSize) under null hypothesis
        meanOfBlockMeans = mean(blockMeans);
        stdOfBlockMeans = std(blockMeans);
        
        if stdOfBlockMeans > 0
            standardizedBlockMeans = (blockMeans - meanOfBlockMeans) / stdOfBlockMeans;
            ELT = sum(standardizedBlockMeans.^2);
            dof = length(blockMeans) - 1;  % Degrees of freedom
            ELT_p_value = 1 - chi2cdf(ELT, dof);
            
            fprintf('    Step A - ELT on block means: %.4f (p-value: %.4f, dof: %d)\n', ...
                ELT, ELT_p_value, dof);
            
            % ELT Test Result
            if ELT_p_value > 0.05
                ELT_result = 'Pass';
            else
                ELT_result = 'Fail';
            end
            fprintf('    ELT Result: %s (Expected: %s)\n', ELT_result, ...
                getExpectedELT(scenarioType));
        else
            ELT = NaN;
            ELT_p_value = NaN;
            ELT_result = 'N/A';
        end
        
        % Step B: Pre-whitening - subtract mean from blockMeans to get residuals
        residualsPreWhitened = blockMeans - meanOfBlockMeans;
        
        % Step C: Compute NCI on residuals to detect Optimism
        % NCI = variance / expected_variance
        % For white noise after averaging, expected variance = σ²/blockSize
        varianceOfResiduals = var(residualsPreWhitened);
        expectedVariance = mean(blockVariances) / blockSize;  % Theoretical variance after averaging
        
        if expectedVariance > 0
            NCI = varianceOfResiduals / expectedVariance;
            fprintf('    Step C - NCI: %.4f (Expected: 1.0 for credible model)\n', NCI);
            
            % NCI Interpretation
            if NCI > 1.5
                NCI_interpretation = 'Pessimism';
            elseif NCI < 0.5
                NCI_interpretation = 'Optimism';
            else
                NCI_interpretation = 'Credible';
            end
            fprintf('    NCI Interpretation: %s (Expected: %s for %s)\n', ...
                NCI_interpretation, getExpectedNCI(scenarioType), scenarioType);
        else
            NCI = NaN;
            NCI_interpretation = 'N/A';
        end
        
        % Store scenario results
        scenarioResult.Scenario = scenarioType;
        scenarioResult.BlockMeans = blockMeans;
        scenarioResult.BlockVariances = blockVariances;
        scenarioResult.BlockMetrics = blockMetrics;
        scenarioResult.Summary = struct('ELT', ELT, 'ELT_p_value', ELT_p_value, ...
            'ELT_result', ELT_result, 'NCI', NCI, 'NCI_interpretation', NCI_interpretation, ...
            'BlockCount', blockCount);
        
        if isempty(results)
            results = scenarioResult;
        else
            results(end+1) = scenarioResult;
        end
    else
        warning('Insufficient blocks for scenario %s. Need at least 2 blocks.', scenarioType);
    end
    
    fprintf('\n');
end

%% ========================================================================
% SAVE RESULTS
% ========================================================================

if ~isempty(results)
    save('commissioning_protocol_results.mat', 'results');
    fprintf('=== Simulation Complete ===\n');
    fprintf('Results saved to: commissioning_protocol_results.mat\n');
    fprintf('Total scenarios completed: %d\n', length(results));
else
    warning('No results to save. Check simulation parameters.');
end

%% ========================================================================
% VISUALIZATION: Demonstrate Protocol Effectiveness
% Per Design Doc: Show ELT and NCI results for each scenario
% ========================================================================

if ~isempty(results)
    % Plot 1: ELT Results by Scenario
    figure('Name', 'ELT by Scenario', 'Position', [100, 100, 900, 600]);
    
    scenarios = {results.Scenario};
    elt_values = arrayfun(@(x) x.Summary.ELT, results);
    elt_p_values = arrayfun(@(x) x.Summary.ELT_p_value, results);
    
    subplot(2,1,1);
    bar(categorical(scenarios), elt_values);
    ylabel('ELT Statistic');
    title('ELT Results by Scenario');
    subtitle('Expected: Multipath Pass, SMM Fail, Optimism Pass');
    grid on;
    
    subplot(2,1,2);
    bar(categorical(scenarios), elt_p_values);
    hold on;
    yline(0.05, 'r--', 'α = 0.05', 'LineWidth', 1.5);
    ylabel('ELT p-value');
    xlabel('Scenario');
    title('ELT Significance Level');
    grid on;
    
    saveas(gcf, 'elt_by_scenario.png');
    
    % Plot 2: NCI Results by Scenario
    figure('Name', 'NCI by Scenario', 'Position', [150, 150, 900, 600]);
    
    nci_values = arrayfun(@(x) x.Summary.NCI, results);
    
    bar(categorical(scenarios), nci_values);
    hold on;
    yline(1.0, 'g--', 'Credible (NCI=1)', 'LineWidth', 1.5);
    yline(0.5, 'b--', 'Optimism Threshold', 'LineWidth', 1.5);
    yline(1.5, 'r--', 'Pessimism Threshold', 'LineWidth', 1.5);
    ylabel('NCI (Normalized Consistency Index)');
    xlabel('Scenario');
    title('NCI Results by Scenario');
    subtitle('Expected: Optimism shows NCI < 1.0');
    grid on;
    
    saveas(gcf, 'nci_by_scenario.png');
    
    % Print Summary Table
    fprintf('\n=== Commissioning Protocol Validation Summary ===\n');
    fprintf('%-12s | %8s | %10s | %8s | %12s\n', 'Scenario', 'ELT', 'p-value', 'NCI', 'Interpretation');
    fprintf('%s\n', repmat('-', 1, 70));
    
    for i = 1:length(results)
        fprintf('%-12s | %8.2f | %10.4f | %8.2f | %12s\n', ...
            results(i).Scenario, ...
            results(i).Summary.ELT, ...
            results(i).Summary.ELT_p_value, ...
            results(i).Summary.NCI, ...
            results(i).Summary.NCI_interpretation);
    end
    fprintf('\n');
    
    % Validate Expected Results
    fprintf('=== Validation Against Design Doc Expectations ===\n');
    for i = 1:length(results)
        scenario = results(i).Scenario;
        elt_result = results(i).Summary.ELT_result;
        nci_interp = results(i).Summary.NCI_interpretation;
        
        expectedELT = getExpectedELT(scenario);
        expectedNCI = getExpectedNCI(scenario);
        
        eltMatch = strcmp(elt_result, expectedELT);
        nciMatch = strcmp(nci_interp, expectedNCI);
        
        fprintf('%-12s: ELT %s (Expected: %s) [%s], NCI %s (Expected: %s) [%s]\n', ...
            scenario, elt_result, expectedELT, getStatus(eltMatch), ...
            nci_interp, expectedNCI, getStatus(nciMatch));
    end
end

%% ========================================================================
% HELPER FUNCTIONS
% ========================================================================

function expected = getExpectedELT(scenario)
    % Per Design Doc Section 3.1
    switch scenario
        case 'Multipath'
            expected = 'Pass';  % Colored noise averages down
        case 'SMM'
            expected = 'Fail';  % Constant bias persists
        case 'Optimism'
            expected = 'Pass';  % White noise averages properly
        otherwise
            expected = 'Unknown';
    end
end

function expected = getExpectedNCI(scenario)
    % Per Design Doc Section 3.1
    switch scenario
        case 'Multipath'
            expected = 'Credible';  % NCI ≈ 1.0
        case 'SMM'
            expected = 'Credible';  % NCI ≈ 1.0 (bias doesn't affect variance estimate)
        case 'Optimism'
            expected = 'Optimism';  % NCI < 1.0 (underestimated uncertainty)
        otherwise
            expected = 'Unknown';
    end
end

function status = getStatus(isMatch)
    if isMatch
        status = 'PASS';
    else
        status = 'FAIL';
    end
end
