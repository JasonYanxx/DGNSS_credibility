**Title**: Refactoring DGNSS Simulation for Commissioning Protocol Verification (Block Averaging & Multipath Diagnosis)
**Author**: [Your Name] / AI Assistant
**Status**: Proposed
**Last Updated**: 2026-01-10

#### 1. Objective

To modify the existing DGNSS simulation pipeline to validate the "15-minute Block Averaging Commissioning Protocol". The goal is to generate three specific scenarios (Multipath, SMM, Optimism), apply block averaging, and demonstrate that the proposed method can distinguish these error sources using ELT and NLL/ES.

#### 2. Background

* **Current State**: The code simulates spatial decorrelation over varying baselines using decimation (sub-sampling).
* **Target State**: The code must simulate **time-correlated multipath (Colored Noise)** and **constant biases (SMM)**. Instead of decimation, it must perform **Block Averaging** (e.g., mean of 900 epochs).

#### 3. Detailed Design

##### 3.1. `inject_errors.m` Refactor (Error Injection Engine)

* **Change**: Replace the `baselineLength`/`correlationLength` logic with a `ScenarioType` switch.
* **New Scenarios**:
* `'Multipath'`: Inject Gauss-Markov process (Time constant ) or Sine wave (Period  min). *Target: ELT Pass after averaging.*
* `'SMM'`: Inject Constant Bias (e.g., 0.5m) + White Noise. *Target: ELT Fail after averaging.*
* `'Optimism'`: Inject White Noise, but tell the solver the variance is smaller (scale reported covariance). *Target: ELT Pass, NLL/NCI shows Optimism.*



##### 3.2. `main_simulation.m` Refactor (Processing Logic)

* **Change 1 (Configuration)**: Replace `baselineLengths` loop with `testScenarios` loop (Multipath, SMM, Optimism).
* **Change 2 (Batching Logic)**:
* Remove `decimationInterval`.
* Implement **Block Buffering**: Accumulate `batchSize` epochs (e.g., 15 mins * 60Hz = 900 samples).
* Compute `blockMean` and `blockVariance`.


* **Change 3 (Diagnosis Pipeline)**:
* **Step A**: Compute ELT on the sequence of `blockMeans`.
* **Step B (Pre-whitening)**: Subtract the mean from `blockMeans` to get residuals.
* **Step C**: Compute NCI/NLL on these residuals to detect Optimism.



##### 3.3. `calculate_metrics.m` (Metric Calculation)

* **No functional change**, but ensure inputs are interpreted as "Block Means" rather than raw residuals.

#### 4. Interfaces

* `inject_errors(..., scenarioType)`: New signature.
* `block_averager(raw_residuals, block_size)`: New helper function (inline or separate).
