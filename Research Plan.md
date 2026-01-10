### **Research Title**

**"Credibility Diagnosis of Stochastic Models in DGNSS: Unmasking the Impact of Spatial Decorrelation via a Multi-Metric Framework"**

### **1. The Core Concept (The "Elevator Pitch")**

Standard DGNSS works on the assumption that errors at the Base and Rover are identical and cancel out. This paper uses the Yan et al. framework to prove that as the baseline length increases, this assumption fails in a way that standard consistency tests (NEES) miss, but your multi-metric framework (ELT + NLL + ES) catches. You will demonstrate exactly *when* and *how* the credibility collapses.

-----

### **2. Objectives**

1.  **Diagnose Optimism:** Quantify how "Standard Elevation Weighting" becomes dangerously optimistic as baseline length increases from 1km to 50km.
2.  **Detect SMM:** Use the **Empirical Location Test (ELT)** to detect the onset of "Atmospheric Bias" (a System Model Misspecification) that appears when the troposphere decorrelates.
3.  **Validate the Fix:** Demonstrate that a more robust stochastic model (e.g., one that accounts for baseline length variance) restores credibility.

-----

### **3. Methodology: The MATLAB Simulation Pipeline**

You will not need to write complex ray-tracing physics. You will use a **"Geometry + Injection"** strategy.

#### **Phase A: Scenario Generation (The "Clean" Data)**

**Tool:** `satelliteScenario` & `gnssMeasurementGenerator`

1.  **Create the Arena:** Initialize a `satelliteScenario` with a specific start time (e.g., current date).
2.  **Add Constellations:** Use `satellite(sc, "gpsAlmanac.txt")` to load real GPS orbits.
3.  **Place Receivers:**
      * **Base Station:** Fixed at location $P_{base}$.
      * **Rover Station:** Placed at distances $L \in \{0.1, 1, 5, 10, 20, 50\}$ km from the Base.
4.  **Generate Observables:** Use `gnssMeasurementGenerator` to output **Clean Pseudoranges** (Geometric Range + Clock Bias). Set the noise to zero in the generator settings.

#### **Phase B: The Error Injection Engine (Your Contribution)**

You will write a MATLAB script to add the "Research Grade" errors.
$$\rho_{obs} = \rho_{clean} + \delta_{tropo} + \delta_{iono} + \delta_{multipath} + \delta_{thermal}$$

  * **Thermal Noise:** Independent Gaussian noise ($ \sigma \approx 20$ cm).
  * **Correlated Atmosphere (The Key):**
    You will simulate the error at the Rover ($\epsilon_r$) based on the error at the Base ($\epsilon_b$) using a spatial correlation parameter $\tau(L)$:
    $$\epsilon_{r} = \epsilon_{b} \cdot e^{-L/L_{corr}} + \eta \cdot \sqrt{1 - e^{-2L/L_{corr}}}$$
      * Where $L_{corr}$ is the correlation distance (e.g., 30km).
      * This ensures that at $L=0$, errors are identical (perfect DGNSS), and at $L=50km$, they are independent (DGNSS fails).

#### **Phase C: The Processing & Diagnosis**

1.  **DGNSS Solver:** Implement a simple Double-Difference (DD) Least Squares solver in MATLAB.
2.  **Decimation:** Sub-sample the output (e.g., 1 epoch every 5 mins) to ensure statistical independence.
3.  **Metrics:** Calculate ELT, NCI, NLL, and ES on the residuals.

-----

### **4. Experimental Design (The "Grid")**

Run the simulation for **24 hours** (to capture full geometry changes) for each cell in this grid:

| Variable | Settings |
| :--- | :--- |
| **Baseline Length ($L$)** | 0km (Zero Baseline), 1km, 10km, 30km, 50km |
| **Stochastic Model ($W$)** | **Model A:** Standard Elevation ($1/\sin^2(El)$)<br>**Model B:** SNR-Based (using C/N0 from generator) |
| **Atmosphere Condition** | **Quiet:** $L_{corr} = 50km$ (Errors correlate well)<br>**Active:** $L_{corr} = 10km$ (Errors decorrelate fast) |

-----

### **5. MATLAB Implementation Guide (Pseudocode)**

Here is the skeleton code you can copy into MATLAB to get started.

#### **Step 1: The Scenario Generator**

```matlab
% 1. Setup Scenario
startTime = datetime('now','TimeZone','UTC');
sc = satelliteScenario(startTime, startTime + hours(24), 30); % 30s interval
sats = satellite(sc, "gpsAlmanac.txt"); % Load GPS

% 2. Define Base and Rover (e.g., 10km apart)
baseLoc = [42.3601, -71.0589, 0]; % Boston
[latR, lonR, hR] = moveLatLon(baseLoc(1), baseLoc(2), 10000); % Custom function to move 10km North
roverLoc = [latR, lonR, 0];

base = groundStation(sc, baseLoc(1), baseLoc(2), "Name", "Base");
rover = groundStation(sc, roverLoc(1), roverLoc(2), "Name", "Rover");

% 3. Create Generators (Zero Noise initially)
gnssBase = gnssMeasurementGenerator('ReferenceLocation', baseLoc, ...
    'Noise', 0, 'SampleRate', 1/30); 
gnssRover = gnssMeasurementGenerator('ReferenceLocation', roverLoc, ...
    'Noise', 0, 'SampleRate', 1/30);
```

#### **Step 2: The Error Injection (The Research Logic)**

```matlab
% Loop through epochs
for i = 1:numEpochs
    % Get Clean Ranges
    [obsBase, statusBase] = gnssBase(basePos, baseVel);
    [obsRover, statusRover] = gnssRover(roverPos, roverVel);
    
    % --- YOUR CONTRIBUTION STARTS HERE ---
    % Simulate Base Troposphere Error (Random Walk or Gauss-Markov)
    err_tropo_base = randn(numSats, 1) * 2.0; % Large Tropo Error (2m)
    
    % Calculate Spatial Correlation Coefficient
    L = 10; % km
    L_corr = 30; % km (Correlation length)
    rho = exp(-L / L_corr);
    
    % Generate Correlated Rover Error
    % E_rover = rho * E_base + sqrt(1 - rho^2) * Independent_Noise
    err_tropo_rover = (rho * err_tropo_base) + ...
                      (sqrt(1 - rho^2) * randn(numSats, 1) * 0.2);
    
    % Add to Clean Ranges
    obsBase.Pseudorange = obsBase.Pseudorange + err_tropo_base;
    obsRover.Pseudorange = obsRover.Pseudorange + err_tropo_rover;
    
    % Store for Processing
    save_epoch_data(i, obsBase, obsRover);
end
```

#### **Step 3: Diagnosis**

After running the DGNSS solver on this data, you compute the metrics.

  * **If $L=0$ (Zero Baseline):** `err_tropo_rover` will equal `err_tropo_base`. They cancel out in Double Difference. Residuals are pure noise. **Diagnosis: Credible.**
  * **If $L=50$ (Long Baseline):** `rho` becomes small. The errors don't cancel. The solver *thinks* the noise is 20cm, but the residual is 1.5m. **Diagnosis: Pessimism + SMM.**

-----

### **6. Next Actions for You**

1.  **Check your Toolbox:** Run `ver` in MATLAB to ensure you have "Navigation Toolbox" and "Satellite Communications Toolbox".
2.  **Copy the Logic:** Use the code skeleton above to generate your first "Synthetic Dataset."
3.  **The Solver:** Do you have an existing DGNSS solver script in MATLAB, or do you need a simple one (Double Difference WLS) to process this data?

This plan allows you to control every variable (the "God mode" of simulation) while using validated MATLAB tools for the geometry. It is perfectly scoped for a high-quality paper.