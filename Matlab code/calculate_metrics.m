function metrics = calculate_metrics(residuals, H, W, nUnknowns)
% CALCULATE_METRICS Computes credibility metrics for DGNSS residuals
%
% Purpose: Implements Research Plan Section 3, Phase C - Metrics Calculation
%          Computes four credibility metrics to diagnose model misspecification:
%          1. ELT (Empirical Location Test): Global model test statistic
%          2. NCI (Normalized Consistency Index): SSE/DOF (expected = 1.0)
%          3. NLL (Negative Log Likelihood): Model fit quality
%          4. ES (Energy Score): Probabilistic forecast verification
%
% Expected Behavior per Research Plan:
%   - Credible model (L=0): NCI≈1, ELT small, NLL reasonable, ES small
%   - Pessimistic model (L=50km): NCI>>1, ELT large, NLL large, ES large
%
% Inputs:
%   residuals - Post-fit double-difference residuals (column vector, meters)
%   H         - Design matrix (Double Difference)
%   W         - Weight matrix (inverse of observation covariance)
%   nUnknowns - Number of estimated parameters (3 for position)
%
% Outputs:
%   metrics   - Structure containing:
%               .ELT: Empirical Location Test statistic (SSE)
%               .NCI: Normalized Consistency Index (SSE/DOF)
%               .NLL: Negative Log Likelihood
%               .ES:  Energy Score

if isempty(residuals)
    metrics.ELT = NaN;
    metrics.NCI = NaN;
    metrics.NLL = NaN;
    metrics.ES = NaN;
    return;
end

nObs = length(residuals);
dof = nObs - nUnknowns;  % Degrees of freedom

%% Metric 1: ELT (Empirical Location Test)
% ELT = v' * W * v (quadratic form of residuals)
% This is the Global Model Test statistic
% Under null hypothesis (model correct): ELT ~ chi2(dof)
sse = residuals' * W * residuals;
metrics.ELT = sse;
metrics.ELT_p_value = 1 - chi2cdf(sse, dof);

%% Metric 2: NCI (Normalized Consistency Index)
% NCI = SSE / DOF
% Expected value: 1.0 for a well-calibrated model
%   - NCI ≈ 1: Credible model
%   - NCI >> 1: Pessimistic model (underestimated uncertainty)
%   - NCI << 1: Optimistic model (overestimated uncertainty)
if dof > 0
    metrics.NCI = sse / dof;
else
    metrics.NCI = NaN;
end

%% Metric 3: NLL (Negative Log Likelihood)
% NLL = 0.5 * (v'*W*v + ln(det(Cov)) + n*ln(2*pi))
% Lower NLL = better model fit
% For residuals v ~ N(0, Cov), where Cov = inv(W)
% ln(det(Cov)) = -ln(det(W))
L_chol = chol(W, 'lower');
log_det_W = 2 * sum(log(diag(L_chol)));
metrics.NLL = 0.5 * (sse - log_det_W + nObs * log(2 * pi));

%% Metric 4: ES (Energy Score)
% Energy Score for probabilistic forecast verification
% ES = E[||X - y||] - 0.5 * E[||X - X'||]
% where X ~ N(0, C_v) is the predictive distribution and y is the observation
% Lower ES = better probabilistic forecast
% C_v = Cov(residuals) = inv(W) - H * inv(H'*W*H) * H'

% Compute residual covariance matrix
Cov_obs = pinv(W);
HWH = H' * W * H;
Cov_x = pinv(HWH);
C_v = Cov_obs - H * Cov_x * H';

% Ensure positive semi-definite
[V, D] = eig(C_v);
D = max(D, 1e-10 * eye(size(D)));
C_v = V * D * V';

% Monte Carlo approximation of Energy Score
nMC = 100;
X = mvnrnd(zeros(nObs, 1), C_v, nMC);  % Samples from N(0, C_v)
X_prime = mvnrnd(zeros(nObs, 1), C_v, nMC);  % Independent samples

% Term 1: E[||X - y||]
diff_X_v = X - repmat(residuals', nMC, 1);
term1 = mean(sqrt(sum(diff_X_v.^2, 2)));

% Term 2: E[||X - X'||]
diff_X_X = X - X_prime;
term2 = mean(sqrt(sum(diff_X_X.^2, 2)));

metrics.ES = term1 - 0.5 * term2;

end
