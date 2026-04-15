function [GMModel, mu_gmm, sig_gmm] = gmm_identification(feature, cfg)

% NOTE: Number of regimes (K)
%
% In the manuscript, K = 3 is used based on the full dataset.
% The example dataset included in this repository is reduced in size,
% so automatic model-selection criteria (e.g., XB index, BIC) may yield
% slightly different optimal K values.
% For reproducibility and testing purposes, K is fixed to 3 here.

% Outputs:
%   GMModel : fitted Gaussian mixture model in normalized feature space
%   mu_gmm  : train-set feature means used for normalization
%   sig_gmm : train-set feature standard deviations used for normalization

arguments
    feature                 (:,:) double
    cfg.n_components        (1,1) double {mustBeInteger, mustBePositive} = 3  % default: fixed to paper setting
end

rng(42, 'twister');

% --- Normalize
mu_gmm = mean(feature, "omitnan");
sig_gmm = std(feature, "omitnan");

sig_gmm(sig_gmm < 1e-8) = 1;  % avoid division by zero
X_norm = (feature - mu_gmm) ./ sig_gmm;

% --- KMeans init
n_components = cfg.n_components;
[idx_km, ~] = kmeans(X_norm, n_components, 'Replicates', 20);

% --- Fit GMM
GMModel = fitgmdist(X_norm, n_components, ...
    'Start', idx_km, ...
    'RegularizationValue', 1e-5, ...
    'Options', statset('MaxIter', 500));

fprintf('GMM fitted with %d components\n', n_components);

end
