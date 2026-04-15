function [] = reproduce_minimal_comparison(cfg)

% REPRODUCE_MINIMAL_COMPARISON
%
% This script reproduces the minimal comparison experiment presented in the manuscript.
% It evaluates the effect of regime partitioning (GMM) and physics-informed (PI)
% regularization on turbulent flux estimation.
%
% Four model configurations are tested:
%   (1) Single model without PI (no GMM, λ = 0)
%   (2) Single model with PI (no GMM, λ = default)
%   (3) Regime-aware model without PI (GMM, λ = 0)
%   (4) Regime-aware model with PI (GMM + PI, proposed method)
%
% The script performs the following steps:
%   1. Load example datasets (train/test)
%   2. Compute observation-derived variables
%   3. Construct input features and target variables
%   4. Run the four configurations using run_regime_model
%   5. Evaluate performance (RMSE, MAE, Pearson R)
%   6. Output results as a comparison table
%
% Inputs (cfg):
%   cfg.target               Target variable ("ustar", "tstar", or "qstar")
%   cfg.min_train_per_regime Minimum training samples required per regime
%
% Notes:
%   - The example dataset is preprocessed and anonymized.
%   - The PI regularization weight (λ) is target-dependent and determined
%     by default settings unless overridden.
%   - This script is intended for reproducibility and comparison only.
%
% See also:
%   main.m                    % proposed method only
%   run_regime_model.m        % core training and prediction routine
%   get_prediction_MLP_SA_PI  % model implementation

arguments
    cfg.min_train_per_regime (1,1) double  = 100
    cfg.target               (1,1) string  = "ustar"
end

clc;

rng(42,'twister');

fprintf('\n')
fprintf('Regime-Aware Flux Estimation (Minimal Comparison on Example Dataset)\n');
fprintf('--------------------------------------------------\n');

%% Setup paths (relative paths)
PATH_exp = fileparts(mfilename('fullpath'));
PATH_code = fullfile(PATH_exp, '..');
PATH_exampledata = fullfile(PATH_code, '..', 'data', 'sample');
addpath(genpath(PATH_code));

fprintf('Working directory: %s\n', PATH_code);
fprintf('Data directory: %s\n', PATH_exampledata);
fprintf('\n')

% Initialize Python environment and import modules
fprintf('Initializing Python environment... \n')
[~, ~] = py_init_mo();

%% 1) Prepare data

% Load data
% Required example files in ../data/sample/:
%   - example_data_train.mat
%   - example_data_test.mat

target = validate_target(cfg.target);
fprintf('\n=== Loading datasets ===\n');

data_train_file = 'example_data_train.mat';
if exist(fullfile(PATH_exampledata, data_train_file), "file")
    tmp = load(fullfile(PATH_exampledata, data_train_file));
    data_train = struct2table_if_needed(tmp.data_train);
    fprintf('Loaded: %s (%d samples)\n', data_train_file, height(data_train));
else
    error('Data file not found: %s\nPlease place your data file in the data folder.', data_train_file);
end

data_test_file = 'example_data_test.mat';
if exist(fullfile(PATH_exampledata, data_test_file), "file")
    tmp = load(fullfile(PATH_exampledata, data_test_file));
    data_test = struct2table_if_needed(tmp.data_test);
    fprintf('Loaded: %s (%d samples)\n', data_test_file, height(data_test));
else
    error('Data file not found: %s\nPlease place your data file in the data folder.', data_test_file);
end

% Obs calculations
fprintf('\n=== Preparing datasets ===\n');

fprintf('Calculating observations...\n');
obs_cal_train = obs_calculation(data_train);
obs_cal_test = obs_calculation(data_test);

% Data clean
fprintf('Cleaning datasets...\n');
[data_train, obs_cal_train, ~]  = data_clean(data_train, obs_cal_train);
[data_test, obs_cal_test, ~]  = data_clean(data_test, obs_cal_test);

% Monin-Obukhov calculations
fprintf('Calculating M-O similarity...\n');
mo_cal_train = mo_calculation(data_train);

% Input feature preparation
fprintf('Preparing input features...\n');

% Net radiation pattern
Rn_pattern = @(data) data.DR_Avg - data.UR_Avg + data.DLR_Avg - data.ULR_Avg;

% Input feature pattern (10 features)
input_pattern = @(data, obs_cal) [  ...
    data.Ta_35M_Avg,...                                % 1: Temperature
    (data.Ta_35M_Avg - data.Ta_10M_Avg) ./ (35-10),... % 2: Temperature gradient
    data.WS_35M,...                                    % 3: Wind speed
    (data.WS_35M - data.WS_10M) ./ (35-10),...         % 4: Wind speed gradient
    data.RH_35M_Avg,...                                % 5: Relative humidity
    (data.RH_35M_Avg - data.RH_10M_Avg) ./ (35-10),... % 6: RH gradient
    data.P,...                                         % 7: Pressure
    Rn_pattern(data),...                               % 8: Net radiation
    data.SAA,...                                       % 9: Solar altitude angle
    obs_cal.bulk_richardson_number...                  % 10: Bulk Richardson number
    ];

% Target pattern
switch target
    case "ustar"
        target_pattern = @(obs_cal) obs_cal.friction_velocity_m_s;
    case "tstar"
        target_pattern = @(obs_cal) obs_cal.temperature_scale_k;
    case "qstar"
        target_pattern = @(obs_cal) obs_cal.moisture_scale_g_kg;
    otherwise
        error("Unexpected target name of cfg.target")
end

% input-output datasets
input_train = input_pattern(data_train, obs_cal_train);
target_train = target_pattern(obs_cal_train);

input_test = input_pattern(data_test, obs_cal_test);
target_test = target_pattern(obs_cal_test);

% data structure check
assert(istable(data_train) && istable(data_test), "Loaded data must be a table/timetable converted to table.");
assert(size(input_train,2) == size(input_test,2), "Train/Test feature dimension mismatch.");
assert(size(input_train,1) == numel(target_train), "Train X/Y length mismatch.");

%% 2) Run four configurations

fprintf('\n==================================================\n');
fprintf('Running comparison models...\n');
fprintf('==================================================\n');

% - 1 - without GMM, without PI
fprintf('\n[1/4] Single model (no GMM, no PI)...\n');
t = tic;
[Ypred_single, ~] = ...
    run_regime_model(input_train, target_train, mo_cal_train, input_test, target=target, n_components=1, lambda_PI_override=0);
fprintf('Done (%.1fs)\n', toc(t));

% - 2 - without GMM, PI
fprintf('\n[2/4] Single model + PI...\n');
t = tic;
[Ypred_single_PI, ~] = ...
    run_regime_model(input_train, target_train, mo_cal_train, input_test, target=target, n_components=1);
fprintf('Done (%.1fs)\n', toc(t));

% - 3 - GMM, without PI
fprintf('\n[3/4] Regime-aware model (GMM), no PI...\n');
t = tic;
[Ypred_GMM, ~] = ...
    run_regime_model(input_train, target_train, mo_cal_train, input_test, target=target, lambda_PI_override=0);
fprintf('Done (%.1fs)\n', toc(t));

% - 4 - GMM + PI (proposed)
fprintf('\n[4/4] Regime-aware model + PI (proposed)...\n');
t = tic;
[Ypred_proposed, ~] = ...
    run_regime_model(input_train, target_train, mo_cal_train, input_test, target=target);
fprintf('Done (%.1fs)\n', toc(t));

%% 3) Evaluate

metrics1 = evaluate_prediction(Ypred_single, target_test);
metrics2 = evaluate_prediction(Ypred_single_PI, target_test);
metrics3 = evaluate_prediction(Ypred_GMM, target_test);
metrics4 = evaluate_prediction(Ypred_proposed, target_test);

resultTable = table( ...
    ["Single (no GMM, no PI)";
     "Single + PI";
     "Regime-aware (GMM), no PI";
     "Regime-aware + PI (proposed)"], ...
    [metrics1.RMSE; metrics2.RMSE; metrics3.RMSE; metrics4.RMSE], ...
    [metrics1.R;    metrics2.R;    metrics3.R;    metrics4.R], ...
    'VariableNames', {'Method', 'RMSE', 'R'});

disp(resultTable)

end

function metrics = evaluate_prediction(Ypred, Ytrue)

% Remove NaNs
valid_idx = ~isnan(Ypred) & ~isnan(Ytrue);
Yp = Ypred(valid_idx);
Yt = Ytrue(valid_idx);

% Metrics
metrics.RMSE = sqrt(mean((Yt - Yp).^2));
metrics.MAE  = mean(abs(Yt - Yp));
metrics.R    = corr(Yt, Yp, "Type", "Pearson");

end