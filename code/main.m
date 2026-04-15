% function [] = main(cfg)

% Regime-Aware Flux Estimation using Gaussian Mixture Models and Self-Attention
% Example script for reproducing the workflow in the manuscript
%
% This code implements:
% 1. Data preparation and observation calculations
% 2. Monin-Obukhov Similarity Theory (MOST) calculations
% 3. Gaussian Mixture Model (GMM) for regime identification
% 4. Self-Attention MLP with Physics-Informed Loss
% 
% Usage:
%   main()
%
% Requirements:
% - MATLAB R2022b or later
% - Deep Learning Toolbox
% - Statistics and Machine Learning Toolbox

% arguments
%     cfg.do_plot              (1,1) logical = true
%     cfg.save_plot            (1,1) logical = false
%     cfg.min_train_per_regime (1,1) double  = 100
%     cfg.target               (1,1) string  = "ustar"
% end

cfg.do_plot               = true;
cfg.save_plot             = false;
cfg.min_train_per_regime  = 100;
cfg.target                = "qstar";

clc;

rng(42,'twister');
t0 = tic;

fprintf('\n')
fprintf('Regime-Aware Flux Estimation (Example Run)\n');
fprintf('--------------------------------------------------\n');

%% Setup paths (relative paths)
PATH_code = fileparts(mfilename('fullpath'));
PATH_exampledata = fullfile(PATH_code, '..', 'data', 'sample');
addpath(genpath(PATH_code));

fprintf('Working directory: %s\n', PATH_code);
fprintf('Data directory: %s\n', PATH_exampledata);
fprintf('\n')

% Initialize Python environment and import modules
fprintf('Initializing Python environment... \n')
[~, ~] = py_init_mo();

%% Load data
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

%% Obs calculations
fprintf('\n=== Preparing datasets ===\n');

fprintf('Calculating observations...\n');
obs_cal_train = obs_calculation(data_train);
obs_cal_test = obs_calculation(data_test);

%% Data clean
fprintf('Cleaning datasets...\n');
[data_train, obs_cal_train, ~]  = data_clean(data_train, obs_cal_train);
[data_test, obs_cal_test, ~]  = data_clean(data_test, obs_cal_test);

%% Monin-Obukhov calculations
fprintf('Calculating M-O similarity...\n');
mo_cal_train = mo_calculation(data_train);
mo_cal_test = mo_calculation(data_test);

%% Input feature preparation
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

%% GMM Regime Identification
fprintf('\n=== Fitting Gaussian Mixture Model ===\n');
[GMModel, mu_gmm, sig_gmm] = gmm_identification(input_train);

% Assign samples to GMM clusters
Xtrain_gmm = (input_train - mu_gmm) ./ sig_gmm;
Xtest_gmm  = (input_test  - mu_gmm) ./ sig_gmm;

idx_GMM = cluster(GMModel, Xtrain_gmm);
jdx_GMM = cluster(GMModel, Xtest_gmm);

%% Train regime-aware model
fprintf('\n=== Training Regime-Aware Self-Attention Model ===\n');

% Train data
X = input_train;
Y = [target_train, mo_cal_train.(target)];  

% Test data
XT = input_test;
YT = [target_test, mo_cal_test.(target)];

Ypred = nan(size(YT, 1), 1);

fprintf("Train regime counts: "); fprintf("%6d ", histcounts(idx_GMM,1:GMModel.NumComponents+1)); fprintf("\n"); 
fprintf("Test  regime counts: "); fprintf("%6d ", histcounts(jdx_GMM,1:GMModel.NumComponents+1)); fprintf("\n"); 

for i_cluster = 1:GMModel.NumComponents
    fprintf('Regime %d...\n', i_cluster);
 
    if ~any(jdx_GMM == i_cluster)
        fprintf('Regime %d: no test samples, skip.\n', i_cluster);
        continue
    end
    if sum(idx_GMM == i_cluster) < cfg.min_train_per_regime
        fprintf('Regime %d: too few train samples, skip.\n', i_cluster);
        continue
    end

    % Select data for this regime
    XTrain = X(idx_GMM == i_cluster, :);
    YTrain = Y(idx_GMM == i_cluster, :);
    XTest = XT(jdx_GMM == i_cluster, :);

    % Train Physics-Informed Self-Attention model
    Ypred(jdx_GMM == i_cluster) = get_prediction_MLP_SA_PI(XTrain, YTrain, XTest, target=target);
end

%% Evaluate results
fprintf('\n=== Results ===\n');

% Filter valid predictions
valid_idx = ~isnan(Ypred) & ~isnan(YT(:,1));
Y_true = YT(valid_idx, 1);
Y_pred = Ypred(valid_idx);

% Calculate metrics
RMSE = sqrt(mean((Y_true - Y_pred).^2));
MAE = mean(abs(Y_true - Y_pred));
R = corr(Y_true, Y_pred, "Type", "Pearson");

fprintf('RMSE: %.4f \n', RMSE);
fprintf('MAE : %.4f \n', MAE);
fprintf('R   : %.4f \n', R);

% Plot results
if cfg.do_plot
    metrics.R = R;
    metrics.RMSE = RMSE;
    metrics.MAE = MAE;
    save_name = fullfile(PATH_code, '..', 'result', "example_result_" + target + ".png");
    plot_example_results(Y_true, Y_pred, metrics, save_plot=cfg.save_plot, save_name=save_name, target_var=target)
end

%% Elapsed time
t1 = toc(t0);
fprintf('Elapsed time: %.2f seconds \n', t1)
fprintf('\n=== Done ===\n');

% end