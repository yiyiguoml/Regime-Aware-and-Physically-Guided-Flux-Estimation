function [Ypred, aux] = run_regime_model(input_train, target_train, mo_cal_train, input_test, cfg)

arguments
    input_train              (:,:) double
    target_train             (:,1) double
    mo_cal_train             (:,:) table
    input_test               (:,:) double

    cfg.target               (1,1) string
    cfg.n_components         (1,1) double {mustBeInteger, mustBePositive} = 3
    cfg.min_train_per_regime (1,1) double  = 100
    cfg.lambda_PI_override   (1,1) double  = NaN
    cfg.verbose              (1,1) logical = false
end

target = validate_target(cfg.target);
n_components = cfg.n_components;

%% GMM Regime Identification
[GMModel, mu_gmm, sig_gmm] = gmm_identification(input_train, n_components=n_components);
Xtrain_gmm = (input_train - mu_gmm) ./ sig_gmm;
Xtest_gmm  = (input_test  - mu_gmm) ./ sig_gmm;
idx_GMM = cluster(GMModel, Xtrain_gmm);
jdx_GMM = cluster(GMModel, Xtest_gmm);

%% Train regime-aware model

% Train data
X = input_train;
Y = [target_train, mo_cal_train.(target)];

% Test data
XT = input_test;

Ypred = nan(size(input_test, 1), 1);

if cfg.verbose
    fprintf("Train regime counts: "); fprintf("%6d ", histcounts(idx_GMM,1:n_components+1)); fprintf("\n");
    fprintf("Test  regime counts: "); fprintf("%6d ", histcounts(jdx_GMM,1:n_components+1)); fprintf("\n");
end

for i_cluster = 1:n_components
    if cfg.verbose
        fprintf('Regime %d...\n', i_cluster);
    end

    if ~any(jdx_GMM == i_cluster)
        if cfg.verbose
            fprintf('Regime %d: no test samples, skip.\n', i_cluster);
        end
        continue
    end
    if sum(idx_GMM == i_cluster) < cfg.min_train_per_regime
        if cfg.verbose
            fprintf('Regime %d: too few train samples, skip.\n', i_cluster);
        end
        continue
    end

    % Select data for this regime
    XTrain = X(idx_GMM == i_cluster, :);
    YTrain = Y(idx_GMM == i_cluster, :);
    XTest = XT(jdx_GMM == i_cluster, :);

    % Train Physics-Informed Self-Attention model
    Ypred(jdx_GMM == i_cluster) = get_prediction_MLP_SA_PI(...
        XTrain, YTrain, XTest, ...
        target=target, ...
        lambda_PI_override=cfg.lambda_PI_override);

end

%% aux
aux.idx_GMM = idx_GMM;
aux.jdx_GMM = jdx_GMM;
aux.n_components = n_components;
aux.lambda_PI_override = cfg.lambda_PI_override;
aux.train_counts = histcounts(idx_GMM, 1:n_components+1);
aux.test_counts  = histcounts(jdx_GMM, 1:n_components+1);

if ~isempty(GMModel)
    aux.GMModel = GMModel;
else
    aux.GMModel = [];
end

end