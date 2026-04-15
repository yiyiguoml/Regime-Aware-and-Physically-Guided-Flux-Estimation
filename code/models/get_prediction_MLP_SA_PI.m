function [Ypred, net, normStat] = get_prediction_MLP_SA_PI(XTrain, YTrain, XTest, cfg)

% GET_PREDICTION_MLP_SA_PI
% Physics-informed Self-Attention MLP for flux estimation.
%
% Inputs:
%   XTrain : (Ntr x P) double
%   YTrain : (Ntr x 2) double, columns = [y_true, y_MOST]
%   XTest  : (Nte x P) double
%   cfg (optional): struct with fields
%       .numHeads (default 4)
%       .attnDim  (default 64)
%       .maxEpochs (default 200)
%       .learnRate (default 0.005)
%       .outlierPct (default [0.5 99.5])   % on raw y_true
%       .minSigma   (default 1e-8)
%       .lambda_PI_override (default NaN)
%
% Outputs:
%   Ypred : (Nte_kept x 1) double, de-normalized prediction
%   net   : trained dlnetwork-like model returned by trainnet
%   normStat: struct storing normalization stats for reproducibility

arguments
    XTrain                  (:,:) double
    YTrain                  (:,2) double
    XTest                   (:,:) double
    cfg.target              (1,1) string
    cfg.numHeads            (1,1) double = 4
    cfg.attnDim             (1,1) double = 64
    cfg.maxEpochs           (1,1) double = 200
    cfg.learnRate           (1,1) double = 0.005
    cfg.outlierPct          (1,2) double = [0.5 99.5]
    cfg.minSigma            (1,1) double = 1e-8
    cfg.lambda_PI_override  (1,1) double = NaN
end

target = validate_target(cfg.target);

% --- 0) data
XTrain(isinf(XTrain)) = nan;
XTest(isinf(XTest))   = nan;

% --- 1) Normalize X using TRAIN only
muX = mean(XTrain, "omitnan");
sigX = std(XTrain, "omitnan");
sigX(sigX < cfg.minSigma) = 1; % avoid division by ~0

XTrain_norm = (XTrain - muX)./sigX;
XTest_norm  = (XTest - muX)./sigX;

% --- 2) Normalize Y using TRAIN y_true stats
YTrain_true = YTrain(:,1);
muY = mean(YTrain_true, "omitnan");
sigY = std(YTrain_true, "omitnan");
sigY(sigY < cfg.minSigma) = 1;

YTrain_norm = (YTrain - muY)./sigY;

% --- 3) Remove NaNs and extreme outliers 
idxKeepTrain = ~any(isnan(XTrain_norm), 2) & ~any(isnan(YTrain_norm), 2);
idxOut = isoutlier(YTrain_true, "percentile", cfg.outlierPct); % only remove extreme outliers, but keep MOST consistency
idxKeepTrain = idxKeepTrain & ~idxOut;

XTrain_norm = XTrain_norm(idxKeepTrain, :);
YTrain_norm = YTrain_norm(idxKeepTrain, :);

idxKeepTest = ~any(isnan(XTest_norm), 2);
XTest_norm = XTest_norm(idxKeepTest, :);

% --- 4) Build network
numFeatures = size(XTrain_norm, 2);

layers = [
    featureInputLayer(numFeatures, Name="in")          
    selfAttentionLayer(cfg.numHeads, cfg.attnDim, Name="self_attention") 
    
    fullyConnectedLayer(128, Name="fc1")  
    reluLayer(Name="relu1")

    fullyConnectedLayer(64, Name="fc2")
    reluLayer(Name="relu2")

    fullyConnectedLayer(1, Name="out") 
    ];

% --- 5) Training options
opts = trainingOptions("adam", ...
    MaxEpochs = cfg.maxEpochs, ...
    InitialLearnRate = cfg.learnRate, ...
    Plots = "none", ...
    Verbose = false);

% PI loss
if isnan(cfg.lambda_PI_override)
    lambda_PI = get_lambda_PI(target);
else
    lambda_PI = cfg.lambda_PI_override;
end
lossFcn = @(Y,T) myloss(Y, T, lambda_PI);

% --- 6) Train
net = trainnet(XTrain_norm, YTrain_norm, layers, lossFcn, opts);

% --- 7) Predict and de-normalize
Ypred_norm = predict(net, XTest_norm);
Ypred = Ypred_norm * sigY + muY;

% --- 8) Return normalization stats for reproducibility
normStat.muX = muX;
normStat.sigX = sigX;
normStat.muY = muY;
normStat.sigY = sigY;
normStat.cfg = cfg;
normStat.lambda_PI = lambda_PI;

end
