function lambda_PI = get_lambda_PI(target)

% Default target-specific lambda_PI values.
% These values are calibrated on the example dataset provided in this repository
% to reproduce the example results.
%
% Note:
% The optimal lambda_PI is target-dependent and may vary with data distribution.
% Users may need to adjust these values for different datasets.

target = validate_target(target);

switch target
    case "ustar"
        lambda_PI = 0.01;
    case "tstar"
        lambda_PI = 0.1;
    case "qstar"
        lambda_PI = 0.05;
end

end