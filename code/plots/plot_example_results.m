function [] = plot_example_results(Y_true, Y_pred, metrics, cfg)

arguments
    Y_true          (:,1) double
    Y_pred          (:,1) double
    metrics         (1,1) struct
    cfg.save_plot   (1,1) logical = false
    cfg.save_name   (1,1) string  = ""
    cfg.target_var  (1,1) string
end

target_var = cfg.target_var;
switch target_var
    case "ustar"
        var_unit = "m/s";
        var_range = [0, 1.5];
        var_delta = 0.01;
    case "tstar"
        var_unit = "K";
        var_range = [-1,0.5];
        var_delta = 0.01;
    case "qstar"
        var_unit = "g kg^-^1";
        var_range = [-0.2, 0.2];
        var_delta = 0.005;
    otherwise
        var_unit = "";
        var_range = [-Inf, Inf];
        var_delta = 0.01;
        warning("Unexpected target name of target_var")
end


figure('Position', [100, 100, 800, 600]);

subplot(2, 2, 1);
hold on;

xx = var_range(1):var_delta:var_range(2);
yy = xx;
[X,~] = meshgrid(xx,yy);
Z = zeros(size(X));
for k = 1:length(Y_true)
    i = find(Y_true(k)>=xx,1,"last");
    j = find(Y_pred(k)>=yy,1,"last");
    Z(i,j) = Z(i,j)+1;
end
Z = Z/sum(Z(:));

C = nan(numel(Y_true),1);
for k = 1:length(Y_true)
    try
        i = find(Y_true(k)>=xx,1,"last");
        j = find(Y_pred(k)>=yy,1,"last");
        C(k,:) = Z(i,j);
    catch
    end
end

scatter(Y_true, Y_pred, 10, C, 'filled', 'MarkerFaceAlpha', 0.35);
cmap = colormap("abyss");
colormap(flipud(cmap));
set(gca, 'ColorScale', 'log')
cb = colorbar;
cb.Location = 'eastoutside';
cb.Label.String = 'Sample frequency';

plot(var_range, var_range, 'LineStyle', '--', 'Color', [1,1,1]*0.3, 'LineWidth', 1.5)

lm = fitlm(Y_true, Y_pred, "Intercept",false);
coeff = lm.Coefficients.Estimate;
xx = [min(Y_true), max(Y_true)];
zz = coeff(1) * xx;
plot(xx, zz, 'Color', [0.85 0.33 0.10], 'LineWidth', 1.5);

set(gca, 'XLim', var_range, 'YLim', var_range)
set(gca, 'Box', 'on')

xlabel(sprintf('Observed %s (%s)', target_var, var_unit));
ylabel(sprintf('Predicted %s (%s)', target_var, var_unit));
title(sprintf('Predicted vs Observed (R = %.3f)', metrics.R));
grid on;

subplot(2, 2, 2);
histogram(Y_true - Y_pred, 50);
xlabel('Prediction Error (m/s)');
ylabel('Frequency');
title(sprintf('Error Distribution (RMSE = %.4f)', metrics.RMSE));
grid on;

subplot(2, 2, [3, 4]);
nplot = min(500, numel(Y_true));
plot(Y_true(1:nplot), 'b-', 'LineWidth', 1);
hold on;
plot(Y_pred(1:nplot), 'r-', 'LineWidth', 1);
xlabel('Sample index (first 500)');
ylabel(sprintf('%s (%s)', target_var, var_unit));
legend('Observed', 'Predicted');
grid on;

if cfg.save_plot
    outdir = fileparts(cfg.save_name);
    if strlength(outdir) > 0 && ~exist(outdir, "dir")
        mkdir(outdir);
    end
    saveas(gcf, cfg.save_name);
    fprintf('\nResults saved to: <root>/result/example_results.png\n');
end

end
