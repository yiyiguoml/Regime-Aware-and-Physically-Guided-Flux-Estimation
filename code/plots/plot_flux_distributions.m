function hfig = plot_flux_distributions(data_train, data_test, cfg)
% PLOT_FLUX_DISTRIBUTIONS
% Plot the distributions of u*, t*, and q* for the released subset.
%
% This function is optional and is not part of the main workflow.
% It can be used to visualize the statistical characteristics of the
% released example dataset.
%
% Inputs:
%   data_train : training subset table
%   data_test  : test subset table
%   cfg        : optional settings
%
% Required variables in obs_subset:
%   - friction_velocity_m_s
%   - temperature_scale_k
%   - moisture_scale_g_kg
%
% Optional cfg fields:
%   cfg.save_plot   (logical, default = false)
%   cfg.save_name   (string,  default = "subset_flux_distributions.png")
%   cfg.dataset_name(string,  default = "Released subset")
%
% Output:
%   hfig : figure handle

arguments
    data_train (:,:) table
    data_test  (:,:) table
    cfg.save_plot    (1,1) logical = false
    cfg.save_name    (1,1) string  = "subset_flux_distributions.png"
    cfg.dataset_name (1,1) string  = "Released subset"
end


% Obs calculations
obs_cal_train = obs_calculation(data_train);
obs_cal_test = obs_calculation(data_test);
obs_cal = [obs_cal_train; obs_cal_test];

% Extract variables and remove NaNs
u_sub = obs_cal.friction_velocity_m_s;
t_sub = obs_cal.temperature_scale_k;
q_sub = obs_cal.moisture_scale_g_kg;

u_sub = u_sub(~isnan(u_sub));
t_sub = t_sub(~isnan(t_sub));
q_sub = q_sub(~isnan(q_sub));

% Bin settings
edges_u = 0:0.02:3.0;
edges_t = -1.5:0.02:1.0;
edges_q = -1.0:0.005:1.0;

xlim_u = [-0.05, 1.5];
xlim_t = [-1.0, 0.5];
xlim_q = [-0.2, 0.2];

% Figure
hfig = figure('Color','w', 'Position',[100, 100, 1200, 380]);
tiledlayout(1,3,"TileSpacing","tight","Padding","compact");

nexttile(1);
plot_one_subset_variable(u_sub, edges_u, xlim_u, 'u_* [m s^-^1]', cfg.dataset_name);

nexttile(2);
plot_one_subset_variable(t_sub, edges_t, xlim_t, '\theta_* [K]', cfg.dataset_name);

nexttile(3);
plot_one_subset_variable(q_sub, edges_q, xlim_q, 'q_* [g kg^-^1]', cfg.dataset_name);

if cfg.save_plot
    out_dir = fileparts(cfg.save_name);
    if ~isempty(out_dir) && ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    exportgraphics(hfig, cfg.save_name, 'Resolution', 300);
end

end


function plot_one_subset_variable(x_sub, edges, xlim_range, x_label_text, dataset_name)

hold on

% Frequency (%)
N_sub = histcounts(x_sub, edges);
freq_sub = N_sub / numel(x_sub) * 100;

yyaxis left
histogram('BinEdges', edges, 'BinCounts', freq_sub, ...
    'FaceAlpha', 0.55, ...
    'EdgeColor', 'none', ...
    'FaceColor', [0.93, 0.69, 0.13]);

ylabel('Frequency (%)')
set(gca, 'YColor', 'k')

% ECDF
yyaxis right
[f, x] = ecdf(x_sub);
plot(x, f, 'LineWidth', 2.0, 'Color', [0.85, 0.33, 0.10]);

ylabel('Cumulative distribution function')
set(gca, 'YColor', 'k')

xlim(xlim_range)
xlabel(x_label_text)
grid on
box on
set(gca, 'FontWeight', 'bold', 'FontSize', 12)

legend( ...
    sprintf('%s histogram', dataset_name), ...
    sprintf('%s ECDF', dataset_name), ...
    'Location', 'best', ...
    'Box', 'off');

end