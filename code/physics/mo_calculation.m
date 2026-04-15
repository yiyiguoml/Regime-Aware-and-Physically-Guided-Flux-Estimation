function [mo_cal] = mo_calculation(data)

% Calculate MOST-based flux estimates using mo.py

% initialization
var_names = {'ustar',...
    'tstar',...
    'qstar',...
    'wthv0',...
    'wqv0',...
    'zeta_high',...
    'phi_m',...
    'phi_h'};
sz = [height(data),length(var_names)];
var_types = repmat({'double'},1,length(var_names));
mo_cal = table('Size', sz,...
    'VariableTypes', var_types,...
    'VariableNames', var_names);

% Surface roughness length used in the manuscript and retained here
% for the released example workflow
% Estimated from neutral wind profile regression over the Huainan forest site
z0 = 2.16; % [m]

% two levels
z_low = 10;
z_high = 35;

% --- extract vectors
t_low   = data.Ta_10M_Avg(:);
t_high  = data.Ta_35M_Avg(:);
rh_low  = data.RH_10M_Avg(:);
rh_high = data.RH_35M_Avg(:);

u_low   = data.U_10M(:);
v_low   = data.V_10M(:);
u_high  = data.U_35M(:);
v_high  = data.V_35M(:);

% --- calculated vectors 
t_low_k  = double(py.derived.celsius_to_kelvin(t_low));
t_high_k = double(py.derived.celsius_to_kelvin(t_high));

tmp     = py.derived.potential_temperature_from_height(t_low_k, z_low);
p_low   = double(tmp{2});
tmp     = py.derived.potential_temperature_from_height(t_high_k, z_high);
p_high  = double(tmp{2});

mr_low  = double(py.derived.mixing_ratio(t_low, rh_low, p_low));
mr_high = double(py.derived.mixing_ratio(t_high, rh_high, p_high));

% --- chunked call to mo
N = numel(t_low);
chunk = 2000;

k = 1;
while k <= N
    kk = k:min(N, k+chunk-1);

    out = double(py.mo.mo_similarity_two_levels_vec( ...
        u_low(kk), v_low(kk), u_high(kk), v_high(kk), ...
        t_low_k(kk), t_high_k(kk), p_high(kk), ...
        z_low, z_high, ...
        mr_low(kk), mr_high(kk), ...
        z0));

    % out should be [numel(kk) x 8] or [8 x numel(kk)] depending on python
    % adjust if needed:
    if size(out,2) ~= numel(var_names) && size(out,1) == numel(var_names)
        out = out.';  % make it (n x 8)
    end

    for j = 1:numel(var_names)
        mo_cal.(var_names{j})(kk) = out(:,j);
    end

    k = kk(end) + 1;
end

end

