function [data_clean, obs_cal_clean, ind_keep]  = data_clean(data, obs_cal, cfg)
% DATA_CLEAN  Quality control for flux-tower dataset.
% Removes records with missing variables, incomplete target fluxes, 
% implausible u* values, numerically ill-conditioned Obukhov length, 
% and near-saturated strongly non-neutral conditions (proxy contamination).

arguments
    data           (:,:) table
    obs_cal        (:,:) table

    cfg.min_ustar  (1,1) double = 0.01
    cfg.max_ustar  (1,1) double = 1.5
    cfg.L_minabs   (1,1) double = 0.1

    cfg.z_high     (1,1) double = 35
    cfg.rh_var     (1,1) string = "RH_35M_Avg"
    cfg.rh_th      (1,1) double = 95
    cfg.zL_th      (1,1) double = 2
end


% (1) Any missing values in observations
ind_data_nan = any(isnan(data.Variables), 2);

% (2) Missing targets
ind_flux_nan = isnan(data.Tau_30) | isnan(data.Hs_30) | isnan(data.LE_30);

% (3) Unrealistic / weak-turbulence friction velocity
% 0.05 m/s: typical lower bound for reliable turbulent flux estimation
% 1.5 m/s: upper bound to exclude spurious spikes
ind_ustar = obs_cal.friction_velocity_m_s > cfg.max_ustar | ...
            obs_cal.friction_velocity_m_s < cfg.min_ustar;

% (4) Numerically ill-conditioned Obukhov length (near-zero |L|)
ind_L_small = abs(obs_cal.obukhov_length) < cfg.L_minabs;

% (5) Near-saturated & strongly non-neutral (proxy for possible precipitation/condensation contamination)
z_high = cfg.z_high;
ind_rh = data.(cfg.rh_var) > cfg.rh_th & abs(z_high ./ obs_cal.obukhov_length) > cfg.zL_th;

% Final keep mask
ind_keep = ~(ind_data_nan | ind_flux_nan | ind_ustar | ind_L_small | ind_rh);

% apply ind_keep to data/obs_cal
data_clean = data(ind_keep, :);
obs_cal_clean = obs_cal(ind_keep, :);

end