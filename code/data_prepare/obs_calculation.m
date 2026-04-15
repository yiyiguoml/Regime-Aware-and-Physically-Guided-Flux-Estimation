function [obs_cal] = obs_calculation(data)

% Calculate observation-derived diagnostic quantities used for training,
% quality control, and regime identification.

%% initialization 
var_names = {'friction_velocity_m_s',...
    'kinematic_sensible_heat_flux_K_s',...
    'temperature_scale_k',...
    'kinematic_latent_heat_flux_g_kg_m_s',...
    'moisture_scale_g_kg',...
    'skin_virtual_potential_temperature_k',...
    'skin_water_vapor_mixing_ratio',...
    'bulk_richardson_number',...
    'obukhov_length'};
sz = [height(data),length(var_names)];
var_types = repmat({'double'},1,length(var_names));
obs_cal = table('Size', sz,...
    'VariableTypes', var_types,...
    'VariableNames', var_names);

%% calculation
% Measurement heights used in the released example dataset
z_atm = 35;
z_skin = 10;

wind_speed          = data.WS_35M(:);
temperature_c       = data.Ta_35M_Avg(:);
relative_humidity   = data.RH_35M_Avg(:);

temperature_k = double(py.derived.celsius_to_kelvin(temperature_c));
tmp = py.derived.potential_temperature_from_height(temperature_k, z_atm);
potential_temperature_k = double(tmp{1});
pressure_hpa = double(tmp{2});
mixing_ratio_g_kg = double(py.derived.mixing_ratio(temperature_c, relative_humidity, pressure_hpa));
virtual_temperature_k = double(py.derived.virtual_temperature(temperature_k, mixing_ratio_g_kg));
air_density_kg_m3 = double(py.derived.air_density(virtual_temperature_k, pressure_hpa));

% z_skin level
temperature_skin_c      = data.Ta_10M_Avg(:);
relative_humidity_skin  = data.RH_10M_Avg(:);

temperature_skin_k = double(py.derived.celsius_to_kelvin(temperature_skin_c));
tmp = py.derived.potential_temperature_from_height(temperature_k, z_atm);
pressure_skin_hpa = double(tmp{2});
mixing_ratio_skin_g_kg = double(py.derived.mixing_ratio(temperature_skin_c, relative_humidity_skin, pressure_skin_hpa));
virtual_potential_skin_temperature_k = double(py.derived.virtual_temperature(temperature_skin_k, mixing_ratio_skin_g_kg));

% ustar
surface_stress_kg_m_s2 = data.Tau_30(:);
friction_velocity_m_s = double(py.derived.friction_velocity(surface_stress_kg_m_s2, air_density_kg_m3));

% thstar
sensible_heat_flux_W_m2 = data.Hs_30(:);
kinematic_sensible_heat_flux_K_s = double(py.derived.kinematic_sensible_heat_flux(sensible_heat_flux_W_m2, air_density_kg_m3));
temperature_scale_k = double(py.derived.temperature_scale(sensible_heat_flux_W_m2, air_density_kg_m3, friction_velocity_m_s));

% qstar
latent_heat_flux_W_m2 = data.LE_30(:);
kinematic_latent_heat_flux_g_kg_m_s = double(py.derived.kinematic_latent_heat_flux(latent_heat_flux_W_m2, air_density_kg_m3));
moisture_scale_g_kg = double(py.derived.moisture_scale(latent_heat_flux_W_m2, air_density_kg_m3, friction_velocity_m_s));

% Rib
bulk_richardson_number = double(py.derived.bulk_richardson_number(potential_temperature_k, (z_atm - z_skin),...
    mixing_ratio_g_kg, virtual_potential_skin_temperature_k, wind_speed,...
    minimum_wind_speed=0.1));

% Obukhov Length
olength = double(py.derived.obukhov_length(potential_temperature_k, temperature_scale_k, friction_velocity_m_s, von_karman_constant=0.4,...
    min_friction_velocity=0.01, min_temperature_scale=0.01));

% copy to data_obs [table]
obs_cal.('friction_velocity_m_s')                = friction_velocity_m_s;
obs_cal.('kinematic_sensible_heat_flux_K_s')     = kinematic_sensible_heat_flux_K_s;
obs_cal.('temperature_scale_k')                  = temperature_scale_k;
obs_cal.('kinematic_latent_heat_flux_g_kg_m_s')  = kinematic_latent_heat_flux_g_kg_m_s;
obs_cal.('moisture_scale_g_kg')                  = moisture_scale_g_kg;
obs_cal.('skin_virtual_potential_temperature_k') = virtual_potential_skin_temperature_k;
obs_cal.('skin_water_vapor_mixing_ratio')        = mixing_ratio_skin_g_kg;
obs_cal.('bulk_richardson_number')               = bulk_richardson_number;
obs_cal.('obukhov_length')                       = olength;

end
