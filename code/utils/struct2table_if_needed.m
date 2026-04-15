function tbl = struct2table_if_needed(data)
% STRUCT2TABLE_IF_NEEDED Convert struct to table if needed
%
% This helper function handles data that may be loaded as either:
% - A MATLAB table (from native MATLAB .mat files)
% - A struct (from Python scipy.io.savemat files)
%
% Usage:
%   tbl = struct2table_if_needed(data)
%
% Input:
%   data - Either a table or struct with field names as columns
%
% Output:
%   tbl - A MATLAB table

if istable(data)
    % Already a table, return as-is
    tbl = data;
elseif isstruct(data)
    % Convert struct to table
    tbl = struct2table(data);

    % Handle date column if it exists (convert from datenum to datetime)
    if ismember('date', tbl.Properties.VariableNames)
        if isnumeric(tbl.date)
            % Convert MATLAB datenum to datetime
            tbl.date = datetime(tbl.date, 'ConvertFrom', 'datenum');
        end
    end
else
    error('Unsupported data type: %s', class(data));
end

end
