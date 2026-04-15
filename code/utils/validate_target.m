function target = validate_target(target)

    validTargets = ["ustar", "tstar", "qstar"];

    if ~isstring(target)
        target = string(target);
    end

    if ~ismember(target, validTargets)
        error('Invalid target: %s. Must be one of: %s', ...
            target, strjoin(validTargets, ', '));
    end
end