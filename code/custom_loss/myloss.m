function [loss] = myloss(Y, T, lambda_PI)

% ensure column-like vector
Y = Y(:);
% Case A: T is 2xB
if size(T,1) == 2
    Y_true = T(1,:); Y_true = Y_true(:);
    Y_MOST = T(2,:); Y_MOST = Y_MOST(:);
% Case B: T is Bx2
elseif size(T,2) == 2
    Y_true = T(:,1);
    Y_MOST = T(:,2);
else
    error("myloss_PI:BadTargetShape", "YTrain must have 2 columns (true, MOST).");
end

% Huber loss between prediction and truth
Lhub = get_Lhub(Y, Y_true);

% MOST regularization (MSE on valid pairs)
kdx = ~isnan(Y) & ~isnan(Y_MOST);
if any(kdx)
    Lpi = mean((Y(kdx) - Y_MOST(kdx)).^2, "all");
else
    Lpi = 0;
end

% final loss
loss = Lhub + lambda_PI * Lpi;

end

function [Lhub] = get_Lhub(Y, Y_true)
delta = 1.0; % Huber threshold in normalized target space
e = Y - Y_true;
absE = abs(e);
quad = 0.5 * (e.^2);
lin  = delta * (absE - 0.5*delta);
L =  (absE <= delta).*quad + (absE > delta).*lin;
Lhub = mean(L, "all");
end