function reward = reward_function(error, error_dot, k, delta_k, nz_ref)
%REWARD_FUNCTION Calculates reward for RL agent in missile autopilot control
%
%   Inputs:
%       error     - Tracking error (nz_ref - nz)
%       error_dot - Derivative of tracking error
%       k         - Current weighting factor output by RL agent
%       delta_k   - Change in k from previous step
%       nz_ref    - Reference normal acceleration
%
%   Output:
%       reward    - Scalar reward value
%
%   The reward function is designed to:
%   1. Minimize tracking error (primary objective)
%   2. Penalize rapid changes in controller weighting (smooth transitions)
%   3. Penalize oscillatory behavior
%   4. Encourage faster settling

%% Weights for different reward components
w_error = 1.0;        % Weight for tracking error penalty
w_error_dot = 0.1;    % Weight for error derivative penalty
w_smooth = 0.05;      % Weight for smooth k transitions
w_settling = 0.2;     % Bonus for small steady-state error

%% Tracking Error Penalty
% Quadratic penalty on tracking error (normalized by reference)
if abs(nz_ref) > 1
    normalized_error = error / abs(nz_ref);
else
    normalized_error = error;
end
r_error = -w_error * normalized_error^2;

%% Error Derivative Penalty (reduce oscillations)
r_error_dot = -w_error_dot * (error_dot/100)^2;

%% Smooth Control Penalty (discourage rapid k changes)
r_smooth = -w_smooth * (delta_k * 10)^2;

%% Settling Bonus
% Give bonus when error is small (system has settled)
if abs(error) < 2  % Within 2 g of reference
    r_settling = w_settling * (1 - abs(error)/2);
else
    r_settling = 0;
end

%% Total Reward
reward = r_error + r_error_dot + r_smooth + r_settling;

% Clip reward to prevent extreme values
reward = max(min(reward, 1), -10);

end
