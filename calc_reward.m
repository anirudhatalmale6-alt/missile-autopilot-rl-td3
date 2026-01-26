function reward = calc_reward(inputs)
%CALC_REWARD Calculate reward for RL agent training
%
%   Inputs (vector):
%       inputs(1) = error     - Tracking error (nz_ref - nz)
%       inputs(2) = error_dot - Derivative of error
%       inputs(3) = k         - Current weighting factor
%       inputs(4) = k_prev    - Previous weighting factor
%       inputs(5) = nz_ref    - Reference acceleration
%
%   Output:
%       reward - Scalar reward value for RL training

% Extract inputs with defaults
if length(inputs) >= 5
    error = inputs(1);
    error_dot = inputs(2);
    k = inputs(3);
    k_prev = inputs(4);
    nz_ref = inputs(5);
else
    error = 0;
    error_dot = 0;
    k = 0.5;
    k_prev = 0.5;
    nz_ref = 10;
end

delta_k = k - k_prev;

%% Reward Weights
w_error = 1.0;        % Primary: minimize tracking error
w_error_dot = 0.1;    % Secondary: reduce oscillations
w_smooth = 0.05;      % Tertiary: smooth controller transitions
w_settling = 0.2;     % Bonus for achieving small error

%% Component 1: Tracking Error Penalty (quadratic)
% Normalize by reference to handle different setpoint magnitudes
if abs(nz_ref) > 1
    norm_error = error / abs(nz_ref);
else
    norm_error = error;
end
r_error = -w_error * norm_error^2;

%% Component 2: Error Derivative Penalty (reduce oscillations)
% Scale error_dot appropriately
r_error_dot = -w_error_dot * (error_dot / 100)^2;

%% Component 3: Smooth Control Transitions
% Penalize rapid changes in k
r_smooth = -w_smooth * (delta_k * 10)^2;

%% Component 4: Settling Bonus
% Reward for achieving small steady-state error
if abs(error) < 2  % Within 2g of reference
    r_settling = w_settling * (1 - abs(error) / 2);
else
    r_settling = 0;
end

%% Total Reward
reward = r_error + r_error_dot + r_smooth + r_settling;

% Clip to prevent extreme values
reward = max(min(reward, 1), -10);

end
