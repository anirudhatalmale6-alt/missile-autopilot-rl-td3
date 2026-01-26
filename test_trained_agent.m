%% Test Trained TD3 Agent for Missile Autopilot
% This script loads the trained agent and runs simulation tests
% Author: Anirudha Talmale

clear all; clc; close all;

%% Load Data and Trained Agent
fprintf('Loading trained agent...\n');
load('Data.mat');
load('trained_td3_agent.mat', 'agent');

%% Create Environment for Testing
env = MissileEnv();

%% Run Simulation with Trained Agent
fprintf('Running simulation with trained TD3 agent...\n');

simOpts = rlSimulationOptions('MaxSteps', 1200);  % 12 seconds
experience = sim(env, agent, simOpts);

%% Extract Results from Experience
% Get observation data
obsData = experience.Observation.observations.Data;
numSteps = size(obsData, 3);
time = (0:numSteps-1) * 0.01;  % Time vector

% Extract signals (observations are [error; errorDot; nz])
error_signal = squeeze(obsData(1,1,:));
error_dot = squeeze(obsData(2,1,:));
nz_actual = squeeze(obsData(3,1,:));

% Get action data
actData = experience.Action.k.Data;
k_values = squeeze(actData);

% Handle dimension mismatch - action is taken before observation
% So action has one less sample or needs alignment
if length(k_values) ~= length(time)
    % Pad k_values to match time
    if length(k_values) < length(time)
        k_values = [k_values; k_values(end) * ones(length(time) - length(k_values), 1)];
    else
        k_values = k_values(1:length(time));
    end
end

% Ensure all are column vectors
time = time(:);
error_signal = error_signal(:);
error_dot = error_dot(:);
nz_actual = nz_actual(:);
k_values = k_values(:);

% Reconstruct reference signal
ref_values = [50, -40, -15, 15, -1, 40];
nz_ref = zeros(size(time));
for i = 1:length(time)
    refIdx = mod(floor(time(i) / 2), 6) + 1;
    nz_ref(i) = ref_values(refIdx);
end

%% Plot Results
figure('Name', 'TD3 Controller Performance', 'Position', [100 100 1200 800]);

% Plot 1: Normal Acceleration Tracking
subplot(2,2,1);
plot(time, nz_ref, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Reference');
hold on;
plot(time, nz_actual, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Actual nz');
hold off;
xlabel('Time (s)');
ylabel('Normal Acceleration (g)');
title('Normal Acceleration Tracking');
legend('Location', 'best');
grid on;

% Plot 2: Tracking Error
subplot(2,2,2);
plot(time, error_signal, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Error (g)');
title('Tracking Error (nz_{ref} - nz)');
grid on;

% Plot 3: Weighting Factor k
subplot(2,2,3);
plot(time, k_values, 'g-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('k');
title('RL Agent Output: Weighting Factor k');
ylim([-0.1 1.1]);
grid on;

% Plot 4: Controller Blend Analysis
subplot(2,2,4);
avg_k = mean(k_values);
bar([avg_k, 1-avg_k]);
set(gca, 'XTickLabel', {'H-inf Weight', 'PID Weight'});
ylabel('Average Weight');
title(sprintf('Average Controller Contribution (k=%.3f)', avg_k));
grid on;

%% Performance Metrics
fprintf('\n=== Performance Metrics ===\n');
fprintf('Mean Absolute Error: %.4f g\n', mean(abs(error_signal)));
fprintf('Max Absolute Error: %.4f g\n', max(abs(error_signal)));
fprintf('RMS Error: %.4f g\n', rms(error_signal));
fprintf('Average k (H-inf weight): %.4f\n', avg_k);

% Calculate ITAE
dt = 0.01;
ITAE = sum(time .* abs(error_signal)) * dt;
fprintf('ITAE: %.4f\n', ITAE);

%% Save figure
saveas(gcf, 'td3_controller_results.png');
fprintf('\nResults saved to td3_controller_results.png\n');

%% Additional Analysis - Step Response Details
figure('Name', 'Step Response Analysis', 'Position', [150 150 800 600]);

% Find settling times for each step
subplot(2,1,1);
plot(time, nz_ref, 'r--', 'LineWidth', 1.5);
hold on;
plot(time, nz_actual, 'b-', 'LineWidth', 1.5);
hold off;
xlabel('Time (s)');
ylabel('nz (g)');
title('Step Response Tracking');
legend('Reference', 'Actual', 'Location', 'best');
grid on;

subplot(2,1,2);
plot(time, k_values, 'g-', 'LineWidth', 1.5);
hold on;
% Add reference change markers
for i = 1:5
    xline(i*2, '--k', 'Alpha', 0.5);
end
hold off;
xlabel('Time (s)');
ylabel('k');
title('Controller Blending Over Time');
grid on;

saveas(gcf, 'td3_step_analysis.png');
fprintf('Step analysis saved to td3_step_analysis.png\n');
