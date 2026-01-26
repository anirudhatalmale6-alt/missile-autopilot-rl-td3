%% Test Trained TD3 Agent for Missile Autopilot
% This script loads the trained agent and runs simulation tests
% Author: Anirudha Talmale

clear; clc; close all;

%% Load Data and Trained Agent
load('Data.mat');
load('trained_td3_agent.mat', 'agent');

%% Open Simulink Model
mdl = 'Hinf_PID_RL';
open_system(mdl);

%% Configure for Simulation (not training)
% Set simulation time
simTime = 12;  % seconds

%% Run Simulation with Trained Agent
disp('Running simulation with trained TD3 agent...');
simOptions = rlSimulationOptions('MaxSteps', simTime/0.01);
experience = sim(env, agent, simOptions);

%% Extract Results
time = experience.Observation.observations.Time;
observations = squeeze(experience.Observation.observations.Data);
actions = squeeze(experience.Action.k.Data);

error = observations(1,:);
error_dot = observations(2,:);
nz = observations(3,:);

%% Load workspace data from simulation
% The ToWorkspace blocks save nz and alpha

%% Plot Results
figure('Name', 'TD3 Controller Performance', 'Position', [100 100 1200 800]);

% Plot 1: Normal Acceleration Tracking
subplot(2,2,1);
plot(time, nz, 'b-', 'LineWidth', 1.5);
hold on;
% Plot reference (from repeating sequence)
ref_values = [50 -40 -15 15 -1 40];
ref_times = 0:2:10;
for i = 1:length(ref_times)
    if i < length(ref_times)
        t_start = ref_times(i);
        t_end = ref_times(i+1);
    else
        t_start = ref_times(i);
        t_end = simTime;
    end
    idx = (time >= t_start) & (time < t_end);
    plot(time(idx), ref_values(i)*ones(sum(idx),1), 'r--', 'LineWidth', 1.5);
end
hold off;
xlabel('Time (s)');
ylabel('Normal Acceleration (g)');
title('Normal Acceleration Tracking');
legend('Actual nz', 'Reference', 'Location', 'best');
grid on;

% Plot 2: Tracking Error
subplot(2,2,2);
plot(time, error, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Error (g)');
title('Tracking Error (nz_{ref} - nz)');
grid on;

% Plot 3: Weighting Factor k
subplot(2,2,3);
plot(time, actions, 'g-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('k');
title('RL Agent Output: Weighting Factor k');
ylim([-0.1 1.1]);
grid on;

% Plot 4: Controller Blend Analysis
subplot(2,2,4);
bar([mean(actions) mean(1-actions)]);
set(gca, 'XTickLabel', {'H-inf', 'PID'});
ylabel('Average Weight');
title('Average Controller Contribution');
grid on;

%% Performance Metrics
disp('=== Performance Metrics ===');
disp(['Mean Absolute Error: ', num2str(mean(abs(error)), '%.4f'), ' g']);
disp(['Max Absolute Error: ', num2str(max(abs(error)), '%.4f'), ' g']);
disp(['RMS Error: ', num2str(rms(error), '%.4f'), ' g']);
disp(['Average k (H-inf weight): ', num2str(mean(actions), '%.4f')]);

%% Calculate ITAE (Integral Time Absolute Error)
dt = time(2) - time(1);
ITAE = sum(time .* abs(error')) * dt;
disp(['ITAE: ', num2str(ITAE, '%.4f')]);

%% Save Results
save('simulation_results.mat', 'time', 'nz', 'error', 'actions');
disp('Results saved to simulation_results.mat');

%% Compare with Fuzzy Logic (Optional)
% If you have results from the fuzzy logic controller, load and compare:
% load('fuzzy_results.mat');
% figure;
% plot(time, error_rl, 'b-', time, error_fuzzy, 'r--');
% legend('RL Controller', 'Fuzzy Logic');
