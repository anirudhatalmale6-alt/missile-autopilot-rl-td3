%% Setup Simulink-based DQN Training for Missile Autopilot
% This script sets up DQN training for discrete switching between
% H-infinity (k=1) and PID (k=0) controllers
%
% Based on the paper's fuzzy logic approach:
% - When |nz_ref| is large -> use H-inf (k=1)
% - When |nz_ref| is small AND |error| is small -> use PID (k=0)
%
% Author: Anirudha Talmale
% MATLAB Version: R2022b with Reinforcement Learning Toolbox
%
% Prerequisites:
%   1. Data.mat with controller parameters
%   2. Hinf_PID_RL_discrete.slx (Simulink model with discrete RL Agent)
%
% IMPORTANT: You need to modify the Simulink model to:
%   1. Add |nz_ref| as a third observation input
%   2. The RL Agent will output 0 or 1 (discrete)

clear all; clc; close all;

%% Step 1: Load Controller Data
fprintf('Loading controller data...\n');
load('Data.mat');
fprintf('PID Parameters: Kp=%.4f, Ki=%.4f, Kd=%.4f\n', Kp, Ki, Kd);

%% Step 2: Define Observation and Action Specifications

% Observations: [error, error_derivative, |nz_ref|] (3 signals)
% error = nz_ref - nz
% error_derivative = d(error)/dt
% |nz_ref| = absolute value of reference (to determine if "large" or "small")
obsInfo = rlNumericSpec([3 1], ...
    'LowerLimit', [-200; -2000; 0], ...
    'UpperLimit', [200; 2000; 100]);
obsInfo.Name = 'observations';
obsInfo.Description = 'error, error_derivative, abs_nz_ref';

% Action: DISCRETE - either 0 (use PID) or 1 (use H-inf)
actInfo = rlFiniteSetSpec([0 1]);
actInfo.Name = 'k';
actInfo.Description = 'controller select: 0=PID, 1=H-inf';

%% Step 3: Create Simulink Environment
fprintf('Creating Simulink RL environment...\n');

mdl = 'Hinf_PID_RL';

if ~exist([mdl '.slx'], 'file')
    error(['Simulink model ' mdl '.slx not found.']);
end

env = rlSimulinkEnv(mdl, [mdl '/RL Agent'], obsInfo, actInfo);
env.ResetFcn = @(in) localResetFcn(in);

%% Step 4: Create DQN Agent Network
fprintf('Creating DQN agent network...\n');

numObs = obsInfo.Dimension(1);
numAct = numel(actInfo.Elements);  % 2 actions: 0 and 1

% Q-Network: outputs Q-value for each action
qNetwork = [
    featureInputLayer(numObs, 'Name', 'obs')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(numAct, 'Name', 'qvalues')
    ];

qNet = dlnetwork(layerGraph(qNetwork));
critic = rlVectorQValueFunction(qNet, obsInfo, actInfo, ...
    'ObservationInputNames', 'obs');

%% Step 5: Configure DQN Agent
fprintf('Configuring DQN agent...\n');

agentOpts = rlDQNAgentOptions(...
    'SampleTime', 0.01, ...
    'DiscountFactor', 0.99, ...
    'TargetSmoothFactor', 0.001, ...
    'ExperienceBufferLength', 1e5, ...
    'MiniBatchSize', 64, ...
    'UseDoubleDQN', true);

% Epsilon-greedy exploration
agentOpts.EpsilonGreedyExploration.Epsilon = 1.0;
agentOpts.EpsilonGreedyExploration.EpsilonMin = 0.05;
agentOpts.EpsilonGreedyExploration.EpsilonDecay = 0.005;

% Learning rate
agentOpts.CriticOptimizerOptions.LearnRate = 1e-3;

% Create DQN agent
agent = rlDQNAgent(critic, agentOpts);

%% Step 6: Training Options
maxEpisodes = 200;
maxSteps = 1200;  % 12 seconds at 0.01s sample time

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', maxEpisodes, ...
    'MaxStepsPerEpisode', maxSteps, ...
    'ScoreAveragingWindowLength', 20, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 1000);

trainOpts.SaveAgentCriteria = 'EpisodeReward';
trainOpts.SaveAgentValue = 100;
trainOpts.SaveAgentDirectory = 'savedAgents_dqn';

%% Step 7: Train the Agent
fprintf('Starting DQN training with Simulink model...\n');
fprintf('Training for up to %d episodes...\n', maxEpisodes);
fprintf('\nThe agent will learn:\n');
fprintf('  - k=1 (H-inf) when |nz_ref| is large\n');
fprintf('  - k=0 (PID) when |nz_ref| is small and |error| is small\n\n');

trainingStats = train(agent, env, trainOpts);

%% Step 8: Save Results
save('trained_dqn_simulink.mat', 'agent', 'trainingStats');
fprintf('Training complete! Agent saved to trained_dqn_simulink.mat\n');

%% Helper Function
function in = localResetFcn(in)
    % Reset function - use default initial conditions
end
