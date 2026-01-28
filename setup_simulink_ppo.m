%% Setup Simulink-based PPO Training for Missile Autopilot
% This script sets up PPO training using the actual Simulink missile model
% PPO often works better than TD3 for this type of bounded action space
%
% Author: Anirudha Talmale
% MATLAB Version: R2022b with Reinforcement Learning Toolbox
%
% Prerequisites:
%   1. Data.mat with controller parameters
%   2. Hinf_PID_RL.slx (modified Simulink model with RL Agent block)

clear all; clc; close all;

%% Step 1: Load Controller Data
fprintf('Loading controller data...\n');
load('Data.mat');
fprintf('PID Parameters: Kp=%.4f, Ki=%.4f, Kd=%.4f\n', Kp, Ki, Kd);

%% Step 2: Define Observation and Action Specifications

% Observations: [error, error_derivative] (2 signals from Simulink)
obsInfo = rlNumericSpec([2 1], ...
    'LowerLimit', [-200; -2000], ...
    'UpperLimit', [200; 2000]);
obsInfo.Name = 'observations';
obsInfo.Description = 'error, error_derivative';

% Action: k in [0, 1] (blending factor)
actInfo = rlNumericSpec([1 1], ...
    'LowerLimit', 0, ...
    'UpperLimit', 1);
actInfo.Name = 'k';
actInfo.Description = 'blend factor (0=PID, 1=H-inf)';

%% Step 3: Create Simulink Environment
fprintf('Creating Simulink RL environment...\n');

mdl = 'Hinf_PID_RL';

if ~exist([mdl '.slx'], 'file')
    error(['Simulink model ' mdl '.slx not found.']);
end

env = rlSimulinkEnv(mdl, [mdl '/RL Agent'], obsInfo, actInfo);
env.ResetFcn = @(in) localResetFcn(in);

%% Step 4: Create PPO Agent Networks
fprintf('Creating PPO agent networks...\n');

numObs = obsInfo.Dimension(1);
numAct = actInfo.Dimension(1);

% Critic Network (Value Function)
criticNet = [
    featureInputLayer(numObs, 'Name', 'obs')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1, 'Name', 'value')
    ];
critic = rlValueFunction(dlnetwork(layerGraph(criticNet)), obsInfo, ...
    'ObservationInputNames', 'obs');

% Actor Network (Stochastic Policy)
% For PPO, we use a Gaussian policy with mean and standard deviation
actorNet = [
    featureInputLayer(numObs, 'Name', 'obs')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(numAct, 'Name', 'mean')
    ];

% Create stochastic actor with learnable standard deviation
actor = rlContinuousGaussianActor(dlnetwork(layerGraph(actorNet)), obsInfo, actInfo, ...
    'ObservationInputNames', 'obs', ...
    'ActionMeanOutputNames', 'mean');

%% Step 5: Configure PPO Agent
fprintf('Configuring PPO agent...\n');

agentOpts = rlPPOAgentOptions(...
    'SampleTime', 0.01, ...
    'DiscountFactor', 0.99, ...
    'ExperienceHorizon', 512, ...
    'MiniBatchSize', 64, ...
    'NumEpoch', 3, ...
    'ClipFactor', 0.2, ...
    'EntropyLossWeight', 0.01, ...  % Encourage exploration
    'GAEFactor', 0.95);

% Learning rates
agentOpts.ActorOptimizerOptions.LearnRate = 3e-4;
agentOpts.CriticOptimizerOptions.LearnRate = 3e-4;

% Create PPO agent
agent = rlPPOAgent(actor, critic, agentOpts);

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
    'StopTrainingValue', 1000);  % High value to train all episodes

trainOpts.SaveAgentCriteria = 'EpisodeReward';
trainOpts.SaveAgentValue = 100;
trainOpts.SaveAgentDirectory = 'savedAgents_ppo';

%% Step 7: Train the Agent
fprintf('Starting PPO training with Simulink model...\n');
fprintf('Training for up to %d episodes...\n', maxEpisodes);

trainingStats = train(agent, env, trainOpts);

%% Step 8: Save Results
save('trained_ppo_simulink.mat', 'agent', 'trainingStats');
fprintf('Training complete! Agent saved to trained_ppo_simulink.mat\n');

%% Helper Function
function in = localResetFcn(in)
    % Reset function - use default initial conditions
end
