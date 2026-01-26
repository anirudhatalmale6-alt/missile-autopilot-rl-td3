%% Complete Setup and Training Script for TD3 RL-based Missile Autopilot
% This script sets up and trains a TD3 agent to replace fuzzy logic
% for combining H-infinity and PID controllers
%
% Author: Anirudha Talmale
% MATLAB Version: R2022b with Reinforcement Learning Toolbox
%
% Usage:
%   1. Place all files in the same folder as your Data.mat
%   2. Run this script
%   3. Training will start automatically
%   4. After training, use test_trained_agent.m to evaluate

clear all; clc; close all;

%% Step 1: Load Controller Data
fprintf('Loading controller data...\n');
load('Data.mat');

% Display loaded parameters
fprintf('PID Parameters: Kp=%.4f, Ki=%.4f, Kd=%.4f\n', Kp, Ki, Kd);

%% Step 2: Create the RL Training Environment
fprintf('Setting up RL environment...\n');

% Create environment (defines its own obs/action specs internally)
env = MissileEnv();

% Get observation and action info from environment
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

fprintf('Observation space: %s\n', mat2str(obsInfo.Dimension));
fprintf('Action space: %s\n', mat2str(actInfo.Dimension));

% Validate environment
validateEnvironment(env);
fprintf('Environment validated successfully!\n');

%% Step 3: Create TD3 Agent Networks
fprintf('Creating TD3 agent networks...\n');

numObs = obsInfo.Dimension(1);
numAct = actInfo.Dimension(1);

% Critic Network 1
criticNet1 = createCriticNetwork(numObs, numAct);
critic1 = rlQValueFunction(criticNet1, obsInfo, actInfo, ...
    'ObservationInputNames', 'obs', ...
    'ActionInputNames', 'act');

% Critic Network 2 (Twin Critic for TD3)
criticNet2 = createCriticNetwork(numObs, numAct);
critic2 = rlQValueFunction(criticNet2, obsInfo, actInfo, ...
    'ObservationInputNames', 'obs', ...
    'ActionInputNames', 'act');

% Actor Network
actorNet = createActorNetwork(numObs, numAct);
actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo, ...
    'ObservationInputNames', 'obs');

%% Step 4: Configure TD3 Agent
fprintf('Configuring TD3 agent...\n');

agentOpts = rlTD3AgentOptions(...
    'SampleTime', 0.01, ...
    'DiscountFactor', 0.99, ...
    'TargetSmoothFactor', 0.005, ...
    'ExperienceBufferLength', 1e6, ...
    'MiniBatchSize', 256);

% Exploration noise
agentOpts.ExplorationModel.StandardDeviation = 0.1;
agentOpts.ExplorationModel.StandardDeviationDecayRate = 1e-5;

% Target policy smoothing (TD3 specific)
agentOpts.TargetPolicySmoothModel.StandardDeviation = 0.2;
agentOpts.TargetPolicySmoothModel.LowerLimit = -0.5;
agentOpts.TargetPolicySmoothModel.UpperLimit = 0.5;

% Policy update frequency (delayed policy updates in TD3)
agentOpts.PolicyUpdateFrequency = 2;

% Create the TD3 agent
agent = rlTD3Agent(actor, [critic1, critic2], agentOpts);

%% Step 5: Training Options
maxEpisodes = 500;
maxSteps = 1200;  % 12 seconds at 0.01s sample time

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', maxEpisodes, ...
    'MaxStepsPerEpisode', maxSteps, ...
    'ScoreAveragingWindowLength', 20, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', -50);

% Save best agents during training
trainOpts.SaveAgentCriteria = 'EpisodeReward';
trainOpts.SaveAgentValue = -100;
trainOpts.SaveAgentDirectory = 'savedAgents';

%% Step 6: Train the Agent
fprintf('Starting TD3 training...\n');
fprintf('This may take a while. Monitor the training progress window.\n');

trainingStats = train(agent, env, trainOpts);

%% Step 7: Save Results
save('trained_td3_agent.mat', 'agent', 'trainingStats');
fprintf('Training complete! Agent saved to trained_td3_agent.mat\n');

%% Helper Functions
function net = createCriticNetwork(numObs, numAct)
    % Create critic network for Q-value estimation
    obsPath = [
        featureInputLayer(numObs, 'Name', 'obs')
        fullyConnectedLayer(256, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(256, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        ];

    actPath = [
        featureInputLayer(numAct, 'Name', 'act')
        fullyConnectedLayer(256, 'Name', 'fc3')
        reluLayer('Name', 'relu3')
        ];

    commonPath = [
        additionLayer(2, 'Name', 'add')
        fullyConnectedLayer(256, 'Name', 'fc4')
        reluLayer('Name', 'relu4')
        fullyConnectedLayer(1, 'Name', 'qvalue')
        ];

    net = layerGraph();
    net = addLayers(net, obsPath);
    net = addLayers(net, actPath);
    net = addLayers(net, commonPath);
    net = connectLayers(net, 'relu2', 'add/in1');
    net = connectLayers(net, 'relu3', 'add/in2');

    net = dlnetwork(net);
end

function net = createActorNetwork(numObs, numAct)
    % Create actor network - outputs k in [0,1] using sigmoid
    layers = [
        featureInputLayer(numObs, 'Name', 'obs')
        fullyConnectedLayer(256, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(256, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(numAct, 'Name', 'fc3')
        sigmoidLayer('Name', 'sigmoid')  % Ensures output in [0,1]
        ];

    net = dlnetwork(layerGraph(layers));
end
