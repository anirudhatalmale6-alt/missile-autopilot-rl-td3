%% TD3 Reinforcement Learning Agent for Missile Autopilot
% Replaces Fuzzy Logic Controller with RL-based weighting between H-inf and PID
% Author: Anirudha Talmale
% MATLAB 2022b with Reinforcement Learning Toolbox

clear; clc; close all;

%% Load Controller Data
load('Data.mat');

%% Define Observation and Action Specifications
% Observation: [error, error_derivative, nz]
% error = nz_ref - nz (tracking error)
% error_derivative = d(error)/dt
% nz = current normal acceleration

numObservations = 3;
observationInfo = rlNumericSpec([numObservations 1], ...
    'LowerLimit', [-100; -500; -100], ...
    'UpperLimit', [100; 500; 100]);
observationInfo.Name = 'observations';
observationInfo.Description = 'error, error_derivative, nz';

% Action: weighting factor k in [0, 1]
% u = hinf*k + pid*(1-k)
numActions = 1;
actionInfo = rlNumericSpec([numActions 1], ...
    'LowerLimit', 0, ...
    'UpperLimit', 1);
actionInfo.Name = 'k';
actionInfo.Description = 'Weighting factor between H-inf and PID';

%% Create Simulink Environment
mdl = 'Hinf_PID_RL';
open_system(mdl);

% Create the environment
env = rlSimulinkEnv(mdl, [mdl '/RL Agent'], observationInfo, actionInfo);

% Set reset function to randomize initial conditions
env.ResetFcn = @(in) localResetFcn(in);

%% Create TD3 Agent
% Critic Network 1
criticLayerSizes = [256 256];
statePath1 = [
    featureInputLayer(numObservations, 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(criticLayerSizes(1), 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(criticLayerSizes(2), 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    ];

actionPath1 = [
    featureInputLayer(numActions, 'Normalization', 'none', 'Name', 'action')
    fullyConnectedLayer(criticLayerSizes(2), 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    ];

commonPath1 = [
    additionLayer(2, 'Name', 'add')
    fullyConnectedLayer(criticLayerSizes(2), 'Name', 'fc4')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(1, 'Name', 'qvalue')
    ];

criticNetwork1 = layerGraph();
criticNetwork1 = addLayers(criticNetwork1, statePath1);
criticNetwork1 = addLayers(criticNetwork1, actionPath1);
criticNetwork1 = addLayers(criticNetwork1, commonPath1);
criticNetwork1 = connectLayers(criticNetwork1, 'relu2', 'add/in1');
criticNetwork1 = connectLayers(criticNetwork1, 'relu3', 'add/in2');

criticOptions1 = rlRepresentationOptions('LearnRate', 1e-3, 'GradientThreshold', 1);
critic1 = rlQValueRepresentation(criticNetwork1, observationInfo, actionInfo, ...
    'Observation', {'state'}, 'Action', {'action'}, criticOptions1);

% Critic Network 2 (TD3 uses twin critics)
criticNetwork2 = layerGraph();
statePath2 = [
    featureInputLayer(numObservations, 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(criticLayerSizes(1), 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(criticLayerSizes(2), 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    ];

actionPath2 = [
    featureInputLayer(numActions, 'Normalization', 'none', 'Name', 'action')
    fullyConnectedLayer(criticLayerSizes(2), 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    ];

commonPath2 = [
    additionLayer(2, 'Name', 'add')
    fullyConnectedLayer(criticLayerSizes(2), 'Name', 'fc4')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(1, 'Name', 'qvalue')
    ];

criticNetwork2 = addLayers(criticNetwork2, statePath2);
criticNetwork2 = addLayers(criticNetwork2, actionPath2);
criticNetwork2 = addLayers(criticNetwork2, commonPath2);
criticNetwork2 = connectLayers(criticNetwork2, 'relu2', 'add/in1');
criticNetwork2 = connectLayers(criticNetwork2, 'relu3', 'add/in2');

criticOptions2 = rlRepresentationOptions('LearnRate', 1e-3, 'GradientThreshold', 1);
critic2 = rlQValueRepresentation(criticNetwork2, observationInfo, actionInfo, ...
    'Observation', {'state'}, 'Action', {'action'}, criticOptions2);

% Actor Network
actorLayerSizes = [256 256];
actorNetwork = [
    featureInputLayer(numObservations, 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(actorLayerSizes(1), 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(actorLayerSizes(2), 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(numActions, 'Name', 'fc3')
    sigmoidLayer('Name', 'sigmoid')  % Outputs k in [0, 1]
    ];

actorOptions = rlRepresentationOptions('LearnRate', 1e-4, 'GradientThreshold', 1);
actor = rlDeterministicActorRepresentation(actorNetwork, observationInfo, actionInfo, ...
    'Observation', {'state'}, 'Action', {'sigmoid'}, actorOptions);

%% Configure TD3 Agent Options
agentOptions = rlTD3AgentOptions(...
    'SampleTime', 0.01, ...
    'TargetSmoothFactor', 5e-3, ...
    'ExperienceBufferLength', 1e6, ...
    'MiniBatchSize', 256, ...
    'NumStepsToLookAhead', 1, ...
    'TargetPolicySmoothModel', rlAdditiveNoiseModel('StandardDeviation', 0.2, 'LowerLimit', -0.5, 'UpperLimit', 0.5), ...
    'ExplorationModel', rlAdditiveNoiseModel('StandardDeviation', 0.1));
agentOptions.PolicyUpdateFrequency = 2;
agentOptions.TargetUpdateFrequency = 2;

% Create TD3 Agent
agent = rlTD3Agent(actor, [critic1 critic2], agentOptions);

%% Training Options
maxEpisodes = 500;
maxSteps = 1200;  % 12 seconds simulation with 0.01s sample time
trainingOptions = rlTrainingOptions(...
    'MaxEpisodes', maxEpisodes, ...
    'MaxStepsPerEpisode', maxSteps, ...
    'ScoreAveragingWindowLength', 20, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', -50, ...
    'SaveAgentCriteria', 'EpisodeReward', ...
    'SaveAgentValue', -100, ...
    'SaveAgentDirectory', 'savedAgents');

%% Train the Agent
disp('Starting TD3 Training...');
trainingStats = train(agent, env, trainingOptions);

%% Save Trained Agent
save('trained_td3_agent.mat', 'agent');
disp('Training complete. Agent saved to trained_td3_agent.mat');

%% Local Reset Function
function in = localResetFcn(in)
    % Randomize reference signal amplitude for diverse training
    refAmplitudes = [50, -40, -15, 15, -1, 40];
    % Optionally shuffle or modify reference

    % Random initial conditions (small perturbations)
    in = setVariable(in, 'alpha_init', 0.1*randn);
    in = setVariable(in, 'q_init', 0.1*randn);
end
