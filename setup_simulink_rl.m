%% Setup Simulink-based RL Training for Missile Autopilot
% This script sets up TD3 training using the actual Simulink missile model
% The RL agent replaces the Fuzzy Logic Controller
%
% Author: Anirudha Talmale
% MATLAB Version: R2022b with Reinforcement Learning Toolbox
%
% Prerequisites:
%   1. Data.mat with controller parameters
%   2. Hinf_PID_RL.slx (modified Simulink model with RL Agent block)
%
% Usage:
%   1. Run this script to train the TD3 agent
%   2. After training, the agent will be saved as 'trained_td3_simulink.mat'

clear all; clc; close all;

%% Step 1: Load Controller Data
fprintf('Loading controller data...\n');
load('Data.mat');
fprintf('PID Parameters: Kp=%.4f, Ki=%.4f, Kd=%.4f\n', Kp, Ki, Kd);

%% Step 2: Define Observation and Action Specifications

% Observations: [error, error_derivative] (2 signals from Simulink)
% error = nz_ref - nz
% error_derivative = d(error)/dt
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

% Model name (we'll create this model)
mdl = 'Hinf_PID_RL';

% Check if model exists
if ~exist([mdl '.slx'], 'file')
    error(['Simulink model ' mdl '.slx not found. Please create it first ' ...
           'by modifying Hinf_PID_Fuzzy.slx to include the RL Agent block.']);
end

% Create the Simulink environment
% The model must have:
%   - An RL Agent block named 'RL Agent'
%   - Observation input port connected to [error; error_derivative]
%   - Action output port connected to k (blending factor)
%   - A reward signal computed from tracking error
env = rlSimulinkEnv(mdl, [mdl '/RL Agent'], obsInfo, actInfo);

% Set reset function to randomize initial conditions
env.ResetFcn = @(in) localResetFcn(in);

%% Step 4: Create TD3 Agent Networks
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

%% Step 5: Configure TD3 Agent
fprintf('Configuring TD3 agent...\n');

agentOpts = rlTD3AgentOptions(...
    'SampleTime', 0.01, ...
    'DiscountFactor', 0.99, ...
    'TargetSmoothFactor', 0.005, ...
    'ExperienceBufferLength', 1e6, ...
    'MiniBatchSize', 128);

% Exploration noise (increased for better k exploration)
agentOpts.ExplorationModel.StandardDeviation = 0.4;
agentOpts.ExplorationModel.StandardDeviationDecayRate = 5e-6;  % Slower decay

% Target policy smoothing (TD3 specific)
agentOpts.TargetPolicySmoothModel.StandardDeviation = 0.2;
agentOpts.TargetPolicySmoothModel.LowerLimit = -0.5;
agentOpts.TargetPolicySmoothModel.UpperLimit = 0.5;

% Policy update frequency
agentOpts.PolicyUpdateFrequency = 2;

% Create the TD3 agent
agent = rlTD3Agent(actor, [critic1, critic2], agentOpts);

%% Step 6: Training Options
maxEpisodes = 300;
maxSteps = 1200;  % 12 seconds at 0.01s sample time

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', maxEpisodes, ...
    'MaxStepsPerEpisode', maxSteps, ...
    'ScoreAveragingWindowLength', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 800);  % Higher value to ensure full training

% Save best agents during training
trainOpts.SaveAgentCriteria = 'EpisodeReward';
trainOpts.SaveAgentValue = 0;
trainOpts.SaveAgentDirectory = 'savedAgents_simulink';

%% Step 7: Train the Agent
fprintf('Starting TD3 training with Simulink model...\n');
fprintf('Training for up to %d episodes...\n', maxEpisodes);

trainingStats = train(agent, env, trainOpts);

%% Step 8: Save Results
save('trained_td3_simulink.mat', 'agent', 'trainingStats');
fprintf('Training complete! Agent saved to trained_td3_simulink.mat\n');

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
        sigmoidLayer('Name', 'sigmoid')
        ];

    net = dlnetwork(layerGraph(layers));
end

function in = localResetFcn(in)
    % Reset function for Simulink environment
    % Randomize initial reference or initial conditions if desired

    % You can add randomization here, for example:
    % in = setVariable(in, 'initial_nz', randn*0.1);

    % For now, just use default initial conditions
end
