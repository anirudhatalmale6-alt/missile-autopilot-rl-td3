%% Setup Simulink-based TD3 Training for Missile Autopilot (V2)
% This script uses TD3 with continuous action k ∈ [0,1]
% with reward shaping to encourage smooth switching like fuzzy logic
%
% Key improvements:
% - Continuous action space (not discrete)
% - 3 observations: error, error_derivative, |nz_ref|
% - Reward penalizes rapid k changes
% - Reward encourages k near 0 or 1 (decisive selection)
%
% Author: Anirudha Talmale
% MATLAB Version: R2022b with Reinforcement Learning Toolbox

clear all; clc; close all;

%% Step 1: Load Controller Data
fprintf('Loading controller data...\n');
load('Data.mat');
fprintf('PID Parameters: Kp=%.4f, Ki=%.4f, Kd=%.4f\n', Kp, Ki, Kd);

%% Step 2: Define Observation and Action Specifications

% Observations: [error, error_derivative, |nz_ref|] (3 signals)
obsInfo = rlNumericSpec([3 1], ...
    'LowerLimit', [-200; -2000; 0], ...
    'UpperLimit', [200; 2000; 100]);
obsInfo.Name = 'observations';
obsInfo.Description = 'error, error_derivative, abs_nz_ref';

% Action: CONTINUOUS k in [0, 1]
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

% Actor Network - outputs k in [0,1] using sigmoid
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

% Exploration noise - moderate to allow exploration but not too noisy
agentOpts.ExplorationModel.StandardDeviation = 0.15;
agentOpts.ExplorationModel.StandardDeviationDecayRate = 1e-5;

% Target policy smoothing
agentOpts.TargetPolicySmoothModel.StandardDeviation = 0.1;
agentOpts.TargetPolicySmoothModel.LowerLimit = -0.3;
agentOpts.TargetPolicySmoothModel.UpperLimit = 0.3;

% Policy update frequency
agentOpts.PolicyUpdateFrequency = 2;

% Create TD3 agent
agent = rlTD3Agent(actor, [critic1, critic2], agentOpts);

%% Step 6: Training Options
maxEpisodes = 300;
maxSteps = 1200;  % 12 seconds at 0.01s sample time

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', maxEpisodes, ...
    'MaxStepsPerEpisode', maxSteps, ...
    'ScoreAveragingWindowLength', 30, ...
    'Verbose', true, ...
    'Plots', 'none', ...  % Disabled to avoid MATLAB UI bug
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 2000);  % High value to train all episodes

trainOpts.SaveAgentCriteria = 'EpisodeReward';
trainOpts.SaveAgentValue = 500;
trainOpts.SaveAgentDirectory = 'savedAgents_td3_v2';

%% Step 7: Train the Agent
fprintf('Starting TD3 training with Simulink model...\n');
fprintf('Training for up to %d episodes...\n', maxEpisodes);
fprintf('\nThe agent will learn continuous k in [0,1]:\n');
fprintf('  - k≈1 (H-inf) when |nz_ref| is large\n');
fprintf('  - k≈0 (PID) when |nz_ref| is small\n');
fprintf('  - Smooth transitions, minimal switching\n\n');

trainingStats = train(agent, env, trainOpts);

%% Step 8: Save Results
save('trained_td3_v2_simulink.mat', 'agent', 'trainingStats');
fprintf('Training complete! Agent saved to trained_td3_v2_simulink.mat\n');

%% Helper Functions
function net = createCriticNetwork(numObs, numAct)
    obsPath = [
        featureInputLayer(numObs, 'Name', 'obs')
        fullyConnectedLayer(128, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(128, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        ];

    actPath = [
        featureInputLayer(numAct, 'Name', 'act')
        fullyConnectedLayer(128, 'Name', 'fc3')
        reluLayer('Name', 'relu3')
        ];

    commonPath = [
        additionLayer(2, 'Name', 'add')
        fullyConnectedLayer(128, 'Name', 'fc4')
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
    % Actor outputs k in [0,1] using sigmoid
    layers = [
        featureInputLayer(numObs, 'Name', 'obs')
        fullyConnectedLayer(128, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(128, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(numAct, 'Name', 'fc3')
        sigmoidLayer('Name', 'sigmoid')
        ];

    net = dlnetwork(layerGraph(layers));
end

function in = localResetFcn(in)
    % Reset function - use default initial conditions
end
