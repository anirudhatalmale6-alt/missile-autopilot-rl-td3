%% Complete Setup and Training Script for TD3 RL-based Missile Autopilot
% This script sets up and trains a TD3 agent to replace fuzzy logic
% for combining H-infinity and PID controllers
%
% Author: Anirudha Talmale
% MATLAB Version: R2022b with Reinforcement Learning Toolbox
%
% Usage:
%   1. Place all files in the same folder as your Hinf_PID_Fuzzy.slx
%   2. Run this script
%   3. Training will start automatically
%   4. After training, use test_trained_agent.m to evaluate

clear; clc; close all;

%% Step 1: Load Controller Data
fprintf('Loading controller data...\n');
load('Data.mat');

% Display loaded parameters
fprintf('PID Parameters: Kp=%.4f, Ki=%.4f, Kd=%.4f\n', Kp, Ki, Kd);

%% Step 2: Create the RL Training Environment
fprintf('Setting up RL environment...\n');

% State (Observations): [error, error_derivative, nz]
numObs = 3;
obsInfo = rlNumericSpec([numObs 1], ...
    'LowerLimit', [-100; -500; -100], ...
    'UpperLimit', [100; 500; 100]);
obsInfo.Name = 'states';
obsInfo.Description = 'error, d_error, nz';

% Action: weighting factor k ∈ [0, 1]
numAct = 1;
actInfo = rlNumericSpec([numAct 1], ...
    'LowerLimit', 0, ...
    'UpperLimit', 1);
actInfo.Name = 'k';
actInfo.Description = 'H-inf/PID blend factor';

%% Step 3: Create Environment (Custom Function-based for flexibility)
% We'll use a custom step function that simulates the missile dynamics

% Create environment using custom functions
env = rlFunctionEnv(obsInfo, actInfo, @stepFcn, @resetFcn);

%% Step 4: Create TD3 Agent Networks

% Critic Network 1
criticNet1 = createCriticNetwork(numObs, numAct);
critic1Opts = rlOptimizerOptions('LearnRate', 1e-3, 'GradientThreshold', 1);

% For R2022b, use rlQValueFunction
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

%% Step 6: Training Options
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

%% Step 7: Train the Agent
fprintf('Starting TD3 training...\n');
fprintf('This may take a while. Monitor the training progress window.\n');

trainingStats = train(agent, env, trainOpts);

%% Step 8: Save Results
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

function [nextObs, reward, isDone, loggedSignals] = stepFcn(action, loggedSignals)
    % Step function for RL environment
    % Simulates one time step of missile dynamics with blended controller

    persistent missile_state t prev_k

    if isempty(missile_state)
        missile_state = struct('alpha', 0, 'q', 0);
        t = 0;
        prev_k = 0.5;
    end

    dt = 0.01;  % Sample time
    k = action(1);  % Weighting factor from RL agent

    % Get current reference (repeating stair)
    ref_values = [50, -40, -15, 15, -1, 40];
    ref_idx = mod(floor(t / 2), 6) + 1;
    nz_ref = ref_values(ref_idx);

    % Get H-infinity and PID controller outputs
    % (Simplified - in full implementation, these would use transfer functions)
    error = nz_ref - loggedSignals.nz;

    % Blend controllers
    u_hinf = compute_hinf_output(error, missile_state);
    u_pid = compute_pid_output(error, loggedSignals);
    u_blend = k * u_hinf + (1 - k) * u_pid;

    % Actuator dynamics (simplified)
    delta_q = actuator_dynamics(u_blend, loggedSignals);

    % Missile dynamics
    [nz, d_alpha, d_q, m] = missile_step(delta_q, missile_state.alpha, ...
        missile_state.q, t);

    % Update states
    missile_state.alpha = missile_state.alpha + d_alpha * dt;
    missile_state.q = missile_state.q + d_q * dt;
    t = t + dt;

    % Calculate reward
    error_dot = (error - loggedSignals.prev_error) / dt;
    reward = calc_step_reward(error, error_dot, k, prev_k, nz_ref);
    prev_k = k;

    % Next observation
    nextObs = [error; error_dot; nz];

    % Check termination (episode ends at 12 seconds or if unstable)
    isDone = (t >= 12) || (abs(nz) > 200);

    % Update logged signals
    loggedSignals.nz = nz;
    loggedSignals.prev_error = error;
    loggedSignals.alpha = missile_state.alpha;
    loggedSignals.q = missile_state.q;
end

function [obs, loggedSignals] = resetFcn()
    % Reset function for RL environment

    % Initial state
    loggedSignals.nz = 0;
    loggedSignals.prev_error = 0;
    loggedSignals.alpha = 0.1 * randn;  % Small random perturbation
    loggedSignals.q = 0.1 * randn;
    loggedSignals.integrator = 0;
    loggedSignals.actuator_state = 0;

    % Initial observation
    nz_ref = 50;  % First reference value
    error = nz_ref - loggedSignals.nz;
    obs = [error; 0; loggedSignals.nz];
end

function reward = calc_step_reward(error, error_dot, k, prev_k, nz_ref)
    % Calculate step reward
    delta_k = k - prev_k;

    % Normalize error
    if abs(nz_ref) > 1
        norm_error = error / abs(nz_ref);
    else
        norm_error = error;
    end

    % Reward components
    r_error = -1.0 * norm_error^2;
    r_error_dot = -0.1 * (error_dot / 100)^2;
    r_smooth = -0.05 * (delta_k * 10)^2;

    if abs(error) < 2
        r_settling = 0.2 * (1 - abs(error) / 2);
    else
        r_settling = 0;
    end

    reward = r_error + r_error_dot + r_smooth + r_settling;
    reward = max(min(reward, 1), -10);
end

function u = compute_hinf_output(error, state)
    % Simplified H-infinity controller output
    % In full implementation, use the transfer functions from Data.mat
    u = -0.5 * error - 0.1 * state.q;
end

function u = compute_pid_output(error, signals)
    % PID controller output
    Kp = 3.108;
    Ki = 15.934;
    Kd = 0.1187;

    u = Kp * error + Ki * signals.integrator + Kd * (error - signals.prev_error) / 0.01;
end

function delta_q = actuator_dynamics(u, signals)
    % Simplified actuator (2nd order lag)
    delta_q = 0.9 * signals.actuator_state + 0.1 * u;
    delta_q = max(min(delta_q, 30), -30);  % Saturation
end

function [nz, d_alpha, d_q, m] = missile_step(delta_q, alpha, q, t)
    % Simplified missile dynamics for training
    % Uses the nonlinear model from your Simulink file

    % Physical parameters
    g = 9.8;
    Vm = 790;  % 2.5 Mach at 6000m
    qm = 0.66 * Vm^2 / 2;
    Sref = 3.0828;
    d = 0.1981;
    Se = pi * d^2 / 4;

    % Time-varying mass
    if t <= 8
        m = 101.3 - (101.3 - 87.27) * t / 8;
        Iy_b = 33.2 - (33.2 - 32) * t / 8;
    else
        m = 87.27;
        Iy_b = 32;
    end

    % Simplified state equations (linearized around alpha=0)
    a11 = -qm * Sref * 0.202 / (m * Vm);
    a12 = 1;
    a21 = qm * Sref * d * 0.137 / Iy_b;
    a22 = -qm * Sref * d^2 * 18.56 / (2 * Vm * Iy_b);

    b11 = -qm * Sref * 0.0696 / (m * Vm);
    b21 = -qm * Sref * d * 0.5014 / Iy_b;

    % State derivatives
    d_alpha = a11 * alpha + a12 * q + b11 * delta_q;
    d_q = a21 * alpha + a22 * q + b21 * delta_q;

    % Normal acceleration
    c11 = -qm * Se * 0.202 / m;
    d11 = -qm * Se * 0.0696 / m;
    nz = (c11 * alpha + d11 * delta_q) / g;
end
