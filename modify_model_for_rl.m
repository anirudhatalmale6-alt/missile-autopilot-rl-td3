%% Modify Existing Simulink Model to Use RL Agent
% This script copies Hinf_PID_Fuzzy.slx and prepares it for RL training
% by replacing the Fuzzy Logic Controller with an RL Agent block.
%
% Author: Anirudha Talmale
% MATLAB Version: R2022b with Reinforcement Learning Toolbox
%
% Run this script ONCE to create Hinf_PID_RL.slx

clear all; clc; close all;

%% Step 1: Copy the original model
srcModel = 'Hinf_PID_Fuzzy';
dstModel = 'Hinf_PID_RL';

fprintf('Creating RL-enabled model from %s.slx...\n', srcModel);

% Check if source exists
if ~exist([srcModel '.slx'], 'file')
    error('Source model %s.slx not found in current directory!', srcModel);
end

% Close if already loaded
if bdIsLoaded(srcModel)
    close_system(srcModel, 0);
end
if bdIsLoaded(dstModel)
    close_system(dstModel, 0);
end

% Copy the model file
copyfile([srcModel '.slx'], [dstModel '.slx']);
fprintf('Copied to %s.slx\n', dstModel);

% Load the new model
load_system(dstModel);
open_system(dstModel);

%% Step 2: Load controller data
load('Data.mat');
fprintf('Loaded Data.mat\n');

%% Step 3: Locate and remove Fuzzy Logic Controller
% The fuzzy block has a name with newline character
fuzzyBlockPath = [dstModel '/Fuzzy Logic ' char(10) 'Controller'];

try
    % Get fuzzy block position for placing RL agent nearby
    fuzzyPos = get_param(fuzzyBlockPath, 'Position');
    fuzzyPorts = get_param(fuzzyBlockPath, 'PortHandles');

    % Find what's connected to fuzzy input/output
    inLine = get_param(fuzzyPorts.Inport(1), 'Line');
    outLine = get_param(fuzzyPorts.Outport(1), 'Line');

    if inLine ~= -1
        inSrcPort = get_param(inLine, 'SrcPortHandle');
        delete_line(inLine);
    end
    if outLine ~= -1
        outDstPort = get_param(outLine, 'DstPortHandle');
        delete_line(outLine);
    end

    % Delete the fuzzy block
    delete_block(fuzzyBlockPath);
    fprintf('Removed Fuzzy Logic Controller block.\n');

catch ME
    warning('Could not automatically remove fuzzy block: %s', ME.message);
    fprintf('Please manually delete the Fuzzy Logic Controller block.\n');
    fuzzyPos = [510, 580, 560, 640];
end

%% Step 4: Add RL Agent block
rlAgentPath = [dstModel '/RL Agent'];
try
    add_block('rlblock/RL Agent', rlAgentPath, ...
        'Position', [fuzzyPos(1)-20, fuzzyPos(2)-30, fuzzyPos(3)+20, fuzzyPos(4)+30]);
    fprintf('Added RL Agent block.\n');
catch ME
    fprintf('Note: Could not add RL Agent block automatically.\n');
    fprintf('Please add it manually: Reinforcement Learning Toolbox > RL Agent\n');
end

%% Step 5: Add observation computation blocks
% We need: error and error_derivative as observations

% Add a branch to compute error derivative
derivPath = [dstModel '/Error_Deriv_RL'];
try
    add_block('simulink/Continuous/Derivative', derivPath, ...
        'Position', [420, 530, 460, 560]);
    fprintf('Added Derivative block for error_derivative.\n');
catch
    fprintf('Derivative block may already exist.\n');
end

% Add Mux for observations
obsMuxPath = [dstModel '/RL_Obs_Mux'];
try
    add_block('simulink/Signal Routing/Mux', obsMuxPath, ...
        'Position', [470, 570, 475, 620], ...
        'Inputs', '2', ...
        'DisplayOption', 'bar');
    fprintf('Added Mux for observations.\n');
catch
    fprintf('Obs Mux may already exist.\n');
end

%% Step 6: Add Reward computation block
rewardPath = [dstModel '/RL_Reward'];
try
    add_block('simulink/User-Defined Functions/MATLAB Function', rewardPath, ...
        'Position', [420, 480, 480, 520]);
    fprintf('Added Reward MATLAB Function block.\n');
    fprintf('   IMPORTANT: You need to edit this block and paste the reward code.\n');
catch
    fprintf('Reward block may already exist.\n');
end

% Add termination signal (constant 0 = never terminate)
termPath = [dstModel '/RL_IsDone'];
try
    add_block('simulink/Sources/Constant', termPath, ...
        'Position', [420, 650, 450, 670], ...
        'Value', '0');
    fprintf('Added IsDone constant (always 0 = continue).\n');
catch
end

%% Step 7: Save model
save_system(dstModel);
fprintf('\nModel saved as %s.slx\n', dstModel);

%% Step 8: Display connection instructions
fprintf('\n');
fprintf('========================================================\n');
fprintf('         MANUAL CONNECTIONS REQUIRED IN SIMULINK        \n');
fprintf('========================================================\n\n');

fprintf('Please open %s.slx and make these connections:\n\n', dstModel);

fprintf('STEP 1: OBSERVATION INPUT (to RL Agent "observation" port)\n');
fprintf('   - Find the error signal (output of subtraction nz_ref - nz)\n');
fprintf('   - This signal currently goes to Mux2 block\n');
fprintf('   - Connect error signal to:\n');
fprintf('     a) Error_Deriv_RL block input\n');
fprintf('     b) RL_Obs_Mux port 1\n');
fprintf('   - Connect Error_Deriv_RL output to RL_Obs_Mux port 2\n');
fprintf('   - Connect RL_Obs_Mux output to RL Agent "observation" port\n\n');

fprintf('STEP 2: ACTION OUTPUT (from RL Agent "action" port)\n');
fprintf('   - Connect RL Agent "action" output to MATLAB Function block\n');
fprintf('   - The MATLAB Function block receives (k, u_hinf, u_pid)\n');
fprintf('   - The k input should come from RL Agent\n\n');

fprintf('STEP 3: REWARD SIGNAL (to RL Agent "reward" port)\n');
fprintf('   - Double-click RL_Reward block and paste this code:\n');
fprintf('   ---------------------------------------------------------\n');
fprintf('   function reward = fcn(error, error_dot)\n');
fprintf('   err_norm = error / 50;\n');
fprintf('   reward = -err_norm^2;\n');
fprintf('   if abs(error) < 5\n');
fprintf('       reward = reward + 0.5;\n');
fprintf('   elseif abs(error) < 10\n');
fprintf('       reward = reward + 0.2;\n');
fprintf('   end\n');
fprintf('   reward = reward - 0.001 * (error_dot/100)^2;\n');
fprintf('   reward = max(-2, min(1, reward));\n');
fprintf('   end\n');
fprintf('   ---------------------------------------------------------\n');
fprintf('   - Connect error signal to RL_Reward input 1\n');
fprintf('   - Connect error_derivative to RL_Reward input 2\n');
fprintf('   - Connect RL_Reward output to RL Agent "reward" port\n\n');

fprintf('STEP 4: IS-DONE SIGNAL (to RL Agent "is-done" port)\n');
fprintf('   - Connect RL_IsDone constant (0) to RL Agent "is-done" port\n\n');

fprintf('STEP 5: VERIFY BLENDING\n');
fprintf('   - The MATLAB Function block should compute: u = k*u_hinf + (1-k)*u_pid\n');
fprintf('   - Make sure RL Agent action (k) goes to the correct input\n\n');

fprintf('========================================================\n');
fprintf('After completing connections:\n');
fprintf('1. Save the model\n');
fprintf('2. Run: setup_simulink_rl\n');
fprintf('========================================================\n');
