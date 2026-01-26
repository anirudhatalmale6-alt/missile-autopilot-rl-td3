================================================================================
TD3 Reinforcement Learning Controller for Missile Autopilot
Replacing Fuzzy Logic with RL for H-infinity/PID Controller Blending
================================================================================

Author: Anirudha Talmale
MATLAB Version: R2022b with Reinforcement Learning Toolbox

================================================================================
FILES INCLUDED:
================================================================================

1. setup_and_train.m     - Main training script (RUN THIS FIRST)
2. train_rl_agent.m      - Alternative Simulink-based training script
3. test_trained_agent.m  - Test and evaluate trained agent
4. create_rl_model.m     - Script to create Simulink model with RL block
5. missile_dynamics.m    - Nonlinear missile dynamics function
6. reward_function.m     - Reward calculation for RL training
7. calc_reward.m         - Reward function for Simulink integration
8. README.txt            - This file

================================================================================
QUICK START:
================================================================================

1. Place all files in the same folder as your original files:
   - Data.mat
   - Hinf_PID_Fuzzy.slx
   - Hinf.slx

2. Open MATLAB R2022b (ensure RL Toolbox is installed)

3. Run the training script:
   >> setup_and_train

4. Wait for training to complete (monitor progress window)

5. After training, test the agent:
   >> test_trained_agent

================================================================================
APPROACH:
================================================================================

The TD3 (Twin Delayed DDPG) agent replaces the fuzzy logic controller:

CURRENT (Fuzzy):
  - Inputs: error (nz_ref - nz), d_error/dt
  - Output: k (weighting factor 0-1)
  - Rule-based: Uses fuzzy membership functions and rules

NEW (TD3 RL):
  - Inputs: error, d_error/dt, nz (state vector)
  - Output: k (weighting factor 0-1)
  - Learning-based: Learns optimal blending through reward maximization

Controller blending formula (unchanged):
  u = k * u_hinf + (1-k) * u_pid

================================================================================
TD3 ALGORITHM DETAILS:
================================================================================

TD3 is an improved version of DDPG with three key enhancements:

1. TWIN CRITICS
   - Uses two Q-value networks instead of one
   - Takes the minimum Q-value to reduce overestimation bias
   - Improves learning stability

2. DELAYED POLICY UPDATES
   - Updates actor network less frequently than critics
   - Allows critic to converge first
   - Reduces variance in policy gradients

3. TARGET POLICY SMOOTHING
   - Adds noise to target actions
   - Prevents exploitation of Q-function errors
   - Similar to regularization

Network Architecture:
- Actor: 3 inputs -> 256 -> 256 -> sigmoid -> k ∈ [0,1]
- Critic: 3 states + 1 action -> 256 -> 256 -> Q-value

================================================================================
REWARD FUNCTION DESIGN:
================================================================================

reward = r_error + r_error_dot + r_smooth + r_settling

Components:
1. r_error = -1.0 * (error/nz_ref)^2
   Primary objective: minimize tracking error

2. r_error_dot = -0.1 * (d_error/100)^2
   Reduce oscillatory behavior

3. r_smooth = -0.05 * (delta_k * 10)^2
   Encourage smooth controller transitions

4. r_settling = 0.2 * (1 - |error|/2) if |error| < 2g
   Bonus for achieving small steady-state error

================================================================================
TRAINING PARAMETERS:
================================================================================

- Sample Time: 0.01 s
- Episode Length: 12 s (1200 steps)
- Max Episodes: 500
- Mini-batch Size: 256
- Experience Buffer: 1,000,000 samples
- Actor Learning Rate: 1e-4
- Critic Learning Rate: 1e-3
- Discount Factor (gamma): 0.99
- Target Smooth Factor (tau): 0.005
- Exploration Noise: 0.1 (decaying)

================================================================================
EXPECTED RESULTS:
================================================================================

After training, the RL agent should:
1. Track step reference inputs with low overshoot
2. Automatically blend H-inf (robustness) and PID (performance)
3. Adapt to changing missile mass during boost phase
4. Maintain stability across operating conditions

Typical performance metrics:
- ITAE: Should be comparable or better than fuzzy logic
- Settling Time: < 1 second for step inputs
- Overshoot: < 20%
- Steady-state Error: < 2g

================================================================================
TROUBLESHOOTING:
================================================================================

1. "RL Toolbox not found"
   -> Install Reinforcement Learning Toolbox from Add-Ons

2. Training diverges (reward goes to -inf)
   -> Reduce learning rates
   -> Check reward function scaling
   -> Increase exploration noise

3. Agent always outputs k=0 or k=1
   -> Training not converged, run more episodes
   -> Check action scaling

4. Slow training
   -> Reduce mini-batch size
   -> Use parallel computing (parpool)

================================================================================
FOR THESIS DOCUMENTATION:
================================================================================

Key points to include:
1. TD3 algorithm overview and advantages over DDPG
2. State/Action space definition
3. Reward function design rationale
4. Network architecture choices
5. Hyperparameter selection
6. Training curves and convergence analysis
7. Comparison with fuzzy logic baseline
8. Robustness analysis (parameter variations, disturbances)

================================================================================
CONTACT:
================================================================================

For questions about this implementation, contact through Freelancer.com

================================================================================
