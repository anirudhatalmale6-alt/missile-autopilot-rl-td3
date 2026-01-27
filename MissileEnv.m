classdef MissileEnv < rl.env.MATLABEnvironment
    %MISSILEENV Custom RL environment for missile autopilot control
    %   This environment simulates missile dynamics and trains an RL agent
    %   to blend H-infinity and PID controllers optimally.

    properties
        % Simulation parameters
        Ts = 0.01           % Sample time (s)
        MaxSteps = 1200     % Max steps per episode (12 seconds)

        % Missile state: [alpha, q, nz, integrator, actuator_state]
        State = zeros(5,1)

        % Controller states
        PrevError = 0       % Previous error for derivative
        PrevK = 0.5         % Previous k value

        % Time tracking
        CurrentStep = 0
        Time = 0

        % Reference signal
        RefValues = [50, -40, -15, 15, -1, 40]
        RefPeriod = 2       % seconds per step

        % PID parameters
        Kp = 3.108
        Ki = 15.934
        Kd = 0.1187

        % H-infinity controller gains (simplified)
        Khinf_nz = 0.5
        Khinf_q = 0.1
    end

    methods
        function this = MissileEnv()
            % Define observation spec: [error, error_dot, nz] as 3x1
            ObservationInfo = rlNumericSpec([3 1], ...
                'LowerLimit', [-inf; -inf; -inf], ...
                'UpperLimit', [inf; inf; inf]);
            ObservationInfo.Name = 'observations';
            ObservationInfo.Description = 'error, error_derivative, nz';

            % Define action spec: k in [0,1]
            ActionInfo = rlNumericSpec([1 1], ...
                'LowerLimit', 0, ...
                'UpperLimit', 1);
            ActionInfo.Name = 'k';
            ActionInfo.Description = 'blend factor';

            % Call superclass constructor
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);

            % Load PID parameters
            try
                data = load('Data.mat');
                this.Kp = double(data.Kp(1));
                this.Ki = double(data.Ki(1));
                this.Kd = double(data.Kd(1));
            catch
            end
        end

        function [observation, reward, isDone, loggedSignals] = step(this, action)
            loggedSignals = [];

            % Extract states
            alpha = this.State(1);      % Angle of attack (deg)
            q = this.State(2);          % Pitch rate (deg/s)
            nz = this.State(3);         % Normal acceleration (g)
            integrator = this.State(4); % PID integrator
            actuatorState = this.State(5);

            % Get action (k value) - clamp to [0,1]
            k = min(max(double(action(1)), 0), 1);

            % Get reference
            refIdx = mod(floor(this.Time / this.RefPeriod), 6) + 1;
            nzRef = this.RefValues(refIdx);

            % Calculate error
            err = nzRef - nz;
            errDot = (err - this.PrevError) / this.Ts;

            % Limit error derivative to avoid spikes
            errDot = max(-500, min(500, errDot));

            % H-infinity controller output (simplified)
            uHinf = this.Khinf_nz * err - this.Khinf_q * q;

            % PID controller output
            uPid = this.Kp * err + this.Ki * integrator + this.Kd * errDot;

            % Blend controllers: u = k*hinf + (1-k)*pid
            uBlend = k * uHinf + (1 - k) * uPid;

            % Limit control output (fin deflection command)
            uBlend = max(-30, min(30, uBlend));

            % Actuator dynamics - first order approximation
            % TF: 32400 / (s^2 + 254.5s + 32400), wn=180, zeta=0.707
            tau_act = 1/180;
            actuatorState = actuatorState + (uBlend - actuatorState) * this.Ts / tau_act;
            deltaQ = max(-30, min(30, actuatorState));  % Saturation ±30 deg

            % Simplified stable missile dynamics
            % Using a more stable linearized model
            g = 9.8;
            Vm = 790;  % Velocity m/s

            % Time-varying mass
            if this.Time <= 8
                m = 101.3 - (101.3 - 87.27) * this.Time / 8;
            else
                m = 87.27;
            end

            % Simplified transfer function approach for stability
            % nz response to fin deflection (second order)
            wn_nz = 8;    % Natural frequency
            zeta_nz = 0.7; % Damping ratio

            % State space for nz dynamics: x = [nz, nz_dot]
            % Using alpha and q as intermediate states
            Kdc = 1.8;  % DC gain from deltaQ to nz

            % Simplified dynamics (stable 2nd order system)
            nz_dot = q * Vm / g;  % Approximate relationship

            % Update alpha based on nz and control
            alpha_dot = -2 * alpha + 0.5 * deltaQ + 0.1 * (nzRef - nz);

            % Update q (pitch rate) - damped response
            q_dot = -5 * q + 2 * deltaQ - 0.5 * alpha;

            % Calculate new nz from alpha and delta
            newNz = Kdc * (0.8 * alpha + 0.3 * deltaQ);

            % Update states with Euler integration
            this.State(1) = alpha + alpha_dot * this.Ts;
            this.State(2) = q + q_dot * this.Ts;
            this.State(3) = newNz;
            this.State(4) = integrator + err * this.Ts;
            this.State(5) = actuatorState;

            % CRITICAL: Clamp all states to prevent numerical instability
            this.State(1) = max(-45, min(45, this.State(1)));   % alpha: ±45 deg
            this.State(2) = max(-200, min(200, this.State(2))); % q: ±200 deg/s
            this.State(3) = max(-100, min(100, this.State(3))); % nz: ±100 g
            this.State(4) = max(-50, min(50, this.State(4)));   % integrator
            this.State(5) = max(-30, min(30, this.State(5)));   % actuator

            this.PrevError = err;
            this.Time = this.Time + this.Ts;
            this.CurrentStep = this.CurrentStep + 1;

            % Reward calculation
            deltaK = k - this.PrevK;

            % Scale error to reasonable range
            errScaled = err / 50;

            % Main reward: tracking performance
            reward = -errScaled^2;

            % Small penalty for oscillations
            reward = reward - 0.001 * (errDot/100)^2;

            % Small penalty for rapid k changes
            reward = reward - 0.01 * deltaK^2;

            % Bonus for good tracking
            if abs(err) < 5
                reward = reward + 0.5 * (1 - abs(err)/5);
            end
            if abs(err) < 10
                reward = reward + 0.2;
            end

            % Clamp reward
            reward = max(-2, min(1, reward));
            this.PrevK = k;

            % Observation (also clamped for safety)
            err_clamped = max(-100, min(100, err));
            errDot_clamped = max(-500, min(500, errDot));
            nz_clamped = max(-100, min(100, newNz));
            observation = [err_clamped; errDot_clamped; nz_clamped];

            % Terminate at max steps
            isDone = this.CurrentStep >= this.MaxSteps;
        end

        function initialObs = reset(this)
            % Reset state to near-zero with small perturbations
            this.State = [0.1*randn; 0.1*randn; 0; 0; 0];
            this.PrevError = 0;
            this.PrevK = 0.5;
            this.Time = 0;
            this.CurrentStep = 0;

            % Initial observation
            nzRef = this.RefValues(1);
            err = nzRef - this.State(3);
            initialObs = [err; 0; this.State(3)];
        end
    end
end
