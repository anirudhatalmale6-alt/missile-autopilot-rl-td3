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
            errDot = max(-1000, min(1000, errDot));

            % H-infinity controller output (simplified proportional-derivative)
            uHinf = -0.8 * err - 0.15 * q;

            % PID controller output
            uPid = this.Kp * err + this.Ki * integrator + this.Kd * errDot;

            % Blend controllers
            uBlend = k * uHinf + (1 - k) * uPid;

            % Limit control output
            uBlend = max(-50, min(50, uBlend));

            % Actuator dynamics (2nd order with damping)
            % Transfer function: 32400 / (s^2 + 254.5s + 32400)
            % Simplified as first-order for stability
            tau_act = 1/180;  % Actuator time constant
            actuatorState = actuatorState + (uBlend - actuatorState) * this.Ts / tau_act;
            deltaQ = max(-30, min(30, actuatorState));  % Saturation ±30 deg

            % Missile dynamics (more stable version)
            g = 9.8;
            Vm = 790;  % 2.5 Mach at 6000m
            qm = 0.66 * Vm^2 / 2;  % Dynamic pressure
            Sref = 3.0828;
            dRef = 0.1981;
            Se = pi * dRef^2 / 4;

            % Time-varying mass
            if this.Time <= 8
                m = 101.3 - (101.3 - 87.27) * this.Time / 8;
                Iy_b = 33.2 - (33.2 - 32) * this.Time / 8;
            else
                m = 87.27;
                Iy_b = 32;
            end

            % Aerodynamic derivatives (from the model)
            Cn_alpha = 0.202;
            Cn_q = 0.5734;
            Cn_deltaq = 0.0696;
            Cm_alpha = 0.137;
            Cm_q = -18.56;
            Cm_deltaq = -0.5014;

            % State space coefficients
            a11 = -qm * Sref * Cn_alpha / (m * Vm);
            a12 = 1 - qm * Sref * dRef * Cn_q / (2 * m * Vm);
            a21 = qm * Sref * dRef * Cm_alpha / Iy_b;
            a22 = qm * Sref * dRef^2 * Cm_q / (2 * Vm * Iy_b);

            b11 = -qm * Sref * Cn_deltaq / (m * Vm);
            b21 = qm * Sref * dRef * Cm_deltaq / Iy_b;

            % State derivatives
            dAlpha = a11 * alpha + a12 * q + b11 * deltaQ;
            dQ = a21 * alpha + a22 * q + b21 * deltaQ;

            % Normal acceleration output
            c11 = -qm * Se * Cn_alpha / m;
            d11 = -qm * Se * Cn_deltaq / m;
            newNz = (c11 * alpha + d11 * deltaQ) / g;

            % Update state with Euler integration
            this.State(1) = alpha + dAlpha * this.Ts;
            this.State(2) = q + dQ * this.Ts;
            this.State(3) = newNz;
            this.State(4) = integrator + err * this.Ts;  % Update integrator
            this.State(5) = actuatorState;

            % Anti-windup for integrator
            this.State(4) = max(-100, min(100, this.State(4)));

            this.PrevError = err;
            this.Time = this.Time + this.Ts;
            this.CurrentStep = this.CurrentStep + 1;

            % Reward calculation - scaled to be more manageable
            deltaK = k - this.PrevK;

            % Scale error to reasonable range
            errScaled = err / 50;  % Normalize by typical reference magnitude

            % Main reward: tracking performance (quadratic penalty, scaled)
            reward = -errScaled^2;

            % Small penalty for oscillations in error derivative
            reward = reward - 0.001 * (errDot/100)^2;

            % Small penalty for rapid k changes (encourage smoothness)
            reward = reward - 0.01 * deltaK^2;

            % Bonus for good tracking (within 5g of reference)
            if abs(err) < 5
                reward = reward + 0.5 * (1 - abs(err)/5);
            end

            % Smaller penalties for extreme states
            if abs(this.State(1)) > 30
                reward = reward - 0.1;
            end
            if abs(newNz) > 100
                reward = reward - 0.1;
            end

            % Clamp reward to reasonable range per step
            reward = max(-2, min(1, reward));
            this.PrevK = k;

            % Observation
            observation = [err; errDot; newNz];

            % Only terminate at max steps (removed early termination for instability)
            isDone = this.CurrentStep >= this.MaxSteps;
        end

        function initialObs = reset(this)
            % Reset state with small perturbations
            this.State = [0.01*randn; 0.01*randn; 0; 0; 0];
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
