classdef MissileEnv < rl.env.MATLABEnvironment
    %MISSILEENV Custom RL environment for missile autopilot control
    %   This environment simulates missile dynamics and trains an RL agent
    %   to blend H-infinity and PID controllers optimally.

    properties
        % Simulation parameters
        Ts = 0.01           % Sample time (s)
        MaxSteps = 1200     % Max steps per episode (12 seconds)

        % Missile state
        State = zeros(5,1)  % [alpha, q, nz, integrator, actuator_state]

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
            alpha = this.State(1);
            q = this.State(2);
            nz = this.State(3);
            integrator = this.State(4);
            actuatorState = this.State(5);

            % Get action
            k = min(max(double(action(1)), 0), 1);

            % Get reference
            refIdx = mod(floor(this.Time / this.RefPeriod), 6) + 1;
            nzRef = this.RefValues(refIdx);

            % Calculate error
            err = nzRef - nz;
            errDot = (err - this.PrevError) / this.Ts;

            % Controllers
            uHinf = -0.5 * err - 0.1 * q;
            uPid = this.Kp * err + this.Ki * integrator + this.Kd * errDot;

            % Blend
            uBlend = k * uHinf + (1 - k) * uPid;

            % Actuator
            actuatorState = 0.9 * actuatorState + 0.1 * uBlend;
            deltaQ = max(-30, min(30, actuatorState));

            % Missile dynamics
            g = 9.8; Vm = 790; qm = 0.66 * Vm^2 / 2;
            Sref = 3.0828; dRef = 0.1981; Se = pi * dRef^2 / 4;

            if this.Time <= 8
                m = 101.3 - (101.3 - 87.27) * this.Time / 8;
                Iy_b = 33.2 - (33.2 - 32) * this.Time / 8;
            else
                m = 87.27; Iy_b = 32;
            end

            a11 = -qm * Sref * 0.202 / (m * Vm);
            a21 = qm * Sref * dRef * 0.137 / Iy_b;
            a22 = -qm * Sref * dRef^2 * 18.56 / (2 * Vm * Iy_b);
            b11 = -qm * Sref * 0.0696 / (m * Vm);
            b21 = -qm * Sref * dRef * 0.5014 / Iy_b;

            dAlpha = a11 * alpha + q + b11 * deltaQ;
            dQ = a21 * alpha + a22 * q + b21 * deltaQ;

            c11 = -qm * Se * 0.202 / m;
            d11 = -qm * Se * 0.0696 / m;
            newNz = (c11 * alpha + d11 * deltaQ) / g;

            % Update state
            this.State(1) = alpha + dAlpha * this.Ts;
            this.State(2) = q + dQ * this.Ts;
            this.State(3) = newNz;
            this.State(4) = integrator + err * this.Ts;
            this.State(5) = actuatorState;

            this.PrevError = err;
            this.Time = this.Time + this.Ts;
            this.CurrentStep = this.CurrentStep + 1;

            % Reward
            if abs(nzRef) > 1
                normErr = err / abs(nzRef);
            else
                normErr = err;
            end
            deltaK = k - this.PrevK;
            reward = -normErr^2 - 0.1*(errDot/100)^2 - 0.05*(deltaK*10)^2;
            if abs(err) < 2
                reward = reward + 0.2 * (1 - abs(err)/2);
            end
            reward = max(-10, min(1, reward));
            this.PrevK = k;

            % Observation - MUST be 3x1 double column vector
            observation = [err; errDot; newNz];

            % Done check
            isDone = this.CurrentStep >= this.MaxSteps || abs(newNz) > 200 || abs(this.State(1)) > 50;
        end

        function initialObs = reset(this)
            % Reset state
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
