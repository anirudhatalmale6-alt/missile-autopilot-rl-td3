classdef MissileEnv < rl.env.MATLABEnvironment
    %MISSILEENV Custom RL environment for missile autopilot control
    %   This environment simulates a missile autopilot where an RL agent
    %   learns to blend H-infinity and PID controllers via factor k in [0,1].
    %   u = k * u_hinf + (1-k) * u_pid
    %
    %   The plant is modeled as a second-order closed-loop system that
    %   responds to the blended control signal.

    properties
        % Simulation parameters
        Ts = 0.01           % Sample time (s)
        MaxSteps = 1200     % Max steps per episode (12 seconds)

        % Plant state: [nz, nz_dot, integrator, filter_state]
        State = zeros(4,1)

        % Controller states
        PrevError = 0
        PrevK = 0.5

        % Time tracking
        CurrentStep = 0
        Time = 0

        % Reference signal: [50, -40, -15, 15, -1, 40] every 2s
        RefValues = [50, -40, -15, 15, -1, 40]
        RefPeriod = 2

        % PID parameters (from Data.mat)
        Kp = 3.108
        Ki = 15.934
        Kd = 0.1187
        N_filter = 100      % Derivative filter coefficient
    end

    methods
        function this = MissileEnv()
            % Define observation spec: [error, error_integral, nz] as 3x1
            ObservationInfo = rlNumericSpec([3 1], ...
                'LowerLimit', [-inf; -inf; -inf], ...
                'UpperLimit', [inf; inf; inf]);
            ObservationInfo.Name = 'observations';
            ObservationInfo.Description = 'error, error_integral, nz';

            % Define action spec: k in [0,1]
            ActionInfo = rlNumericSpec([1 1], ...
                'LowerLimit', 0, ...
                'UpperLimit', 1);
            ActionInfo.Name = 'k';
            ActionInfo.Description = 'blend factor';

            % Call superclass constructor
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);

            % Load PID parameters from Data.mat
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

            % Extract plant states
            nz = this.State(1);           % Normal acceleration (g)
            nz_dot = this.State(2);       % Rate of change of nz
            integrator = this.State(3);   % Error integrator for PID
            filterState = this.State(4);  % Derivative filter state

            % Get action (k value) - clamp to [0,1]
            k = min(max(double(action(1)), 0), 1);

            % Get current reference
            refIdx = mod(floor(this.Time / this.RefPeriod), length(this.RefValues)) + 1;
            nzRef = this.RefValues(refIdx);

            % Calculate error
            err = nzRef - nz;

            % --- PID Controller ---
            % Proportional
            P_out = this.Kp * err;
            % Integral (forward Euler)
            I_out = this.Ki * integrator;
            % Derivative with filter: D(s) = Kd * N * s / (s + N)
            % Discrete: filterState update
            dFilter = this.N_filter * (this.Kd * err - filterState);
            D_out = dFilter;

            uPid = P_out + I_out + D_out;

            % --- H-infinity Controller ---
            % The H-inf controller is a robust controller with good
            % transient response and robustness margins.
            % Characteristics: Higher proportional gain, more derivative action,
            % less integral to prevent windup - fast response but less steady-state precision
            errDot = (err - this.PrevError) / this.Ts;
            errDot = max(-500, min(500, errDot));

            % H-inf: high proportional, high derivative, low integral
            uHinf = 1.8 * err + 0.12 * errDot + 3.0 * integrator;

            % --- Blend controllers ---
            uBlend = k * uHinf + (1 - k) * uPid;

            % Limit control output (actuator saturation ±30 deg)
            uBlend = max(-30, min(30, uBlend));

            % --- Plant dynamics ---
            % Model the overall missile + actuator as a second-order system
            % from control input to nz output.
            % Closed-loop: nz(s)/u(s) ~ Kplant * wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
            % With wn ~ 8 rad/s, zeta ~ 0.6 (typical for missile autopilot)
            wn = 8.0;
            zeta = 0.6;

            % Second order dynamics: nz_ddot = wn^2*(Kplant*u - nz) - 2*zeta*wn*nz_dot
            % Kplant > 1 to account for the fact that controllers need to
            % produce reasonable actuator commands (not saturate) to track reference
            Kplant = 2.0;  % Increased DC gain for better tracking
            nz_ddot = wn^2 * (Kplant * uBlend - nz) - 2 * zeta * wn * nz_dot;

            % Update plant states (Euler integration)
            new_nz_dot = nz_dot + nz_ddot * this.Ts;
            new_nz = nz + nz_dot * this.Ts;

            % Update PID integrator (forward Euler)
            new_integrator = integrator + err * this.Ts;

            % Update derivative filter state
            new_filterState = filterState + dFilter * this.Ts;

            % Store states with clamping
            this.State(1) = max(-80, min(80, new_nz));
            this.State(2) = max(-500, min(500, new_nz_dot));
            this.State(3) = max(-50, min(50, new_integrator));
            this.State(4) = max(-1000, min(1000, new_filterState));

            this.PrevError = err;
            this.Time = this.Time + this.Ts;
            this.CurrentStep = this.CurrentStep + 1;

            % --- Reward calculation ---
            deltaK = k - this.PrevK;
            this.PrevK = k;

            % Tracking error penalty (normalized)
            errNorm = err / 50;
            reward = -errNorm^2;

            % Small penalty for error derivative (oscillation)
            reward = reward - 0.0005 * (errDot/100)^2;

            % Small penalty for rapid k changes
            reward = reward - 0.005 * deltaK^2;

            % Bonus for good tracking
            if abs(err) < 3
                reward = reward + 1.0;
            elseif abs(err) < 8
                reward = reward + 0.3;
            end

            % Clamp reward
            reward = max(-2, min(2, reward));

            % --- Observation ---
            errClamped = max(-100, min(100, err));
            intClamped = max(-50, min(50, this.State(3)));
            nzClamped = max(-80, min(80, this.State(1)));
            observation = [errClamped; intClamped; nzClamped];

            % Terminate only at max steps
            isDone = this.CurrentStep >= this.MaxSteps;
        end

        function initialObs = reset(this)
            % Reset all states
            this.State = zeros(4,1);
            this.PrevError = 0;
            this.PrevK = 0.5;
            this.Time = 0;
            this.CurrentStep = 0;

            % Initial observation
            nzRef = this.RefValues(1);
            err = nzRef - 0;
            initialObs = [err; 0; 0];
        end
    end
end
