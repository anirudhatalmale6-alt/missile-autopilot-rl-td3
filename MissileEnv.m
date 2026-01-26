classdef MissileEnv < rl.env.MATLABEnvironment
    %MISSILEENV Custom RL environment for missile autopilot control
    %   This environment simulates missile dynamics and trains an RL agent
    %   to blend H-infinity and PID controllers optimally.

    properties
        % Simulation parameters
        Ts = 0.01           % Sample time (s)
        MaxSteps = 1200     % Max steps per episode (12 seconds)

        % Missile state
        Alpha = 0           % Angle of attack (deg)
        Q = 0               % Pitch rate (deg/s)
        Nz = 0              % Normal acceleration (g)

        % Controller states
        Integrator = 0      % PID integrator state
        ActuatorState = 0   % Actuator state
        PrevError = 0       % Previous error for derivative
        PrevK = 0.5         % Previous k value

        % Time tracking
        CurrentStep = 0
        Time = 0

        % Reference signal
        RefValues = [50, -40, -15, 15, -1, 40]
        RefPeriod = 2       % seconds per step

        % PID parameters (loaded from Data.mat)
        Kp = 3.108
        Ki = 15.934
        Kd = 0.1187
    end

    properties (Access = protected)
        % Flag to check if done
        IsDone = false
    end

    methods
        function this = MissileEnv(obsInfo, actInfo)
            % Constructor - initialize the environment
            this = this@rl.env.MATLABEnvironment(obsInfo, actInfo);

            % Try to load PID parameters from Data.mat
            try
                data = load('Data.mat');
                if isfield(data, 'Kp'), this.Kp = double(data.Kp(1)); end
                if isfield(data, 'Ki'), this.Ki = double(data.Ki(1)); end
                if isfield(data, 'Kd'), this.Kd = double(data.Kd(1)); end
            catch
                % Use defaults if file not found
            end
        end

        function [observation, reward, isDone, loggedSignals] = step(this, action)
            % STEP Execute one step of the environment

            loggedSignals = [];

            % Get action (k value) - ensure scalar double
            k = double(max(0, min(1, action(1))));

            % Get current reference
            refIdx = mod(floor(this.Time / this.RefPeriod), length(this.RefValues)) + 1;
            nzRef = double(this.RefValues(refIdx));

            % Calculate error
            currentError = double(nzRef - this.Nz);
            errorDot = double((currentError - this.PrevError) / this.Ts);

            % H-infinity controller output (simplified)
            uHinf = this.computeHinfOutput(currentError);

            % PID controller output
            uPid = this.computePIDOutput(currentError, errorDot);

            % Blend controllers
            uBlend = k * uHinf + (1 - k) * uPid;

            % Actuator dynamics
            deltaQ = this.actuatorDynamics(uBlend);

            % Missile dynamics
            [nz, dAlpha, dQ] = this.missileDynamics(deltaQ);

            % Update states
            this.Alpha = double(this.Alpha + dAlpha * this.Ts);
            this.Q = double(this.Q + dQ * this.Ts);
            this.Nz = double(nz);
            this.Integrator = double(this.Integrator + currentError * this.Ts);
            this.PrevError = currentError;
            this.Time = double(this.Time + this.Ts);
            this.CurrentStep = this.CurrentStep + 1;

            % Calculate reward
            reward = double(this.calcReward(currentError, errorDot, k, this.PrevK, nzRef));
            this.PrevK = k;

            % Build observation vector [error; errorDot; nz] as 3x1 column vector
            observation = zeros(3, 1);
            observation(1) = double(currentError);
            observation(2) = double(errorDot);
            observation(3) = double(this.Nz);

            % Check termination conditions
            isDone = (this.CurrentStep >= this.MaxSteps) || ...
                     (abs(this.Nz) > 200) || ...
                     (abs(this.Alpha) > 50);

            this.IsDone = isDone;
        end

        function initialObservation = reset(this)
            % RESET Reset the environment to initial state

            % Reset missile states with small random perturbations
            this.Alpha = double(0.1 * randn);
            this.Q = double(0.1 * randn);
            this.Nz = 0;

            % Reset controller states
            this.Integrator = 0;
            this.ActuatorState = 0;
            this.PrevError = 0;
            this.PrevK = 0.5;

            % Reset time
            this.Time = 0;
            this.CurrentStep = 0;
            this.IsDone = false;

            % Initial observation as 3x1 column vector
            nzRef = double(this.RefValues(1));
            currentError = double(nzRef - this.Nz);

            initialObservation = zeros(3, 1);
            initialObservation(1) = currentError;
            initialObservation(2) = 0;
            initialObservation(3) = double(this.Nz);
        end
    end

    methods (Access = private)
        function u = computeHinfOutput(this, err)
            % Simplified H-infinity controller
            % In full implementation, use transfer functions from Data.mat
            u = double(-0.5 * err - 0.1 * this.Q);
        end

        function u = computePIDOutput(this, err, errDot)
            % PID controller
            u = double(this.Kp * err + this.Ki * this.Integrator + this.Kd * errDot);
        end

        function deltaQ = actuatorDynamics(this, u)
            % Second-order actuator dynamics (simplified)
            this.ActuatorState = double(0.9 * this.ActuatorState + 0.1 * u);
            deltaQ = double(max(-30, min(30, this.ActuatorState)));  % Saturation
        end

        function [nz, dAlpha, dQ] = missileDynamics(this, deltaQ)
            % Nonlinear missile dynamics

            % Physical parameters
            g = 9.8;
            Vm = 790;  % 2.5 Mach at 6000m
            qm = 0.66 * Vm^2 / 2;
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

            alpha = double(this.Alpha);
            q = double(this.Q);

            % Simplified state equations
            a11 = -qm * Sref * 0.202 / (m * Vm);
            a12 = 1;
            a21 = qm * Sref * dRef * 0.137 / Iy_b;
            a22 = -qm * Sref * dRef^2 * 18.56 / (2 * Vm * Iy_b);

            b11 = -qm * Sref * 0.0696 / (m * Vm);
            b21 = -qm * Sref * dRef * 0.5014 / Iy_b;

            % State derivatives
            dAlpha = double(a11 * alpha + a12 * q + b11 * deltaQ);
            dQ = double(a21 * alpha + a22 * q + b21 * deltaQ);

            % Normal acceleration
            c11 = -qm * Se * 0.202 / m;
            d11 = -qm * Se * 0.0696 / m;
            nz = double((c11 * alpha + d11 * deltaQ) / g);
        end

        function reward = calcReward(~, err, errDot, k, prevK, nzRef)
            % Calculate step reward
            deltaK = k - prevK;

            % Normalize error
            if abs(nzRef) > 1
                normError = err / abs(nzRef);
            else
                normError = err;
            end

            % Reward components
            rError = -1.0 * normError^2;
            rErrorDot = -0.1 * (errDot / 100)^2;
            rSmooth = -0.05 * (deltaK * 10)^2;

            if abs(err) < 2
                rSettling = 0.2 * (1 - abs(err) / 2);
            else
                rSettling = 0;
            end

            reward = double(rError + rErrorDot + rSmooth + rSettling);
            reward = double(max(-10, min(1, reward)));
        end
    end
end
