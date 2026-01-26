function outputs = missile_dynamics(inputs)
%MISSILE_DYNAMICS Nonlinear missile dynamics for autopilot simulation
%
%   This function implements the nonlinear missile model based on
%   aerodynamic coefficients from lookup tables.
%
%   Inputs (via Simulink):
%       inputs(1) = delta_q - Fin deflection angle (degrees)
%       inputs(2) = alpha   - Angle of attack (degrees) [state feedback]
%       inputs(3) = q       - Pitch rate (deg/s) [state feedback]
%       inputs(4) = t       - Simulation time (s)
%
%   Outputs:
%       outputs(1) = nz      - Normal acceleration (g)
%       outputs(2) = d_alpha - Derivative of angle of attack
%       outputs(3) = d_q     - Derivative of pitch rate
%       outputs(4) = m       - Current missile mass (kg)

% Extract inputs
if length(inputs) >= 4
    delta_q = inputs(1);
    alpha = inputs(2);
    q = inputs(3);
    t = inputs(4);
else
    % Default values for initialization
    delta_q = 0;
    alpha = 0;
    q = 0;
    t = 0;
end

%% Missile Physical Parameters
h = 6000;           % Flight altitude (m)
g = 9.8;            % Gravitational acceleration (m/s^2)
Makha = 316;        % Speed of sound at 6000m (m/s)
Vm = 2.5 * Makha;   % Missile velocity (Mach 2.5)
ro = 0.66;          % Air density at 6000m (kg/m^3)
qm = ro * Vm^2 / 2; % Dynamic pressure (N/m^2)

% Geometric parameters
Sref = 3.0828;      % Reference area (m^2)
d = 0.1981;         % Reference diameter (m)
Se = pi * d^2 / 4;  % Cross-sectional area

%% Time-varying mass properties
m1 = 101.3;         % Initial mass (kg)
m2 = 87.27;         % Burnout mass (kg)
Iy_b1 = 33.2;       % Initial moment of inertia
Iy_b2 = 32;         % Burnout moment of inertia

L_momentcenter = 1.4562;
L_masscenter_endboots = 1.4362;
L_masscenter_burnout = 1.2877;

s_ngang1 = (L_masscenter_endboots - L_momentcenter) / d;
s_ngang2 = (L_masscenter_burnout - L_momentcenter) / d;

t_burning = 8;      % Burning time (s)

% Calculate current mass properties based on time
if t <= t_burning
    d_m = (m1 - m2) / t_burning;
    d_Iy = (Iy_b1 - Iy_b2) / t_burning;
    d_s_ngang = (s_ngang1 - s_ngang2) / t_burning;

    m = m1 - t * d_m;
    Iy_b = Iy_b1 - t * d_Iy;
    s_ngang = s_ngang1 - t * d_s_ngang;
else
    m = m2;
    Iy_b = Iy_b2;
    s_ngang = s_ngang2;
end

%% Aerodynamic Coefficients
% Axial force coefficients
Ca_0 = 4.362e-1;
deltaCa_b = 1.062e-1;
Ct = 1.2;           % Thrust coefficient

% Normal force coefficients
Cn_alphacube = -1.131e-5;
Cn_alpha = 2.020e-1;
Cn_alpha_absalpha = 1.110e-2;
Cn_alphadot = -2.781e-2;
Cn_q = 5.734e-1;
Cn_deltaq = 6.961e-2;
Cn_alpha_deltaq = 4.066e-4;

% Axial force derivatives
Ca_alpha = 3.886e-3;
Ca_alphasqua = -7.642e-5;
Ca_alphacube = 2.111e-6;
Ca_deltaq = -2.282e-2;
Ca_alpha_deltaq = 1.904e-3;
Ca_alphasqua_deltaq = -2.708e-5;

% Pitching moment coefficients
Cm_alpha = 1.373e-1;
Cm_alpha_absalpha = -1.020e-2;
Cm_alphacube = -6.864e-5;
Cm_q = -18.560;

% Cm_deltaq polynomial coefficients
Cm_0alpha_deltaq = -5.014e-1;
Cm_1alpha_deltaq = 3.520e-3;
Cm_2alpha_deltaq = 1.384e-4;
Cm_3alpha_deltaq = -1.705e-5;
Cm_4alpha_deltaq = 1.916e-7;

%% Calculate Cm_alpha_deltaq based on alpha sign
if alpha >= 0
    Cm_alpha_deltaq = Cm_0alpha_deltaq + Cm_1alpha_deltaq*alpha + ...
        Cm_2alpha_deltaq*alpha^2 + Cm_3alpha_deltaq*alpha^3 + ...
        Cm_4alpha_deltaq*alpha^4;
else
    Cm_alpha_deltaq = 2*Cm_0alpha_deltaq - Cm_1alpha_deltaq*abs(alpha) - ...
        Cm_2alpha_deltaq*abs(alpha)^2 - Cm_3alpha_deltaq*abs(alpha)^3 - ...
        Cm_4alpha_deltaq*abs(alpha)^4;
end

%% Calculate sinc function
if alpha == 0
    mysinc = 1;
else
    mysinc = sind(alpha) / alpha;
end

%% State Space Matrices
D_alphadot = 1 + qm*Sref*cosd(alpha)*d*Cn_alphadot / (2*m*Vm^2);

a11 = qm*Sref * (sind(alpha)*(Ca_alpha + Ca_alphasqua*abs(alpha) + Ca_alphacube*alpha^2)*sign(alpha) + ...
    mysinc*(Ca_0 + deltaCa_b - Ct) - ...
    cosd(alpha)*(Cn_alpha + Cn_alpha_absalpha*abs(alpha) + Cn_alphacube*alpha^2)) / (m*Vm*D_alphadot);

a12 = (1 - qm*Sref*d*cosd(alpha)*Cn_q / (2*m*Vm^2)) / D_alphadot;

a21 = qm*Sref*d * (Cm_alpha + Cm_alpha_absalpha*abs(alpha) + Cm_alphacube*alpha^2 + ...
    s_ngang*(Cn_alpha + Cn_alpha_absalpha*abs(alpha) + Cn_alphacube*alpha^2)) / Iy_b;

a22 = (qm*Sref*d^2) * (Cm_q + s_ngang*Cn_q) / (2*Vm*Iy_b);

b11 = qm*Sref * ((sind(alpha)*Ca_deltaq*sign(delta_q) - cosd(alpha)*Cn_deltaq) + ...
    alpha*(sind(alpha)*Ca_alpha_deltaq - cosd(alpha)*Cn_alpha_deltaq) + ...
    sind(alpha)*Ca_alphasqua_deltaq*alpha^2) / (m*Vm*D_alphadot);

b21 = qm*Sref*d * (Cm_alpha_deltaq + s_ngang*(Cn_deltaq + alpha*Cn_alpha_deltaq)) / Iy_b;

%% Output coefficients for normal acceleration
c11 = -qm*Se * (Cn_alpha + Cn_alpha_absalpha*abs(alpha) + Cn_alphacube*alpha^2 + ...
    Cn_alphadot*d*a11/(2*Vm)) / m;
c12 = -qm*Se*d * (Cn_q + Cn_alphadot*a12) / (2*m*Vm);
d11 = -qm*Se * (Cn_deltaq + Cn_alpha_deltaq*alpha + Cn_alphadot*d*b11/(2*Vm)) / m;

%% State Equations
d_alpha = a11*alpha + a12*q + b11*delta_q;
d_q = a21*alpha + a22*q + b21*delta_q;

%% Output: Normal Acceleration (in g's)
nz = (c11*alpha + c12*q + d11*delta_q) / g;

%% Pack outputs
outputs = [nz; d_alpha; d_q; m];

end
