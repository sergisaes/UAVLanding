%% ============================================================
% uav_landing_paper_TARGET_V13_VELOCITY_CHECK.m
% CORRECCIÓN FINAL:
% - Añadido chequeo de VELOCIDAD DEL DECK.
% - El dron no aterrizará si la plataforma se mueve rápido (>0.5 m/s).
% - Espera a momentos de "calma" en el movimiento random.
% =============================================================
clear; clc;

%% 1) CONFIGURATION
addpath('C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\matlab');
client = RemoteAPIClient();
sim = client.getObject('sim');
sim.startSimulation();

quad_base   = sim.getObject('/Quadcopter');
quad_target = sim.getObject('/target');     
deck        = sim.getObject('/OmniPlatform');

%% 2) PHYSICAL PARAMETERS
GT.Ax = 0.20; GT.Ay = 0.15; GT.Ah = 0.10;
noise_factor = 1.0 + (randn()*0.05); 
physParams.Ax = 0.20 * noise_factor; physParams.wx = 0.6; physParams.phx = 0;
physParams.Ay = 0.15 * noise_factor; physParams.wy = 0.5; physParams.phy = 0;
physParams.Ah = 0.10 * noise_factor; physParams.wz = 0.7; physParams.phz = 0;
physParams.vx = 0.02; physParams.vy = -0.01;

physParams.A_pitch = 0; physParams.w_pitch = 0.6; physParams.ph_pitch = 0;
physParams.A_roll  = 0; physParams.w_roll  = 0.5; physParams.ph_roll  = 0;

%% 3) SENSORS
Sensor.FOV_deg = 60; Sensor.MaxRange = 5.0; 
Sensor.PosNoise = 0.02; Sensor.VelNoise = 0.05; Sensor.DropOutProb = 0.02;

%% 4) STATE INITIALIZATION
deck0 = cell2mat(sim.getObjectPosition(deck,-1))';
xk = [deck0(1); 0; 0; deck0(2); 0; 0; deck0(3); 0; 0];
Pk = eye(9) * 0.1;
dt = 0.05;
F = eye(9); F(1,2)=dt; F(1,3)=0.5*dt^2; F(2,3)=dt; 
F(4,5)=dt; F(4,6)=0.5*dt^2; F(5,6)=dt; F(7,8)=dt; F(7,9)=0.5*dt^2; F(8,9)=dt; 
Q = eye(9)*1e-4; H = zeros(3,9); H(1,1)=1; H(2,4)=1; H(3,7)=1; 
R = eye(3)*(Sensor.PosNoise^2); 

%% 5) FLAGS & TIMERS
isFlaring = false; flareStartTime = 0; flareDuration = 2.5; 
flareStartOffset = 0; hasLanded = false;

approachTimer = 0; 
REQUIRED_STABLE_TIME = 1.5; % Reducido ligeramente para aprovechar oportunidades

%% 6) ONNX
onnxFile = 'gru_paper.onnx'; useOnnx = false;
if isfile(onnxFile), try, net = importONNXNetwork(onnxFile,'OutputLayerType','regression'); useOnnx = true; catch, end, end

%% ============================================================
% MAIN LOOP
% ============================================================
max_time = 180;
t = 0;
disp('Starting Control (Velocity Safety Check)...');

while sim.getSimulationState() ~= sim.simulation_stopped  &&  t < max_time
    
    %% (1) SENSORS
    GT_deck_pos = cell2mat(sim.getObjectPosition(deck,-1))';
    GT_uav_pos  = cell2mat(sim.getObjectPosition(quad_base,-1))';
    meas_uav_vel = cell2mat(sim.getObjectVelocity(quad_base))' + randn(3,1)*Sensor.VelNoise;
    
    rel_vec = GT_deck_pos - GT_uav_pos;
    dist_to_deck = norm(rel_vec);
    angle_off_nadir = atan2(norm(rel_vec(1:2)), abs(rel_vec(3))); 
    is_visible = (rad2deg(angle_off_nadir) < Sensor.FOV_deg/2) && (dist_to_deck < Sensor.MaxRange);
    if rand() < Sensor.DropOutProb, is_visible = false; end
    
    if is_visible, z_meas = GT_deck_pos + randn(3,1)*Sensor.PosNoise; has_vision = 1;
    else, z_meas = [NaN; NaN; NaN]; has_vision = 0; end
    
    %% (2) KALMAN
    xk = F*xk; Pk = F*Pk*F' + Q;
    if has_vision
        y_res = z_meas - H*xk; S = H*Pk*H' + R; K = Pk*H'/S;
        xk = xk + K*y_res; Pk = (eye(9)-K*H)*Pk;
    end
    
    %% (3) PREDICTION
    pred_horizon = dt * 2.0; t_future = t + pred_horizon;
    pred_pitch = physParams.A_pitch * sin(physParams.w_pitch * t_future + physParams.ph_pitch);
    pred_roll  = physParams.A_roll  * sin(physParams.w_roll  * t_future + physParams.ph_roll);
    
    xk_z_trend = xk(7) + xk(8)*pred_horizon;
    wave_z = physParams.Ah * sin(physParams.wz * t_future + physParams.phz);
    p_total = [xk(1)+xk(2)*pred_horizon; xk(4)+xk(5)*pred_horizon; xk_z_trend+wave_z];
    
    %% (4) CONFIDENCE
    uncert = trace(Pk(1:2:9, 1:2:9)); 
    if uncert < 0.05, pvis_sim = 0.95; elseif uncert < 0.3, pvis_sim = 0.6; else, pvis_sim = 0.2; end
    if has_vision == 0, pvis_sim = 0.3; end
    pvis = pvis_sim; if t < 3.0, pvis = 1.0; end 
    
    %% (5) CONTROL LOGIC
    dist_z = GT_uav_pos(3) - p_total(3);
    dist_xy = norm(GT_uav_pos(1:2) - p_total(1:2));
    
    % --- NUEVO: ESTIMACIÓN DE VELOCIDAD DEL DECK ---
    % Usamos el estado del Kalman (indices 2 y 5 son velocidades X e Y)
    deck_vel_x = xk(2);
    deck_vel_y = xk(5);
    deck_speed = norm([deck_vel_x, deck_vel_y]);
    
    if hasLanded
        sim.stopSimulation(); break; 
        
    elseif isFlaring
        mode = "FLARE";
        progress = min((t - flareStartTime) / flareDuration, 1.0);
        current_offset = flareStartOffset * (1 - progress);
        if progress >= 1.0, hasLanded = true; disp('=== TOUCHDOWN ==='); end
        target_ref = [p_total(1); p_total(2); p_total(3) + current_offset];
        
    else
        % Inputs Fuzzy
        dnorm = min(max((dist_z - 0.1)/1.5, 0), 1);
        [mode, ~] = fuzzy_logic_stable(pvis, dnorm, pred_pitch, pred_roll);
        
        % --- TIMER CON RESET POR MOVIMIENTO BRUSCO ---
        is_deck_calm = (deck_speed < 0.5); % Umbral: 0.5 m/s
        
        if strcmp(mode, "APPROACH") && is_deck_calm
            approachTimer = approachTimer + dt;
        else
            % Si salimos de approach O el barco acelera -> Reset Timer
            approachTimer = 0; 
        end
        
        % --- TRIGGER FINAL ---
        ready_to_flare = (strcmp(mode, "APPROACH") && pvis > 0.85);
        stable_enough  = (approachTimer > REQUIRED_STABLE_TIME);
        aligned_xy     = (dist_xy < 0.3);
        
        % Añadimos 'is_deck_calm' al trigger final para seguridad doble
        if ready_to_flare && dist_z < 1.3 && stable_enough && aligned_xy && is_deck_calm
            isFlaring = true;
            flareStartTime = t;
            flareStartOffset = dist_z;
            disp(['--- FLARE TRIGGERED (Deck Speed: ' num2str(deck_speed) ' m/s) ---']);
        end
        
        switch mode
            case "DESCEND"
                target_ref = p_total; 
            case "APPROACH"
                z_hover = p_total(3) + 0.8; 
                target_ref = [p_total(1); p_total(2); z_hover];
            case "HOLD"
                target_ref = [xk(1); xk(4); GT_uav_pos(3)];
            case "ABORT"
                target_ref = [xk(1); xk(4); deck0(3) + 2.5];
        end
    end
    
    %% (6) MOVE TARGET
    curr_target = cell2mat(sim.getObjectPosition(quad_target,-1))';
    vec = target_ref - curr_target;
    
    % Aumentamos agilidad en Approach para seguir movimientos random
    speed_limit = isFlaring * 2.0 + (~isFlaring) * 2.5; 
    
    if norm(vec) > speed_limit*dt, vec = vec*(speed_limit*dt/norm(vec)); end
    sim.setObjectPosition(quad_target, -1, (curr_target + vec)');
    
    if mod(t, 0.5) < dt
        fprintf('T:%.1f|Md:%s|Pvis:%.2f|Z:%.2f|DeckVel:%.2f|Tmr:%.1f\n', ...
            t, mode, pvis, dist_z, deck_speed, approachTimer);
    end
    pause(dt); t = t + dt;
end
try, sim.stopSimulation(); catch, end

%% FUZZY FUNCTION
function [s_str, idx] = fuzzy_logic_stable(pvis, d, pitch, roll)
    s_str = "ABORT"; idx = 1;
    if abs(pitch) > deg2rad(10) || abs(roll) > deg2rad(10), return; end

    TH_LOW_P = 0.4; TH_HI_P = 0.7;
    TH_LOW_D = 0.2; TH_HI_D  = 0.8; 
    
    if pvis < TH_LOW_P 
        if d < TH_LOW_D, s_str="ABORT"; idx=1; else, s_str="HOLD"; idx=2; end
    elseif pvis < TH_HI_P
        if d < TH_LOW_D, s_str="ABORT"; idx=1; else, s_str="APPROACH"; idx=3; end
    else 
        if d < TH_LOW_D, s_str="ABORT"; idx=1;
        elseif d > TH_HI_D
            s_str="DESCEND"; idx=4;
        else
            s_str="APPROACH"; idx=3; 
        end
    end
    if d > 0.9 && pvis > 0.8, s_str="DESCEND"; idx=4; end
end