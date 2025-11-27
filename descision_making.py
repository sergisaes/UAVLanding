#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math
import onnxruntime as ort  # MOTOR DE INFERENCIA ONNX

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String

class FuzzyMamdaniEngine:
    """
    Implementación ligera de lógica difusa Mamdani para el Paper.
    Entradas: Pvis [0,1], Distancia [0,1.5]
    Salida: Score de Acción [0, 10] -> (0-3 Abort, 2-6 Hold, 5-9 Appr, 8-10 Desc)
    """
    def __init__(self):
        # Definición de rangos de salida (Crisp values para defuzzificación simplificada o centroides)
        # En una implementación real de centroide, integramos áreas. 
        # Aquí usamos el método de Singleton Ponderado (Takagi-Sugeno simplificado) 
        # que es computacionalmente eficiente para ROS2 y equivalente funcionalmente.
        
        # Centros de los conjuntos de salida (ver Fig. 3 del paper)
        self.OUT_ABORT = 1.5
        self.OUT_HOLD = 4.0
        self.OUT_APPROACH = 7.0
        self.OUT_DESCEND = 9.5

    def triangle(self, x, a, b, c):
        return max(0.0, min((x - a)/(b - a + 1e-6), (c - x)/(c - b + 1e-6)))

    def trapezoid(self, x, a, b, c, d):
        return max(0.0, min(min((x - a)/(b - a + 1e-6), 1.0), (d - x)/(d - c + 1e-6)))

    def compute(self, pvis, d_norm):
        # 1. FUZZIFICACIÓN (Funciones de Pertenencia - Inputs)
        
        # Pvis (Confianza): Low, Med, High
        # Asumimos rango [0, 1]
        mu_p_low  = self.trapezoid(pvis, -0.1, 0.0, 0.3, 0.5)
        mu_p_med  = self.triangle(pvis, 0.3, 0.5, 0.8)
        mu_p_high = self.trapezoid(pvis, 0.6, 0.8, 1.0, 1.1)
        
        # Distancia (d): Low (Cerca/Peligro), Med, High (Lejos/Seguro)
        # Asumimos d normalizada [0, 1]
        mu_d_low  = self.trapezoid(d_norm, -0.1, 0.0, 0.2, 0.4)
        mu_d_med  = self.triangle(d_norm, 0.2, 0.5, 0.8)
        mu_d_high = self.trapezoid(d_norm, 0.6, 0.8, 1.0, 1.1)

        # 2. INFERENCIA (Reglas de la Tabla I)
        # Regla = min(antecedente1, antecedente2)
        
        # R1: Low Pvis & Low d -> Abort
        r1 = min(mu_p_low, mu_d_low)
        # R2: Low Pvis & Med d -> Hold
        r2 = min(mu_p_low, mu_d_med)
        # R3: Low Pvis & High d -> Hold
        r3 = min(mu_p_low, mu_d_high)
        
        # R4: Med Pvis & Low d -> Abort
        r4 = min(mu_p_med, mu_d_low)
        # R5: Med Pvis & Med d -> Approach
        r5 = min(mu_p_med, mu_d_med)
        # R6: Med Pvis & High d -> Approach
        r6 = min(mu_p_med, mu_d_high)
        
        # R7: High Pvis & Low d -> Abort
        r7 = min(mu_p_high, mu_d_low)
        # R8: High Pvis & Med d -> Approach
        r8 = min(mu_p_high, mu_d_med)
        # R9: High Pvis & High d -> Descend
        r9 = min(mu_p_high, mu_d_high)

        # 3. AGREGACIÓN Y DEFUZZIFICACIÓN (Centroide Ponderado)
        # Sumamos el "peso" de cada acción
        w_abort    = max(r1, r4, r7)
        w_hold     = max(r2, r3)
        w_approach = max(r5, r6, r8)
        w_descend  = r9

        # Calculamos el centro de gravedad
        numerator = (w_abort * self.OUT_ABORT) + \
                    (w_hold * self.OUT_HOLD) + \
                    (w_approach * self.OUT_APPROACH) + \
                    (w_descend * self.OUT_DESCEND)
                    
        denominator = w_abort + w_hold + w_approach + w_descend + 1e-6 # Evitar div/0
        
        score = numerator / denominator
        
        # 4. INTERPRETACIÓN FINAL (Salida Lingüística)
        # Mapeamos el score continuo [0-10] a modo discreto
        if score < 3.0: return "ABORT"
        elif score < 5.5: return "HOLD"
        elif score < 8.5: return "APPROACH"
        else: return "DESCEND"


class LandingControllerONNX(Node):
    def __init__(self):
        super().__init__('landing_controller_node')

        # --- 1. CARGA DEL MODELO ONNX ---
        onnx_path = "/home/tu_usuario/ros2_ws/src/tu_paquete/models/gru_residual.onnx"
        try:
            self.ort_session = ort.InferenceSession(onnx_path)
            self.get_logger().info(f"ONNX Model loaded from {onnx_path}")
            self.use_onnx = True
        except Exception as e:
            self.get_logger().error(f"FAILED to load ONNX: {e}")
            self.use_onnx = False

        # --- NUEVO: Inicializar Motor Fuzzy Mamdani ---
        self.fuzzy_engine = FuzzyMamdaniEngine()

        # --- 2. CONFIGURACIÓN GRU (HISTORIAL) ---
        self.history_len = 20
        self.input_dim = 6  # [pos_x, pos_y, vel_x, vel_y, wind, mask]
        # Buffer circular para el historial
        self.history_buffer = np.zeros((self.history_len, self.input_dim), dtype=np.float32)

        # --- 3. PARAMETROS FISICOS & KALMAN ---
        self.dt = 0.05
        self.physParams = {
            'Ah': 0.10, 'wz': 0.7, 'phz': 0.0,
            'A_pitch': 0.0, 'w_pitch': 0.6, 'ph_pitch': 0.0,
            'A_roll': 0.0, 'w_roll': 0.5, 'ph_roll': 0.0
        }

        # Kalman 9-Estados (Para estimar velocidad y tendencia base)
        self.xk = np.zeros((9, 1))
        self.Pk = np.eye(9) * 0.1
        dt = self.dt
        self.F = np.eye(9)
        self.F[0, 1] = dt; self.F[0, 2] = 0.5*dt**2; self.F[1, 2] = dt # X
        self.F[3, 4] = dt; self.F[3, 5] = 0.5*dt**2; self.F[4, 5] = dt # Y
        self.F[6, 7] = dt; self.F[6, 8] = 0.5*dt**2; self.F[7, 8] = dt # Z
        self.Q = np.eye(9) * 1e-4
        self.H = np.zeros((3, 9)); self.H[0,0]=1; self.H[1,3]=1; self.H[2,6]=1
        self.R = np.eye(3) * (0.02**2)

        # --- 4. LÓGICA DE CONTROL ---
        self.is_flaring = False
        self.flare_start_time = 0.0
        self.flare_duration = 2.5
        self.flare_start_offset = 0.0
        self.has_landed = False
        self.approach_timer = 0.0
        self.REQUIRED_STABLE_TIME = 1.5

        self.uav_pos = np.zeros(3)
        self.last_deck_meas = None
        self.last_vision_time = self.get_clock().now()
        self.vision_timeout = 0.5

        # --- ROS2 INTERFACE ---
        self.sub_odom = self.create_subscription(Odometry, '/uav/odom', self.odom_callback, 10)
        self.sub_deck = self.create_subscription(PoseStamped, '/vision/deck_pose', self.deck_callback, 10)
        self.pub_setpoint = self.create_publisher(PoseStamped, '/uav/setpoint_position/local', 10)
        self.pub_status = self.create_publisher(String, '/landing/status', 10)
        
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.start_sim_time = self.get_clock().now().nanoseconds / 1e9

    def odom_callback(self, msg):
        self.uav_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def deck_callback(self, msg):
        x = msg.pose.position.x; y = msg.pose.position.y; z = msg.pose.position.z
        self.last_deck_meas = np.array([[x], [y], [z]])
        self.last_vision_time = self.get_clock().now()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def control_loop(self):
        if self.has_landed: return

        now = self.get_clock().now()
        t_sim = (now.nanoseconds / 1e9) - self.start_sim_time
        
        # 1. GESTION VISION
        time_since_vision = (now - self.last_vision_time).nanoseconds / 1e9
        has_vision = time_since_vision < self.vision_timeout
        
        # 2. KALMAN (ESTIMACIÓN DE ESTADO BASE)
        self.xk = np.dot(self.F, self.xk)
        self.Pk = np.dot(np.dot(self.F, self.Pk), self.F.T) + self.Q
        
        if has_vision and self.last_deck_meas is not None:
            z_meas = self.last_deck_meas
            y_res = z_meas - np.dot(self.H, self.xk)
            S = np.dot(np.dot(self.H, self.Pk), self.H.T) + self.R
            K = np.dot(np.dot(self.Pk, self.H.T), np.linalg.inv(S))
            self.xk = self.xk + np.dot(K, y_res)
            self.Pk = np.dot((np.eye(9) - np.dot(K, self.H)), self.Pk)

        # 3. PREPARAR INPUTS PARA LA RED NEURONAL (GRU)
        # Input vector: [pos_x, pos_y, vel_x, vel_y, wind, mask]
        # Usamos el estado estimado por Kalman para alimentar la red
        pos_x = float(self.xk[0, 0])
        pos_y = float(self.xk[3, 0])
        vel_x = float(self.xk[1, 0])
        vel_y = float(self.xk[4, 0])
        wind_proxy = 0.0 # Placeholder si no tienes sensor de viento
        mask = 1.0 if has_vision else 0.0
        
        new_feature = np.array([pos_x, pos_y, vel_x, vel_y, wind_proxy, mask], dtype=np.float32)
        
        # Actualizar buffer (FIFO: desplazar y añadir al final)
        self.history_buffer = np.roll(self.history_buffer, -1, axis=0)
        self.history_buffer[-1, :] = new_feature

        # 4. INFERENCIA ONNX (PREDICCIÓN DEL RESIDUAL)
        dx, dy, pvis_logit = 0.0, 0.0, -5.0 # Valores por defecto (baja confianza)
        
        if self.use_onnx:
            # Formato esperado por ONNX (Batch, Seq, Features) -> (1, 20, 6)
            # OJO: Revisa si tu modelo espera (Seq, Batch, Feat) o (Batch, Feat, Seq)
            # Asumo (1, 20, 6) standard. Si tu Matlab usaba permute, puede ser distinto.
            input_tensor = self.history_buffer.reshape(1, self.history_len, self.input_dim)
            
            # Si tu modelo fue entrenado en Matlab y exportado, a veces espera (InputSize, SeqLen, Batch)
            # Prueba transponer si da error: input_tensor = input_tensor.transpose(2, 1, 0)
            
            try:
                # Ejecutar inferencia
                inputs = {self.ort_session.get_inputs()[0].name: input_tensor}
                outputs = self.ort_session.run(None, inputs)
                
                # Outputs del paper: [dx, dy, pvis_logit, var_x, var_y]
                # Asumimos que la salida es un vector de 5 elementos
                res = outputs[0].flatten()
                dx = float(res[0])
                dy = float(res[1])
                pvis_logit = float(res[2])
                # logvar_x = res[3], logvar_y = res[4] (No usados en fuzzy simple)
                
            except Exception as e:
                self.get_logger().warn(f"ONNX Error: {e}")

        # Confianza real de la red (Sigmoide)
        pvis = self.sigmoid(pvis_logit)

        # 5. PREDICCIÓN TOTAL (FÍSICA + RESIDUAL GRU)
        pred_horizon = self.dt * 2.0
        t_future = t_sim + pred_horizon
        
        # Modelo Físico Base (Kalman Trend + Olas)
        wave_z = self.physParams['Ah'] * math.sin(self.physParams['wz'] * t_future)
        p_phys_x = self.xk[0, 0] + self.xk[1, 0] * pred_horizon
        p_phys_y = self.xk[3, 0] + self.xk[4, 0] * pred_horizon
        p_phys_z = self.xk[6, 0] + self.xk[7, 0] * pred_horizon + wave_z
        
        # Sumar corrección de la GRU (Solo afecta X e Y)
        p_total_x = p_phys_x + dx
        p_total_y = p_phys_y + dy
        p_total_z = p_phys_z # GRU no predice Z en este paper generalmente
        
        p_total = np.array([p_total_x, p_total_y, p_total_z])
        
        # Predicciones secundarias para seguridad
        pred_pitch = 0.0 # self.physParams['A_pitch'] * ...
        pred_roll = 0.0

        # 6. LÓGICA DE CONTROL (MAMDANI)
        dist_z = self.uav_pos[2] - p_total[2]
        deck_speed = math.sqrt(self.xk[1,0]**2 + self.xk[4,0]**2)
        
        target_ref = np.zeros(3)
        mode = "INIT"

        if self.is_flaring:
            mode = "FLARE"
            progress = min((t_sim - self.flare_start_time) / self.flare_duration, 1.0)
            current_offset = self.flare_start_offset * (1.0 - progress)
            if progress >= 1.0: 
                self.has_landed = True
                self.get_logger().info("TOUCHDOWN")
            target_ref = p_total.copy()
            target_ref[2] += current_offset
        else:
            # --- FUZZY MAMDANI REAL ---
            dnorm = min(max((dist_z - 0.1)/1.5, 0.0), 1.0)
            
            # Pitch/Roll check (simulado como 0.0 en este caso)
            pred_pitch = 0.0
            pred_roll = 0.0
            
            if abs(pred_pitch) > 0.2 or abs(pred_roll) > 0.2:
                mode = "ABORT"
            else:
                # Llamamos al motor Mamdani con pvis real de la red
                mode = self.fuzzy_engine.compute(pvis, dnorm)
            
            # Timer y control de estabilidad
            is_calm = (deck_speed < 0.5)
            if mode == "APPROACH" and is_calm:
                self.approach_timer += self.dt
            else:
                self.approach_timer = 0.0
            
            # Trigger Mamdani: Usamos pvis real de la red neuronal
            if (mode == "APPROACH" and 
                pvis > 0.85 and 
                self.approach_timer > self.REQUIRED_STABLE_TIME and 
                is_calm):
                self.is_flaring = True
                self.flare_start_time = t_sim
                self.flare_start_offset = dist_z
                self.get_logger().info(f"FLARE! Pvis:{pvis:.2f} Mode:{mode}")

            # Asignar setpoint según modo Mamdani
            if mode == "DESCEND":
                target_ref = p_total.copy()
            elif mode == "APPROACH":
                target_ref = np.array([p_total[0], p_total[1], p_total[2] + 0.8])
            elif mode == "HOLD":
                target_ref = np.array([self.xk[0,0], self.xk[3,0], self.uav_pos[2]])
            else:  # ABORT
                target_ref = np.array([self.xk[0,0], self.xk[3,0], self.uav_pos[2] + 2.0])

        # Publicar setpoint
        msg = PoseStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(target_ref[0])
        msg.pose.position.y = float(target_ref[1])
        msg.pose.position.z = float(target_ref[2])
        self.pub_setpoint.publish(msg)
        
        # Publicar estado
        status_msg = String(data=f"Md:{mode}|Pv:{pvis:.2f}|T:{self.approach_timer:.1f}|D:{dist_z:.2f}")
        self.pub_status.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LandingControllerONNX()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()