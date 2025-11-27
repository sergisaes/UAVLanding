# ============================================================
# GRU EXACTO SEGÚN EL PAPER: 
# - Input: [pos_phys, vel_phys, wind, mask]
# - Output: [dx, dy, pvis, logvar_x, logvar_y]
# - Entrenamiento: NLL + dropout visión + ruido + modelo físico
# - Exportacion a ONNX
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

# ------------------------------------------------------------
# 1. Physical Model (EXACTO según paper)
# ------------------------------------------------------------
def physical_model(t, Ax, wx, Ay, wy, vx, vy):
    x = vx*t + Ax*np.sin(wx*t)
    y = vy*t + Ay*np.sin(wy*t)
    return x, y

# ------------------------------------------------------------
# 2. Dataset Sintético EXACTO según el paper
# ------------------------------------------------------------
class SyntheticDataset:
    def __init__(self, T=200, dt=0.05):
        self.T = T
        self.dt = dt

    def generate(self, N=1000):
        X = []
        Y = []

        for _ in range(N):
            t = np.arange(self.T)*self.dt

            # parámetros físicos
            Ax = np.random.uniform(0.1,0.4)
            wx = np.random.uniform(0.3,1.0)
            Ay = np.random.uniform(0.1,0.4)
            wy = np.random.uniform(0.3,1.0)
            vx = np.random.uniform(-0.03,0.03)
            vy = np.random.uniform(-0.03,0.03)

            x_phys, y_phys = physical_model(t, Ax, wx, Ay, wy, vx, vy)
            pos_phys = np.stack([x_phys, y_phys], axis=1)

            # velocidades físicas exactas
            vel = np.gradient(pos_phys, axis=0) / self.dt

            # viento sintético (ruido lento)
            wind = np.random.normal(0, 0.02, size=(self.T,1))

            # ruido no modelado (residual real)
            residual = np.random.normal(0, 0.03, size=(self.T,2))

            # simulación final como en paper: p_sim = p_phys + residual
            pos_sim = pos_phys + residual

            # vision dropout mask (m(t))
            mask = (np.random.rand(self.T,1) > 0.15).astype(np.float32)

            # El dropout se aplica al sim en entrenamiento
            pos_obs = pos_sim.copy()
            pos_obs[mask[:,0] == 0] = np.nan

            # Ventana temporal N pasos
            Nw = 20
            for i in range(Nw, self.T):
                window_phys = pos_phys[i-Nw:i]          # (20,2)
                window_vel  = vel[i-Nw:i]               # (20,2)
                window_wind = wind[i-Nw:i]              # (20,1)
                window_mask = mask[i-Nw:i]              # (20,1)

                # Entrada final según paper
                feat = np.concatenate(
                    [window_phys, window_vel, window_wind, window_mask],
                    axis=1
                )                                       # (20,6)
                
                # Residual objetivo
                dx = residual[i,0]
                dy = residual[i,1]

                # pvis = probabilidad de visión válida
                pvis_target = mask[i,0]

                # varianza a aprender
                logvar_x = np.log(0.03)
                logvar_y = np.log(0.03)

                target = np.array([dx, dy, pvis_target, logvar_x, logvar_y])

                X.append(feat.astype(np.float32))
                Y.append(target.astype(np.float32))

        return np.array(X), np.array(Y)

# ------------------------------------------------------------
# 3. GRU EXACTO (input 6 dims, output 5 dims)
# ------------------------------------------------------------
class GRUResidual(nn.Module):
    def __init__(self, in_dim=6, hidden=64, out_dim=5):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x):
        out,_ = self.gru(x)
        h = out[:, -1, :]
        return self.fc(h)

# ------------------------------------------------------------
# 4. Loss NLL EXACTA del paper
# ------------------------------------------------------------
def NLLLoss(output, target):
    dx_pred, dy_pred, pvis_pred, logvar_x, logvar_y = output.T
    dx_t, dy_t, pvis_t, lx_t, ly_t = target.T

    var_x = torch.exp(logvar_x)
    var_y = torch.exp(logvar_y)

    # NLL exacto:
    loss_dx = (dx_t - dx_pred)**2 / var_x + logvar_x
    loss_dy = (dy_t - dy_pred)**2 / var_y + logvar_y

    # pérdida por confianza visión: pvis ≈ mask
    loss_pvis = nn.functional.binary_cross_entropy_with_logits(pvis_pred, pvis_t)

    return (loss_dx.mean() + loss_dy.mean() + loss_pvis)

# ------------------------------------------------------------
# 5. Entrenamiento + Exportación ONNX EXACTO
# ------------------------------------------------------------
def train_and_export():
    print("Generating dataset according to paper...")
    dataset = SyntheticDataset()
    X, Y = dataset.generate(N=800)

    # dims
    X = torch.tensor(X, dtype=torch.float32) # [B,20,6]
    Y = torch.tensor(Y, dtype=torch.float32) # [B,5]

    model = GRUResidual(in_dim=6, hidden=64, out_dim=5)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    B = 64
    nepoch = 8

    print("Training model with NLL loss...")
    for epoch in range(nepoch):
        perm = torch.randperm(len(X))
        total_loss = 0
        for i in range(0, len(X), B):
            idx = perm[i:i+B]
            xb = X[idx]
            yb = Y[idx]

            opt.zero_grad()
            pred = model(xb)
            loss = NLLLoss(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{nepoch} | Loss={total_loss/(len(X)//B):.6f}")

    print("Exporting ONNX...")
    dummy = torch.randn(1, 20, 6)
    torch.onnx.export(model, dummy, "gru_paper.onnx",
                      input_names=["x"], output_names=["y"],
                      opset_version=11,  dynamo=False)
    print("Saved as gru_paper.onnx")

if __name__ == "__main__":
    train_and_export()
