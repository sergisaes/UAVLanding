<h1 align="center">UAV Landing on Moving Platforms</h1>

<p align="center">
  Autonomous landing of a multirotor UAV on a moving deck — a hybrid controller that fuses a
  physics deck-motion model, a data-driven GRU residual predictor, and a fuzzy decision layer.
</p>

<p align="center"><b>Research from my exchange semester at NTUST (Taipei) · co-authored paper submitted to IEEE.</b></p>

---

## 📄 Paper

**Real-Time Hybrid Predictive Fuzzy Control for UAV Landing on Moving Platforms**
Marcel Naderer, **Sergi Sanmartin**, Gaëtan Soudée, Yu-Chen Liu, Min-Fan Ricky Lee *(Member, IEEE)*
*MCI Innsbruck · UPC Barcelona · ECE Paris · National Taiwan University of Science and Technology*

> **Abstract.** The autonomous landing of multirotor UAVs on moving platforms is challenging due to
> coupled deck motion, environmental disturbances and intermittent perception. We present a real-time
> hybrid predictive fuzzy landing supervisor combining a physics-based deck-motion model, a data-driven
> residual predictor and a fuzzy logic decision layer. The physical model captures regular platform
> dynamics (sinusoidal heave, linear drift) as a deterministic baseline; a Gated Recurrent Unit network
> predicts stochastic residual corrections and a confidence measure (wind gusts, irregular wave motion,
> vision dropouts); and a Mamdani-type fuzzy supervisor fuses GRU confidence and obstacle clearance into
> interpretable behavioural modes (**hold, approach, descend, abort**). Implemented on a quadrotor testbed
> and evaluated through ground experiments, disturbance tests and moving-platform simulations, the system
> selects consistent, conservative modes under degraded perception — descending only when confidence and
> clearance are sufficient.

*Index terms: deep learning, fuzzy control, predictive models, real-time systems.*

## 🧠 How it works

1. **Physics model** — sinusoidal heave + linear drift give a deterministic baseline of the deck's future pose.
2. **GRU residual predictor** — a trained recurrent net corrects the baseline and outputs a confidence signal, absorbing wind gusts, irregular waves and vision dropouts.
3. **Fuzzy supervisor** — a Mamdani controller fuses GRU confidence + obstacle clearance into discrete, human-interpretable modes (hold / approach / descend / abort).
4. **Safety** — descent is only authorised when both confidence and clearance are high enough.

## 📁 Repo layout

| File | Role |
|------|------|
| `main.py` | CoppeliaSim ZMQ remote-API loop: deck motion + UAV proportional control |
| `train_export_gru.py` | Trains the GRU residual predictor and exports to ONNX |
| `gru_infer.py`, `gru_residual.onnx`, `gru_paper.onnx` | GRU inference + exported models |
| `fuzzy_mode_matlab.m`, `simulation_code.m` | Mamdani fuzzy supervisor + simulation (MATLAB) |
| `descision_making.py` | Decision layer wiring model + GRU + fuzzy modes |
| `ScenDrone.ttt` | CoppeliaSim scene (quadrotor + moving deck) |

## 🛠️ Stack

Python · GRU (PyTorch → ONNX) · MATLAB (fuzzy logic) · CoppeliaSim (ZMQ remote API) · JeroMQ · NumPy
