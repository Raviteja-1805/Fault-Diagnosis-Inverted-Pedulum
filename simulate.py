import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from plant import InvertedPendulum, FaultInjector
from observer import LuenbergerObserver

plant = InvertedPendulum()
fi = FaultInjector()
obs = LuenbergerObserver(plant)

# LQR Controller for the Pendulum
Q = np.diag([10.0, 1.0, 100.0, 10.0])
R = np.array([[0.1]])
p = solve_discrete_are(plant.Ad, plant.Bd, Q, R)
k = np.linalg.inv(R + plant.Bd.T @ p @ plant.Bd) @ plant.Bd.T @ p @ plant.Ad

# Inject a gain-loss fault at step 500
fi.set_fault( 'gain_loss' , start_step=500)

state = np.array([0.0, 0.0, np.deg2rad(5), 0.0])
hist_state, hist_res, hist_det = [], [], []
fault_detected_at = None

for step in range(1000):
    u_nominal = float((-k @ state).flatten()[0])
    u_actual = fi.apply(u_nominal, step)
    residual = obs.update(u_actual, plant.C @ state)
    state = plant.step(state, u_actual)
    detected = obs.fault_detected()
    if detected and fault_detected_at is None:
        fault_detected_at = step
        print( f"FAULT DETECTED at step {step} (injected at 500)" )
        print( f"Detection delay: {step - 500} steps" )
    hist_state.append(state.copy())
    hist_res.append(residual)
    hist_det.append(float(detected))

hist_state = np.array(hist_state)
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(np.rad2deg(hist_state[:, 2]), 'b-' , label= 'Pendulum angle' )
axes[0].axvline(500, color= 'r' , ls= '--' , label= 'Fault injected' )
axes[0].set_ylabel( 'Angle (deg)' ); axes[0].legend(); axes[0].grid(True)
axes[0].set_title( 'Fault detection : gain-loss fault on inverted pendulum')
axes[1].plot(hist_res, 'g-' , label= 'Residual norm' )
axes[1].axhline(0.15, color= 'r' , ls= '--', label= 'Threshold' )
axes[1].axvline(500, color= 'r' , ls= '--' )
axes[1].set_ylabel( 'Residual' )
axes[1].legend()
axes[1].grid(True)
axes[2].plot(hist_det, 'k-' , lw=2)
axes[2].set_yticks([0, 1]); axes[2].set_yticklabels([ 'Normal' , 'Fault' ])
axes[2].set_xlabel( 'Time steps' ); axes[2].set_ylabel( 'Status' ); axes[2].grid(True)
plt.tight_layout()
plt.savefig( 'fault_detection_results.png' , dpi=150)
plt.show()