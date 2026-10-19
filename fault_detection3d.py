import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import solve_discrete_are
from scipy.signal import place_poles

class pendulum:
    def __init__(self, dt=0.02):
        self.dt = dt
        g = 9.81; L = 0.6 ; m = 0.3 ; M = 1.2
        self.A = np.array([[0, 1, 0, 0],
                           [0, 0, -(m*g)/M, 0],
                           [0, 0, 0, 1],
                           [0, 0, (M+m)*g/(M*L), 0]], dtype=float)
        self.B = np.array([[0], [1/M], [0], [-1/(M*L)]], dtype=float)
        self.C = np.eye(4)
        self.Ad = np.eye(4) + self.A * dt
        self.Bd = self.B * dt


    def step(self, state, u):
        return self.Ad @ state + self.Bd.flatten() * u

class FaultInjector:
    def __init__(self, fault_type='gain_loss', fault_step=250):
        self.fault_type = fault_type
        self.fault_step = fault_step

    def apply(self, u, step):
        if step < self.fault_step:
            return u
        if self.fault_type == 'stuck':
            return 0.0
        elif self.fault_type == 'gain_loss':
            return u * 0.25
        elif self.fault_type == 'bias':
            return u + 3.0
        return u

class Observer:
    def __init__(self, plant, threshold=0.12):
        self.Ad  = plant.Ad
        self.Bd  = plant.Bd
        self.C  = plant.C
        self.threshold = threshold

        poles = [-2, -2.5, -3, -3.5]
        result = place_poles(self.Ad.T, self.C.T,
                             np.array(poles, dtype= complex))
        self.L = result.gain_matrix.T
        self.X_hat = np.zeros(4)

    def update(self, u, y_real):
        y_predicted = self.C @ self.X_hat
        residual    = y_real - y_predicted

        self.X_hat = (self.Ad @ self.X_hat
                      + self.Bd.flatten() * u
                      + self.L @ residual)
        return float(np.linalg.norm(residual))

plant = pendulum()
fi    = FaultInjector(fault_type='gain_loss', fault_step=250)
obs   =  Observer(plant)

Q = np.diag([10.0, 1.0, 100.0, 10.0])
R = np.array([[0.1]])
p = solve_discrete_are(plant.Ad, plant.Bd, Q, R)
k = (np.linalg.inv(R + plant.Bd.T @ p @ plant.Bd)
        @ plant.Bd.T @ p @ plant.Ad)

state = np.array([0.0, 0.0, np.deg2rad(8), 0.0])
N_STEPS = 600
hist_state = []
hist_res   = []
fault_detected_at = None

print("Simulating fault detection")
for step in range(N_STEPS):
    u_nominal = float((-k @ state).flatten()[0])
    u_actual = fi.apply(u_nominal, step)
    residual = obs.update(u_actual, plant.C @ state)
    state  = plant.step(state, u_actual)
    hist_state.append(state.copy())
    hist_res.append(residual)

    if (residual > obs.threshold
            and fault_detected_at is None
            and step > fi.fault_step):
        fault_detected_at = step
        print(f" Fault injected at step {fi.fault_step}"
                f" (t = {fi.fault_step*0.02:.2f} s)")
        print(f" Fault DETECTED at step {step} "
                f"(t = {step*0.02:.2f} s)")
        print(f" Detection delay: {(step - fi.fault_step)*0.02:.2f} s")

hist_state = np.array(hist_state)
hist_res   = np.array(hist_res)
hist_res_clean = np.nan_to_num(hist_res, nan=0.0, posinf=1.0, neginf=0.0)
t_arr      = np.arange(N_STEPS) * plant.dt
L_pend     = 0.6

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('black')

ax3d = fig.add_subplot(121, projection='3d')
ax3d.set_facecolor('black')
ax3d.set_xlim(-1.5, 1.5)
ax3d.set_ylim(-1.5, 1.5)
ax3d.set_zlim(-0.1, 1.4)
ax3d.set_xlabel('X (m)', color='silver', fontsize=9)
ax3d.set_ylabel('Y (m)', color= 'silver', fontsize=9)
ax3d.set_zlabel('Z (m)', color= 'silver', fontsize=9)
ax3d.tick_params(colors='dimgrey', labelsize=7)
ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor('darkslategrey')
ax3d.yaxis.pane.set_edgecolor('darkslategrey')
ax3d.zaxis.pane.set_edgecolor('darkslategrey')
ax3d.set_title('Inverted Pendulum - 3D view', color='white', fontsize=11, pad=10)

# Draw the cart rail (grey horizontal bar)
ax3d.plot([-1.4, 1.4], [0, 0], [0, 0], color='dimgrey', linewidth=4)

# Initial pendulum geometry
theta0 = hist_state[0, 2]
x_cart0 = hist_state[0, 0]
px0 = L_pend * np.sin(theta0)
pz0 = L_pend * np.cos(theta0)

# Cart body (blue box Outline)
cart_line, = ax3d.plot(
    [x_cart0-0.15, x_cart0+0.15, x_cart0+0.15, x_cart0-0.15, x_cart0-0.15],
    [0, 0, 0, 0, 0],
    [0, 0, 0.12, 0.12, 0],
    color='royalblue', linewidth=2)

# Pedulum rod(Orange when normal, red when fault)
rod_line, = ax3d.plot(
    [x_cart0, x_cart0 + px0],
    [0, 0],
    [0.12, 0.12 + pz0],
    color='orange',linewidth=3)


# pendulum bob (ball at the top)
bob_dot, = ax3d.plot(
    [x_cart0 + px0], [0], [0.12 + pz0],
    marker='o', color='orangered',markersize=14)

# Status text in the 3D view
status_text = ax3d.text2D(
    0.02, 0.06, 'Status: NORMAL',
    transform=ax3d.transAxes,
    color='limegreen', fontsize=11,
    fontweight='bold', va='bottom',
    fontfamily='monospace')

state_text = ax3d.text2D(
    0.02, 0.95,
    'theta = 0.00 deg\nx = 0.00 m\nt = 0.00 s',
    transform=ax3d.transAxes,
    color='lightgrey', fontsize=9,
    va='top', fontfamily='monospace')

# Right panels: rtesidual plot and angle plot
ax_res = fig.add_subplot(222)
ax_ang = fig.add_subplot(224)

for ax in [ax_res, ax_ang]:
    ax.set_facecolor('black')
    ax.tick_params(colors='grey')
    for spine in ax.spines.values():
        spine.set_color('dimgrey')

# Residual plot
ax_res.set_title('Observer residual (fault indicator)',
                 color='White', fontsize=10)
ax_res.set_ylabel('Residual size', color='silver', fontsize=9)
ax_res.set_xlabel('Residual size', color='silver', fontsize=9)
ax_res.axhline(obs.threshold,
               color='red', linewidth=1, linestyle='dashed',
               label='Threshold', alpha=0.8)
ax_res.axvline(fi.fault_step * plant.dt,
               color='orange', linewidth=1, linestyle='dashed',
               label='Fault injected', alpha=0.8)
ax_res.legend(fontsize=8, facecolor='black', labelcolor='white')
ax_res.set_xlim(0, t_arr[-1])
hist_res_clean = np.nan_to_num(hist_res_clean, nan=0.0, posinf=1.0, neginf=0.0)
ax_res.set_ylim(0, max(hist_res_clean) * 1.2 + 0.01)

# Angle plot
ax_ang.set_title('pendulum angle', color='white',fontsize=10)
ax_ang.set_ylabel('Angle (degrees)', color='silver', fontsize=9)
ax_ang.set_xlabel('Time (seconds)', color='silver', fontsize=9)
ax_ang.axvline(fi.fault_step * plant.dt,
               color='orange',linewidth=1, linestyle='dashed', alpha=0.8)
ax_ang.axhline(0, color='dimgrey', linewidth=0.5)
ax_ang.set_xlim(0, t_arr[-1])

# Live plot lines (updated in Animation)
res_line, = ax_res.plot([], [], color='deepskyblue', linewidth=1.5)
ang_line, = ax_ang.plot([], [], color='orange', linewidth=1.5)
res_dot, = ax_res.plot([], [], marker='o', color='deepskyblue', markersize=5)
ang_dot, = ax_ang.plot([], [], marker='o', color='orange', markersize=5)

plt.tight_layout(pad=2.0)

# Animation
def update(frame):
    i = min(frame * 3, N_STEPS - 1)
    state_i = hist_state[i]
    x_cart = state_i[0]
    theta_i = state_i[2]

    # pendulum tip position
    px = L_pend * np.sin(theta_i)
    pz = L_pend * np.cos(theta_i)

    # Move Cart
    cart_line.set_data(
        [x_cart-0.15, x_cart+0.15, x_cart+0.15, x_cart-0.15, x_cart-0.15],
        [0, 0, 0, 0, 0])
    cart_line.set_3d_properties([0, 0, 0.12, 0.12, 0])


    # Move pendulum rod
    rod_line.set_data([x_cart, x_cart + px], [0, 0])
    rod_line.set_3d_properties([0.12, 0.12 + pz])

    # Move bob
    bob_dot.set_data([x_cart + px], [0])
    bob_dot.set_3d_properties([0.12 + pz])

    # change color when fault is active 
    is_fault = i >= fi.fault_step
    is_detected = (fault_detected_at is not None
                  and i >= fault_detected_at)

    if is_detected:
        rod_line.set_color('red')
        bob_dot.set_color('red')
        status_text.set_text('Status: FAULT DETECTED')
        status_text.set_color('red')
    elif is_fault:
        rod_line.set_color('tomato')
        bob_dot.set_color('tomato')
        status_text.set_text('Status:FAULT ACTIVE')
        status_text.set_color('orange')
    else:
        rod_line.set_color('orange')
        bob_dot.set_color('orange')
        status_text.set_text('Status:NORMAL')
        status_text.set_color('limegreen')

    state_text.set_text(
        f"theta = {np.rad2deg(theta_i):+6.2f} deg\n"
        f"x     = {x_cart:+5.3f} m\n"
        f"t     = {i*plant.dt:.2f} s"
    )

    # Update live plots
    t_now = t_arr[:i+1]
    res_line.set_data(t_now, hist_res_clean[:i+1])
    ang_line.set_data(t_now, np.rad2deg(hist_state[:i+1, 2]))
    res_dot.set_data([t_now[-1]], [hist_res_clean[i]])
    ang_dot.set_data([t_now[-1]], [np.rad2deg(hist_state[i, 2])])

    # slowly rotate the 3d camera
    ax3d.view_init(elev=20, azim=45 + frame * 0.3)

    return rod_line, bob_dot, cart_line, res_line, ang_line

ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False)

plt.tight_layout(pad=2.0)
plt.suptitle('Fault Detection — Luenberger Observer on Inverted Pendulum',
             color='white', fontsize=13)

# Save as GIF
print("Saving GIF... please wait 2-3 minutes")
ani.save('fault_detection_3d.gif', writer='pillow', fps=15, dpi=80)
print("Saved: fault_detection_3d.gif")

# Show interactive window
plt.show()


