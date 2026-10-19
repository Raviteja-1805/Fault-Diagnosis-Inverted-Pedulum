import numpy as np

class InvertedPendulum:
    def __init__(self, dt=0.01):
        self.dt = dt
        g = 9.81; L = 0.5; m = 0.2; M = 1.0
        #state : [x, x_dot, theta, theta_dot]
        self.A = np.array([[0,1,0,0],[0,0,-(m*g)/M,0],
                          [0,0,0,1],[0,0,(M+m)*g/(M*L),0]], dtype=float)
        self.B = np.array([[0],[1/M],[0],[-1/M*L]], dtype=float)
        self.C = np.eye(4)
        self.Ad = np.eye(4) + self.A * dt
        self.Bd = self.B * dt

    def step(self, state, u):
        return self.Ad @ state + self.Bd * u

class FaultInjector:
    def __init__(self):
        self.fault_type = 'none'
        self.fault_start = 999999

    def set_fault(self, fault_type, start_step):
        self.fault_type = fault_type
        self.fault_start = start_step
        print(f"Fault scheduled: {fault_type} at step {start_step}" )

    def apply(self, u, step):
        if step < self.fault_start:
            return u
        if self.fault_type == 'stuck' :
            return 0.0
        elif self.fault_type == 'gain' :
            return u * 0.3
        elif self.fault_type == 'bias' :
            return u + 2.0
        return u