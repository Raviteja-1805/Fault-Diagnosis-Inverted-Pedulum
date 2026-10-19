import numpy as np
from scipy.signal import place_poles

class LuenbergerObserver:
    def __init__(self, plant, threshold=0.15):
        self.Ad = plant.Ad
        self.Bd = plant.Bd
        self.C = plant.C
        self.threshold = threshold

        poles = [-8,-9,-10,-11]
        result = place_poles(self.Ad.T, self.C.T, poles)
        self.L = result.gain_matrix.T
        self.x_hat = np.zeros(4)
        self.residual_history = []

    def update(self, u, y_real):
        y_hat = self.C @ self.x_hat  # predicted output
        residual = y_real - y_hat    # innovation / residual
        # Observer update : correct estimate using residual
        self.x_hat = (self.Ad @ self.x_hat + 
                      self.Bd.flatten() * u +
                      self.L @ residual)
        r_norm = np.linalg.norm(residual)
        self.residual_history.append(r_norm)
        return r_norm

    def fault_detected(self):
        if len(self.residual_history) < 10:
            return False
        # Fault if recent average residual exceeds threshold
        recent = np.mean(self.residual_history[-10:])
        return recent > self.threshold