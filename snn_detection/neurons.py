"""Neuron and synapse models shared by the event-driven SNNs."""

import numpy as np


class AdaptiveLIF:
    """Leaky integrate-and-fire neuron with an adaptive (homeostatic) threshold.

    After every spike the threshold rises by ``theta_plus`` and then decays back to
    ``v_th_base`` with time constant ``tau_theta``. The neuron is clamped to rest
    for ``refractory_period`` steps after firing.
    """

    def __init__(self, v_rest=0.0, v_th_base=35.0, tau_v=10.0, resistance=1.0,
                 refractory_period=3, tau_theta=100.0, theta_plus=1.0):
        self.v_rest = v_rest
        self.v_th_base = v_th_base
        self.tau_v = tau_v
        self.resistance = resistance
        self.refractory_period = refractory_period
        self.tau_theta = tau_theta
        self.theta_plus = theta_plus
        self.reset()

    def step(self, current, dt=1):
        """Advance one time step. Returns ``(spike, membrane_potential)``."""
        self.v_th += (self.v_th_base - self.v_th) / self.tau_theta * dt

        if self.refractory_time > 0:
            self.refractory_time -= dt
            self.v = self.v_rest
            return 0, self.v

        self.v += (-(self.v - self.v_rest) + self.resistance * current) / self.tau_v * dt
        if self.v >= self.v_th:
            self.v = self.v_rest
            self.refractory_time = self.refractory_period
            self.v_th += self.theta_plus
            return 1, self.v
        return 0, self.v

    def reset(self):
        """Reset potential, refractory counter and threshold (called per image)."""
        self.v = self.v_rest
        self.refractory_time = 0
        self.v_th = self.v_th_base


def stdp_dw(delta_t, a_plus=0.1, a_minus=0.1, tau_plus=5.0, tau_minus=5.0):
    """Pair-based STDP weight change for ``delta_t = t_post - t_pre`` (scalar or array).

    Pre-before-post (``delta_t > 0``) potentiates, post-before-pre depresses, and
    simultaneous spikes leave the weight unchanged.
    """
    delta_t = np.asarray(delta_t, dtype=float)
    ltp = a_plus * np.exp(-np.abs(delta_t) / tau_plus)
    ltd = -a_minus * np.exp(-np.abs(delta_t) / tau_minus)
    return np.where(delta_t > 0, ltp, np.where(delta_t < 0, ltd, 0.0))
