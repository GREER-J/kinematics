import numpy as np
import pytest
from typing import Optional
from src.dynamics_library.integration import FO_euler, rk4, scipy_odeint, IntegratorFunction




def falling_projectile(t: float, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
    g = 9.81
    return np.array([x[1], -g])  # [ds/dt, dv/dt]


# Unified simulation helper
def simulate_trajectory(integrator: IntegratorFunction, times: np.ndarray):
    x = np.array([0.0, 0.0])  # [position, velocity]
    trajectory = [x.copy()]
    for i in range(1, len(times)):
        t_span = (times[i-1], times[i])
        x = integrator(falling_projectile, x, None, t_span)
        trajectory.append(x.copy())
    return np.array(trajectory)


@pytest.mark.parametrize("name, integrator", [
    ("Euler", FO_euler),
    ("RK4", rk4),
    ("odeint", scipy_odeint),
])
def test_integrator_behaviour(name, integrator):
    g = 9.81
    T = 10
    N = 100
    times = np.linspace(0, T, N)
    s_exact = -0.5 * g * times**2
    v_exact = -g * times
    traj = simulate_trajectory(integrator, times)

    # 1. Shape check
    assert traj.shape == (N, 2), f"{name} returned shape {traj.shape}"

    # 2. Trend check — position should decrease over time
    pos = traj[:, 0]
    assert pos[-1] < pos[0], f"{name} did not fall downward: pos[0]={pos[0]:.2f}, pos[-1]={pos[-1]:.2f}"

    # 3. Error check — just a sanity ceiling (very generous for Euler)
    max_pos_err = np.max(np.abs(pos - s_exact))
    tolerance = {
        "Euler": 10.0,
        "RK4": 1e-2,
        "odeint": 1e-4
    }[name]

    assert max_pos_err < tolerance, f"{name} exceeded max error: {max_pos_err:.2f} > {tolerance}"
