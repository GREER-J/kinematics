from typing import Protocol, Optional
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt


class IntegratorFunction(Protocol):
    def __call__(self, t: float, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        pass


class IntegrationProtocol(Protocol):
    def __call__(self, f: IntegratorFunction, x: np.ndarray, u: Optional[np.ndarray], t: tuple[float, float]) -> np.ndarray:
        ...


def scipy_odeint(f: IntegratorFunction, x: np.ndarray, u: np.ndarray, t: tuple[float, float]):
    t_eval = np.linspace(t[0], t[1], 2)

    def wrapped(x, t_):
        return f(t_, x, u)

    sol = odeint(wrapped, x, t_eval)
    return sol[-1]


def rk4(f: IntegratorFunction, x: np.ndarray, u: np.ndarray, t: tuple[float, float]):
    dt = t[1] - t[0]
    k1 = f(t[0], x, u)
    k2 = f(t[0], x+(dt/2.0)*k1, u)
    k3 = f(t[0], x+(dt/2.0)*k2, u)
    k4 = f(t[0], x+dt*k3, u)

    xn = x + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return xn


def FO_euler(f: IntegratorFunction, x: np.ndarray, u: np.ndarray, t: tuple[float, float]):
    dt = t[1] - t[0]
    dx = f(t[0], x, u)
    xn = x + dx*dt
    return xn
