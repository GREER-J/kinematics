from abc import abstractmethod
from src.dynamics_library.event import BaseEvent
from src.dynamics_library.system import BaseSystem


class Measurement(BaseEvent):
    def __init__(self, system: BaseSystem, time: float, update_method='NewtonTrustEig', need_to_simulate=False, **kwargs):
        super().__init__(time=time, system=system, **kwargs)
        self.update_method = update_method
        self.need_to_simulate = need_to_simulate

    @abstractmethod
    def simulate(self, x, system):
        """ Generate a synthetic measurement given state x and system """
        if self.system is None:
            raise RuntimeError("Measurement is not bound to a system")

    @abstractmethod
    def log_likelihood(self, x):
        """ Return (log_prob, gradient, Hessian) of log-likelihood p(z | x) """
        if self.system is None:
            raise RuntimeError("Measurement is not bound to a system")

    def cost_joint_density(self, x):
        """ Return cost = -log(p(z | x)), gradient, Hessian """
        l, g, H = self.log_likelihood(x)
        return -l, -g, -H

    def update(self, system: BaseSystem):
        """ Optional override — default implementation can be optimizer-based """
        self.system = system  # if not already assigned
        self._do_update()  # use self.system inside

    @abstractmethod
    def _do_update(self):
        """ Do update with system"""
        raise NotImplementedError("Update method must be implemented in subclass")
