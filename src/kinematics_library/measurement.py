from abc import abstractmethod
from src.kinematics_library.event import BaseEvent


class Measurement(BaseEvent):
    def __init__(self, update_method='NewtonTrustEig', need_to_simulate=False):
        self.update_method = update_method
        self.need_to_simulate = need_to_simulate

    @abstractmethod
    def simulate(self, x, system):
        """ Generate a synthetic measurement given state x and system """

    @abstractmethod
    def log_likelihood(self, x, system):
        """ Return (log_prob, gradient, Hessian) of log-likelihood p(z | x) """

    def cost_joint_density(self, x, system):
        """ Return cost = -log(p(z | x)), gradient, Hessian """
        l, g, H = self.log_likelihood(x, system)
        return -l, -g, -H

    def update(self, system):
        """ Optional override — default implementation can be optimizer-based """
        raise NotImplementedError("Update method must be implemented in subclass")
