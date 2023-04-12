from src.observation import History
from src.policy import Policy
from src.treatmentplan import TreatmentPlan
from src.observation import Observation, Context

from dataclasses import dataclass, field
from typing import List

import numpy
from numpy.random import default_rng


@dataclass
class Model:
    mu_p: float
    epsilon_i_sigma: float
    c_sigma: float
    mu_T: List[float]
    alpha: List[float]
    rng: numpy.random.Generator = field(init=False)

    def __post_init__(self):
        RANDOM_SEED = 8927
        numpy.random.seed(RANDOM_SEED)
        self.rng = default_rng()

    def generate_context(self):
        return Context(c=[self.rng.standard_normal() * self.c_sigma])

    def observe_outcome(self, action, context) -> Observation:
        epsilon = self.rng.standard_normal() * self.epsilon_i_sigma
        y = (
            self.mu_p
            + epsilon
            + ((action == 1) * (self.mu_T[0] + self.alpha[0] * context.c[0]))
            + ((action == 2) * (self.mu_T[1] + self.alpha[1] * context.c[0]))
        )
        return Observation(
            **{
                "context": context,
                "treatment": action,
                "outcome": y,
            }
        )

    def sample_observation(self) -> Observation:
        n = self.treatment_plan.n_observations
        treatments = self.treatment_plan.treatments()

        epsilon = self.rng.standard_normal(n) * self.epsilon_i_sigma
        c = self.rng.standard_normal(n) * self.c_sigma
        y = (
            self.mu_p
            + epsilon
            + ((treatments == 1) * (self.mu_T[0] + self.alpha[0] * c))
            + ((treatments == 2) * (self.mu_T[1] + self.alpha[1] * c))
        )
        return Observation(
            **{
                "c": c,
                "treatments": self.treatment_plan.treatments(),
                "y": y,
            }
        )

    def true_coefficients(self):
        return {
            "mu_T_1": self.mu_T[0],
            "mu_T_2": self.mu_T[1],
            "alpha_T_1": self.alpha[0],
            "alpha_T_2": self.alpha[1],
        }

    def reference_values(self):
        return {
            "mu_T": self.mu_T,
            "mu_p": self.mu_p,
            "alpha": self.alpha,
            "epsilon_i_sigma": epsilon_i_sigma,
        }


@dataclass
class Simulation:
    history: History
    policy: Policy
    model: Model

    def step(self):
        context = self.model.generate_context()
        action = self.policy.choose_action(self.history, context)
        outcome = self.model.observe_outcome(action, context)
        self.history.add_outcome(outcome)
