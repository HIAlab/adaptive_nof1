from observation import Observation
from treatmentplan import TreatmentPlan
from dataclasses import dataclass, field
from typing import List

import numpy
from numpy.random import default_rng


@dataclass
class Simulation:
    mu_p: float
    epsilon_i_sigma: float
    c_sigma: float
    treatment_plan: TreatmentPlan
    mu_T: List[float]
    alpha: List[float]
    rng: numpy.random.Generator = field(init=False)

    def __post_init__(self):
        RANDOM_SEED = 8927
        numpy.random.seed(RANDOM_SEED)
        self.rng = default_rng()

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
