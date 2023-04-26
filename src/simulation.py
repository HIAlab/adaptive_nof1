from src.observation import History
from src.policy import Policy
from src.treatmentplan import TreatmentPlan
from src.observation import Observation, Context, Outcome, Treatment
from src.metric import plot_score, score_df

from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import matplotlib

import numpy
from numpy.random import default_rng

from sinot.simulation import Simulation as SinotSimulation
import pandas as pd



import json


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
        return {"c": self.rng.standard_normal() * self.c_sigma}

    def observe_outcome(self, action, context) -> Observation:
        epsilon = self.rng.standard_normal() * self.epsilon_i_sigma
        y = (
            self.mu_p
            + epsilon
            + ((action == 1) * (self.mu_T[0] + self.alpha[0] * context["c"]))
            + ((action == 2) * (self.mu_T[1] + self.alpha[1] * context["c"]))
        )
        return Observation(
            **{
                "context": context,
                "treatment": Treatment(i=action),
                "outcome": Outcome({"outcome": y}),
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
class SinotModel:
    parameter_file_path: str
    sinotSimulation: SinotSimulation = field(init=False)
    pat_complete: pd.DataFrame = field(init=False)
    days_per_period: int = 1

    def __post_init__(self):
        # Load example params
        with open(self.parameter_file_path) as fp:
            study_params = json.load(fp)

        self.sinotSimulation = SinotSimulation(study_params)
        self.pat_complete = SinotSimulation.empty_dataframe()
        self.days_per_period = 1

    def __str__(self):
        return f"SinotModel"

    def generate_context(self):
        return {}

    def best_treatment(self):
        return 2

    def observe_outcome(self, action, context) -> Observation:
        action_to_treatment = {1: "Treatment_1", 2: "Treatment_2"}
        first_day = None if len(self.pat_complete) > 0 else "2018-01-01"
        self.pat_complete = self.sinotSimulation.step_patient(
            action_to_treatment[action],
            self.days_per_period,
            data=self.pat_complete,
            first_day=first_day,
        )
        last_row = self.pat_complete.iloc[-1]
        return Observation(
            **{
                "context": {},
                "treatment": Treatment(i=action),
                "outcome": Outcome({"outcome": last_row["Uncertain_Low_Back_Pain"]}),
            }
        )


@dataclass
class Simulation:
    history: History
    policy: Policy
    model: Model

    @staticmethod
    def from_model_and_policy(model: Model, policy: Policy):
        return Simulation(history=History(observations=[]), model=model, policy=policy)

    @staticmethod
    def simulation_study(models, policies, metrics, iterations):
        assert len(models) == len(policies)
        simulations = [
            Simulation.from_model_and_policy(model, policy)
            for [model, policy] in zip(models, policies)
        ]
        for _ in range(iterations):
            for simulation in simulations:
                simulation.step()
        plot_score(simulations, metrics)
        return score_df(simulations, metrics, minmax_normalization=True)

    def plot(self):
        axes = plt.axes()
        self.history.plot(axes)
        plt.title(str(self.policy) + str(self.model))

    def __str__(self):
        return f"Simulation\n{self.policy}{self.model}\n"

    def step(self):
        context = self.model.generate_context()
        action = self.policy.choose_action(self.history, context)
        outcome = self.model.observe_outcome(action, context)
        self.history.add_outcome(outcome)
