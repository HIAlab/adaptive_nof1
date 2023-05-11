import copy
import dataclasses
import hvplot.pandas
import json
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import panel
import seaborn as sns
from dataclasses import dataclass, field
from numpy.random import default_rng
from pandas.core.frame import dataclasses_to_dicts
from tqdm.auto import tqdm as progressbar
from typing import List, Callable

from sinot.simulation import Simulation as SinotSimulation
from src.metric import plot_score, score_df, score_df_iterative
from src.observation import History
from src.observation import Observation, Context, Outcome, Treatment
from src.policy import Policy, BlockPolicy
from src.treatmentplan import TreatmentPlan


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
    patient_id: int
    days_per_period: int = 1

    def __post_init__(self):
        # Load example params
        with open(self.parameter_file_path) as fp:
            study_params = json.load(fp)

        self.sinotSimulation = SinotSimulation(
            study_params, random_generator=numpy.random.default_rng(self.patient_id)
        )
        self.pat_complete = SinotSimulation.empty_dataframe()
        self.days_per_period = 1

    def __str__(self):
        return f"SinotModel(id:{self.patient_id})"

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
                "context": context,
                "treatment": Treatment(i=action),
                "outcome": Outcome(**{"outcome": -last_row["Uncertain_Low_Back_Pain"]}),
            }
        )


@dataclass
class Simulation:
    history: History
    policy: Policy
    model: Model

    @staticmethod
    def from_model_and_policy_with_copy(model: Model, policy: Policy):
        return Simulation(
            history=History(observations=[]),
            model=copy.deepcopy(model),
            policy=copy.deepcopy(policy),
        )

    @staticmethod
    def from_model_and_policy(model: Model, policy: Policy):
        return Simulation(history=History(observations=[]), model=model, policy=policy)

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

    def __getitem__(self, index):
        return dataclasses.replace(self, history=self.history[index])


@dataclass
class SeriesOfSimulations:
    simulations: List[Simulation]

    def __init__(
        self,
        model_from_patient_id: Callable[[int], Model],
        n_patients: int,
        policy,
        block_length=20,
        length=100,
    ):
        self.simulations = [
            Simulation.from_model_and_policy_with_copy(
                model_from_patient_id(index),
                BlockPolicy(policy, block_length=block_length),
            )
            for index in range(n_patients)
        ]
        for _ in progressbar(range(length), desc="Step"):
            for simulation in progressbar(
                self.simulations, desc="Simulation", leave=False
            ):
                simulation.step()
        self.n_patients = n_patients
        self.block_length = block_length

    def plot_line(self, metric):
        df = score_df_iterative(self.simulations, [metric], range(1, 100))
        ax = sns.lineplot(data=df, x="t", y="Score", hue="Simulation")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        return df

    def plot_allocations(self):
        data = []
        for patient_id in range(self.n_patients):
            patient_history = self.simulations[patient_id].history
            for block in range(100 // 20):
                index = block * self.block_length
                observation = patient_history.observations[index]
                data.append(
                    {
                        "patient_id": patient_id,
                        "block": block,
                        "treatment": observation.treatment.i,
                        "debug_info": self.simulations[
                            patient_id
                        ].policy.internal_policy.debug_information[block],
                    }
                )
        df = pd.DataFrame(data)
        return panel.panel(
            df.hvplot.heatmap(
                x="block",
                y="patient_id",
                C="treatment",
                hover_cols=["debug_info"],
                cmap="Category10",
                clim=(0, 10),
                grid=True,
            )
        )
