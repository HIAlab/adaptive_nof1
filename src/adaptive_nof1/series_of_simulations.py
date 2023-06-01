from __future__ import annotations

from adaptive_nof1.metrics.metric import score_df
from adaptive_nof1.models.model import Model
from adaptive_nof1.simulation import Simulation

import pandas as pd

from dataclasses import dataclass
from typing import List, Callable

from tqdm.auto import tqdm as progressbar
import seaborn as sns
import panel
import hvplot.pandas  # noqa
import matplotlib.pyplot as plt


@dataclass
class SeriesOfSimulations:
    simulations: List[Simulation]

    def __init__(
        self,
        model_from_patient_id: Callable[[int], Model],
        n_patients: int,
        policy,
        length=100,
    ):
        self.simulations = [
            Simulation.from_model_and_policy_with_copy(
                model_from_patient_id(index),
                policy,
            )
            for index in range(n_patients)
        ]

        for _ in progressbar(range(length)):
            for simulation in self.simulations:
                simulation.step()
        self.n_patients = n_patients

    def plot_line(self, metric):
        df = score_df(self.simulations, [metric])
        ax = sns.lineplot(data=df, x="t", y="Score", hue="Simulation")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        return df

    @staticmethod
    def plot_lines(listofseries: List[SeriesOfSimulations], metric):
        dataframes = [score_df(series.simulations, [metric]) for series in listofseries]
        ax = sns.lineplot(
            data=pd.concat(dataframes),
            x="t",
            y="score",
            hue="Simulation",
        )
        ax.set(xlabel="t", ylabel="Regret")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    def plot_allocations(self):
        data = []
        for patient_id in range(self.n_patients):
            patient_history = self.simulations[patient_id].history
            for index in range(len(patient_history)):
                observation = patient_history.observations[index]
                data.append(
                    {
                        "patient_id": patient_id,
                        "index": index,
                        "treatment": observation.treatment.i,
                        "debug_info": self.simulations[
                            patient_id
                        ].policy.debug_information[index],
                        "context": str(observation.context),
                        "outcome": str(observation.outcome),
                        "counterfactual_outcomes": str(
                            observation.counterfactual_outcomes
                        ),
                    }
                )
        df = pd.DataFrame(data)
        return panel.panel(
            df.hvplot.heatmap(
                x="index",
                y="patient_id",
                C="treatment",
                hover_cols=[
                    "debug_info",
                    "context",
                    "outcome",
                    "counterfactual_outcomes",
                ],
                cmap="Category10",
                clim=(0, 10),
                grid=True,
            )
        )
