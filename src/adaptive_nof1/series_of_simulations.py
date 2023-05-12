from __future__ import annotations

from adaptive_nof1.metrics.metric import score_df_iterative
from adaptive_nof1.models.model import Model
from adaptive_nof1.policies.block_policy import BlockPolicy
from adaptive_nof1.simulation import Simulation

import pandas as pd

from dataclasses import dataclass
from typing import List, Callable

from tqdm.auto import tqdm as progressbar
import seaborn as sns
import panel
import hvplot.pandas # noqa

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

    @staticmethod
    def plot_lines(listofseries: List[SeriesOfSimulations], metric):
        dataframes = [score_df_iterative(series.simulations, [metric], range(1, 100)) for series in listofseries]
        ax = sns.lineplot(data=pd.concat(dataframes), x="t", y="Score", hue="Simulation")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


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
