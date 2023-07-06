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
    def score_data(list_of_series: List[SeriesOfSimulations], metrics):
        dataframes = [
            score_df(series.simulations, metrics) for series in list_of_series
        ]
        return pd.concat(
            dataframes,
        )

    @staticmethod
    def plot_bar(list_of_series: List[SeriesOfSimulations], metrics):
        scored_df = score_df(
            [
                simulation
                for series in list_of_series
                for simulation in series.simulations
            ],
            metrics,
            minmax_normalization=False,
        )
        # Select only rows at the end of the trial
        filterd_score_df = scored_df[scored_df["t"] == scored_df["t"].max()]
        sns.boxplot(
            data=filterd_score_df,
            x="score",
            y="simulation",
            hue="metric",
        )

    @staticmethod
    def plot_lines(list_of_series: List[SeriesOfSimulations], metrics):
        scored_df = score_df(
            [
                simulation
                for series in list_of_series
                for simulation in series.simulations
            ],
            metrics,
            minmax_normalization=False,
        )
        scored_df["simulation_x_metric"] = scored_df["simulation"] + scored_df["metric"]
        print(scored_df)
        ax = sns.lineplot(
            data=scored_df,
            x="t",
            y="score",
            hue="simulation_x_metric",
            # units="patient_id",
            # estimator=None,
        )
        ax.set(xlabel="t", ylabel="Regret")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    def serialize(self):
        return pickle.dumps(self)

    @staticmethod
    def deserialize(data):
        return pickle.loads(data)

    def serialize_to_file(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)
        return filename

    @staticmethod
    def load_from_file(filename):
        with open(filename, "rb") as file:
            object = pickle.load(file)
        return object

    def plot_allocations(self, treatment_name="treatment"):
        data = []
        for patient_id in range(self.n_patients):
            patient_history = self.simulations[patient_id].history
            for index in range(len(patient_history)):
                observation = patient_history.observations[index]
                data.append(
                    {
                        "patient_id": patient_id,
                        "index": index,
                        **observation.treatment,
                        "debug_info": str(
                            self.simulations[patient_id].policy.debug_information[index]
                        ),
                        "context": str(observation.context),
                        "outcome": str(observation.outcome),
                        "counterfactual_outcomes": str(
                            observation.counterfactual_outcomes
                        ),
                    }
                )
        df = pd.DataFrame(data)
        return df.hvplot.heatmap(
            x="index",
            y="patient_id",
            C=treatment_name,
            hover_cols=[
                "debug_info",
                "context",
                "outcome",
                "counterfactual_outcomes",
                treatment_name,
            ],
            cmap="Category10",
            clim=(0, 10),
            grid=True,
        )
