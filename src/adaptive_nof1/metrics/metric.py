from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import pandas as pd
import seaborn as sb
from adaptive_nof1.basic_types import History

if TYPE_CHECKING:
    from adaptive_nof1.simulation_data import SimulationData

from sklearn.preprocessing import minmax_scale


class Metric(ABC):
    def __init__(self, outcome_name="outcome", treatment_name="treatment"):
        self.outcome_name = outcome_name
        self.treatment_name = treatment_name

    @abstractmethod
    def score(self, data: SimulationData) -> List[float] | float:
        pass

    def score_simulations(self, simulations: List[SimulationData]):
        df_list = [
            pd.DataFrame(
                {
                    "t": [
                        observation.t for observation in simulation.history.observations
                    ],
                    "score": self.score(simulation),
                    "simulation": str(simulation),
                    "patient_id": simulation.patient_id,
                    "model": str(simulation.model),
                    "policy": str(simulation.policy),
                    "pooled": simulation.configuration["pooled"],
                }
            )
            for index, simulation in enumerate(simulations)
        ]

        return pd.concat(df_list)

    @abstractmethod
    def __str__(self) -> str:
        pass


class GetValue(Metric):
    def score(self, data: SimulationData):
        return data.history.to_df()[self.outcome_name]

    def __str__(self):
        return f"GetValue({self.outcome_name})"


def plot_score(simulations: list[SimulationData], metrics, minmax_normalization=False):
    sb.barplot(
        data=score_df(simulations, metrics, minmax_normalization=minmax_normalization),
        x="Score",
        y="Simulation",
        hue="Metric",
    )


def score_df(histories: list[History], metrics, minmax_normalization=False):
    df_list = []
    scores = {str(metric): metric.score_simulations(histories) for metric in metrics}
    for metric_name, metric_df in scores.items():
        metric_df["metric"] = metric_name
        df_list.append(metric_df)
    df = pd.concat(df_list)
    if minmax_normalization:
        for metric in metrics:
            df[str(metric)] = minmax_scale(df[str(metric)])
    return df
