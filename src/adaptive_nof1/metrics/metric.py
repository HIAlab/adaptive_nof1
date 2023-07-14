from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import pandas as pd
import seaborn as sb
import numpy as np

if TYPE_CHECKING:
    from adaptive_nof1.simulation import Simulation

from sklearn.preprocessing import minmax_scale


class Metric(ABC):
    def __init__(self, outcome_name="outcome", treatment_name="treatment"):
        self.outcome_name = outcome_name
        self.treatment_name = treatment_name

    @abstractmethod
    def score(self, simulation: Simulation) -> List[float]:
        pass

    def score_simulations(self, simulations: list[Simulation]):
        df_list = [
            pd.DataFrame(
                {
                    "t": range(len(simulation.history)),
                    "score": self.score(simulation),
                    "simulation": str(simulation),
                    "patient_id": simulation.model.patient_id,
                    "model": str(simulation.model),
                    "policy": str(simulation.policy),
                }
            )
            for index, simulation in enumerate(simulations)
        ]

        return pd.concat(df_list)

    @abstractmethod
    def __str__(self) -> str:
        pass


def plot_score(simulations: list[Simulation], metrics, minmax_normalization=False):
    sb.barplot(
        data=score_df(simulations, metrics, minmax_normalization=minmax_normalization),
        x="Score",
        y="Simulation",
        hue="Metric",
    )


def score_df(simulations: list[Simulation], metrics, minmax_normalization=False):
    df_list = []
    scores = {str(metric): metric.score_simulations(simulations) for metric in metrics}
    for metric_name, metric_df in scores.items():
        metric_df["metric"] = metric_name
        df_list.append(metric_df)
    df = pd.concat(df_list)
    if minmax_normalization:
        for metric in metrics:
            df[str(metric)] = minmax_scale(df[str(metric)])
    return df
