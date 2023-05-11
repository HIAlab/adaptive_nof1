from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.adaptive_nof1.simulation import Simulation

from sklearn.preprocessing import minmax_scale


class Metric(ABC):
    @abstractmethod
    def score(self, simulation: Simulation) -> float:
        pass

    def score_simulations(self, simulations: list[Simulation]):
        return [self.score(simulation) for simulation in simulations]

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


def score_df_iterative(
    simulations: list[Simulation], metrics, range, minmax_normalization=False
):
    iterative_metrics = pd.DataFrame()
    for end in range:
        sliced_simulations = [simulation[0:end] for simulation in simulations]
        sliced_metrics = score_df(sliced_simulations, metrics, minmax_normalization)
        sliced_metrics["t"] = end
        iterative_metrics = pd.concat([iterative_metrics, sliced_metrics])
    return iterative_metrics


def score_df(simulations: list[Simulation], metrics, minmax_normalization=False):
    scores = {str(metric): metric.score_simulations(simulations) for metric in metrics}
    df = pd.DataFrame(scores)
    df["Simulation"] = [str(simulation) for simulation in simulations]
    if minmax_normalization:
        for metric in metrics:
            df[str(metric)] = minmax_scale(df[str(metric)])
    return pd.melt(
        df,
        id_vars=["Simulation"],
        var_name="Metric",
        value_name="Score",
    )


