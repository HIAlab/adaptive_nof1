from __future__ import annotations
import seaborn as sb
from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.simulation import Simulation

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


class SimpleRegret(Metric):
    def score(self, simulation: Simulation) -> float:
        return simulation.history.to_df()["outcome"].mean()

    def __str__(self) -> str:
        return "Simple Regret"


class FrequentistOptimalSelection(Metric):
    def score(self, simulation: Simulation) -> float:
        outcome_groupby = (
            simulation.history.to_df().groupby("treatment")["outcome"].mean()
        )
        best_mean = outcome_groupby.idxmin()
        return {True: 1.0, False: 0.0}[best_mean == simulation.model.best_treatment()]

    def __str__(self) -> str:
        return "Optimal Selection"
