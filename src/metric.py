from src.simulation import Simulation
import seaborn as sb
from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt


class Metric(ABC):
    @abstractmethod
    def score(self, simulation: Simulation) -> float:
        pass

    def score_simulations(self, simulations: list[Simulation]):
        return [self.score(simulation) for simulation in simulations]

    @abstractmethod
    def __str__(self) -> str:
        pass


def plot_score(simulations: list[Simulation], metrics):
    sb.barplot(
        data=score_df(simulations, metrics), x="Score", y="Simulation", hue="Metric"
    )


def score_df(simulations: list[Simulation], metrics):
    scores = {str(metric): metric.score_simulations(simulations) for metric in metrics}
    return pd.melt(
        pd.DataFrame(
            scores, index=[str(simulation) for simulation in simulations]
        ).reset_index(names=["Simulation"]),
        id_vars="Simulation",
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
