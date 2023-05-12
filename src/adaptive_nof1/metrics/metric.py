from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd
import seaborn as sb
import numpy as np

if TYPE_CHECKING:
    from adaptive_nof1.simulation import Simulation

from sklearn.preprocessing import minmax_scale


class Metric(ABC):
    @abstractmethod
    def score(self, simulation: Simulation) -> float:
        pass

    def score_simulations(self, simulations: list[Simulation]):
        df =  pd.DataFrame({i: self.score(simulation) for i, simulation in enumerate(simulations)})
        return pd.melt(df, var_name='simulation', value_name='score', ignore_index=False)

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
    import pdb; pdb.set_trace()
    df["Simulation"] = np.repeat([str(simulation) for simulation in simulations], len(simulations[0].history))
    if minmax_normalization:
        for metric in metrics:
            df[str(metric)] = minmax_scale(df[str(metric)])
    return df

