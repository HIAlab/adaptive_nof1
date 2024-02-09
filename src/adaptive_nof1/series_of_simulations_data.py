from __future__ import annotations

from dataclasses import dataclass
from typing import List, Callable
from adaptive_nof1.basic_types import History
from adaptive_nof1.metrics import score_df
from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.simulation_data import SimulationData

import holoviews
import seaborn
import pandas
import pickle
import numpy
import bokeh


@dataclass
class SeriesOfSimulationsData:
    configuration: dict
    simulations: List[SimulationData]

    def pooled_histories(self):
        histories = [simulation.history for simulation in self.simulations]
        pooled_history = History.fromListOfHistories(histories)
        return pooled_history

    @staticmethod
    def score_data(
        list_of_series: List[SeriesOfSimulationsData],
        metrics,
        renaming: dict[str, Callable] = {},
    ):
        scored_df = score_df(
            [
                simulation
                for series in list_of_series
                for simulation in series.simulations
            ],
            metrics,
            minmax_normalization=False,
        )
        for key, function in renaming.items():
            scored_df[key] = scored_df[key].apply(function)
        return scored_df

    @staticmethod
    def plot_bar(
        list_of_series: List[SeriesOfSimulations],
        metrics,
        y="simulation",
        hue="policy_#_metric",
        simulation_naming=lambda x: x,
    ):
        scored_df = score_df(
            [
                simulation
                for series in list_of_series
                for simulation in series.simulations
            ],
            metrics,
            minmax_normalization=False,
        )
        scored_df["policy_#_metric"] = scored_df["policy"] + "_#_" + scored_df["metric"]
        scored_df[y] = scored_df[y].apply(simulation_naming)
        # Select only rows at the end of the trial
        filterd_score_df = scored_df[scored_df["t"] == scored_df["t"].max()]
        ax = seaborn.boxplot(
            data=filterd_score_df,
            x="score",
            y=y,
            hue=hue,
        )
        seaborn.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    @staticmethod
    def plot_lines(
        list_of_series: List[SeriesOfSimulationsData],
        metrics: List[Metric],
        hue: str = "policy_#_metric_#_model_p",
        process_df=lambda x: x,
        legend_position=None,
    ):
        scored_df = score_df(
            [
                simulation
                for series in list_of_series
                for simulation in series.simulations
            ],
            metrics,
            minmax_normalization=False,
        )
        scored_df["policy_#_metric_#_model_p"] = (
            scored_df["policy"]
            + "_#_"
            + scored_df["metric"]
            + "_#_"
            + scored_df["model"]
            + "_"
            + scored_df["pooled"].astype(str)
        )
        ax = seaborn.lineplot(
            data=process_df(scored_df),
            x="t",
            y="score",
            hue=hue,
            # units="patient_id",
            # estimator=numpy.median,
            # errorbar=lambda x: (numpy.quantile(x, 0.25), numpy.quantile(x, 0.75)),
        )
        ax.set(xlabel="t", ylabel="Regret")
        if not legend_position:
            legend_position = (0, 1 + 0.1)
        seaborn.move_legend(
            ax, "upper left", title=None, bbox_to_anchor=legend_position
        )
        return ax

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

    def draw_legend(self, min, max):
        colormap = holoviews.Palette("Category10", samples=10).values
        legend_items = []
        width = 2
        height = 2
        for treatment in range(min, max):
            x = 0
            y = (treatment - min) * height
            xx = width
            yy = y + height
            rect = holoviews.Rectangles(
                [(x, y, xx, yy, colormap[treatment - min])], vdims="value"
            )
            rect.opts(color="value")
            label = holoviews.Text(
                width + 0.9,
                (treatment - min + 0.5) * height,
                f"{treatment}",
                halign="center",
                valign="center",
            )
            legend_items.append(rect * label)

        return holoviews.Overlay(legend_items).opts(
            xaxis=None,
            yaxis=None,
            width=200,
            height=200,
            show_frame=False,
            toolbar=None,
            xlim=(-0.2, 10),
            ylim=(0, (max - min) * height + 5),
        )

    def plot_allocations(self, treatment_name="treatment", offset=1):
        data = []
        for patient_id in range(len(self.simulations)):
            patient_history = self.simulations[patient_id].history
            for index in range(len(patient_history)):
                observation = patient_history.observations[index]
                data.append(
                    {
                        "i": patient_id + offset,
                        "index": index + offset,
                        treatment_name: observation.treatment[treatment_name] + offset,
                        "color_index": observation.treatment[treatment_name],
                        "debug_info": str(
                            self.simulations[patient_id].history.debug_information()[
                                index
                            ]
                        ),
                        "debug_data": str(
                            self.simulations[patient_id].history.debug_data()[index]
                        ),
                        "context": str(observation.context)[0:50],
                        "outcome": str(observation.outcome),
                        "counterfactual_outcomes": str(
                            observation.counterfactual_outcomes
                        )[0:50],
                        "j": observation.t + 1,
                    }
                )

        df = pandas.DataFrame(data)
        plot = df.hvplot.heatmap(
            x="j",
            y="i",
            C="color_index",
            hover_cols=[
                "debug_info",
                "context",
                "outcome",
                "counterfactual_outcomes",
                "debug_data",
                treatment_name,
            ],
            cmap="Category10",
            clim=(0, 9),
            grid=True,
            colorbar=False,
        )
        return plot

    def __getitem__(self, index) -> SeriesOfSimulationsData | SimulationData:
        if isinstance(index, slice):
            simulations = [
                self.simulations[i]
                for i in range(*index.indices(len(self.simulations)))
            ]
        else:
            simulations = [self.simulations[index]]
        return SeriesOfSimulationsData(
            configuration=self.configuration,
            simulations=simulations,
        )


def plot_allocations_for_calculated_series(calculated_series):
    panels = []
    for series in calculated_series:
        min, max = (
            series["result"].pooled_histories().to_df()["treatment"].agg(["min", "max"])
        )
        panels.append(
            series["result"].plot_allocations()
            + series["result"].draw_legend(min + 1, max + 2)
        )
    for panel, i in zip(panels, range(len(panels))):
        panel.opts(
            title=f"{calculated_series[i]['configuration']['policy']}, {calculated_series[i]['configuration']['model']}",
            fontsize={"title": "80%"},
        )
    return holoviews.Layout(panels).cols(2)
