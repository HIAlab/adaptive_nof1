from __future__ import annotations

from adaptive_nof1.series_of_simulations_data import SeriesOfSimulationsData
from adaptive_nof1.metrics.metric import score_df
from adaptive_nof1.models.model import Model
from adaptive_nof1.simulation_runner import SimulationRunner
from adaptive_nof1.helpers import all_equal

from adaptive_nof1.basic_types import History


import pandas as pd

from dataclasses import dataclass
from typing import List, Callable, Dict

from tqdm.auto import tqdm as progressbar
import seaborn as sns
import panel
import hvplot.pandas  # noqa
import matplotlib.pyplot as plt


@dataclass
class SeriesOfSimulationsRunner:
    simulations: List[SimulationRunner]

    def __init__(
        self,
        model_from_patient_id: Callable[[int], Model],
        n_patients: int,
        policy,
    ):
        self.simulations = [
            SimulationRunner.from_model_and_policy_with_copy(
                model_from_patient_id(index),
                policy,
            )
            for index in range(n_patients)
        ]
        assert all_equal([str(s.policy) for s in self.simulations])
        assert all_equal([str(s.model) for s in self.simulations])

        self.n_patients = n_patients

    def simulate(self, length) -> SeriesOfSimulationsData:
        for _ in progressbar(range(length)):
            for simulation in self.simulations:
                simulation.step()

        return SeriesOfSimulationsData(
            simulations=[simulation.get_data() for simulation in self.simulations],
            configuration=self.configuration,
        )

    @property
    def configuration(self):
        return {
            "policy": str(self.simulations[0].policy),
            "model": str(self.simulations[0].model),
        }
