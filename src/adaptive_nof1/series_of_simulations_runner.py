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
    pooling: bool = False

    def __init__(
        self,
        model_from_patient_id: Callable[[int], Model],
        n_patients: int,
        policy,
        pooling=False,
    ):
        self.simulations = [
            SimulationRunner.from_model_and_policy_with_copy(
                model_from_patient_id(index),
                policy,
            )
            for index in range(n_patients)
        ]
        assert all_equal(
            [str(s.policy) for s in self.simulations]
        ), "Not all policies are the same. Usually, you need to set __str__() somewhere"
        assert all_equal(
            [str(s.model) for s in self.simulations]
        ), "Not all models are the same. Usually, you need to set __str__() somewhere"

        self.n_patients = n_patients
        self.model_from_patient_id = model_from_patient_id
        self.pooling = pooling

    def simulate(self, length) -> SeriesOfSimulationsData:
        for _ in progressbar(range(length)):
            for simulation in self.simulations:
                simulation.step()
            if self.pooling:
                histories = [simulation.history for simulation in self.simulations]
                pooled_history = History.fromListOfHistories(histories)
                for simulation in self.simulations:
                    simulation.pooledHistory = pooled_history

        return SeriesOfSimulationsData(
            simulations=[simulation.get_data() for simulation in self.simulations],
            configuration=self.configuration,
        )

    def clone_with_policy(self, new_policy):
        runner = SeriesOfSimulationsRunner(
            model_from_patient_id=self.model_from_patient_id,
            n_patients=self.n_patients,
            policy=new_policy,
        )
        return runner

    @property
    def configuration(self):
        return {
            "policy": str(self.simulations[0].policy),
            "model": str(self.simulations[0].model),
            "pooling": self.pooling,
        }


def simulate_configurations(configurations, length):
    calculated_series = []
    for configuration in configurations:
        result = SeriesOfSimulationsRunner(**configuration).simulate(length)
        calculated_series.append(
            {"configuration": result.configuration, "result": result}
        )

    config_to_simulation_data = {
        str(simulation.configuration): simulation
        for d in calculated_series
        for simulation in d["result"].simulations
    }
    return calculated_series, config_to_simulation_data
