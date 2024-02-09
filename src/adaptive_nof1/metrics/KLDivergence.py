from __future__ import annotations
from re import S
from typing import List

from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.simulation_data import SimulationData
from adaptive_nof1.helpers import flatten_dictionary

import torch


class KLDivergence(Metric):
    def __init__(
        self, data_to_true_distribution, debug_data_to_posterior_distribution, **kwargs
    ):
        self.data_to_true_distribution = data_to_true_distribution
        self.debug_data_to_posterior_distribution = debug_data_to_posterior_distribution
        super().__init__(**kwargs)

    def score(self, data: SimulationData) -> List[float]:
        debug_data = data.history.debug_data()
        scores = [
            torch.distributions.kl_divergence(
                self.debug_data_to_posterior_distribution(d),
                self.data_to_true_distribution(data),
            ).item()
            for d in debug_data
        ]

        return scores

    def __str__(self) -> str:
        return "KL Divergence"
