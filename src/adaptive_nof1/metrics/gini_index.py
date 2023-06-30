from __future__ import annotations

from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.simulation import Simulation

import numpy


class GiniIndex(Metric):
    def score(self, simulation: Simulation) -> float:
        return self.gini(simulation.history.to_df()[self.treatment_name])

    def __str__(self) -> str:
        return "Gini Index"

    # Taken from https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
    def gini(self, x, w=None):
        # The rest of the code requires numpy arrays.
        x = numpy.asarray(x)
        if w is not None:
            w = numpy.asarray(w)
            sorted_indices = numpy.argsort(x)
            sorted_x = x[sorted_indices]
            sorted_w = w[sorted_indices]
            # Force float dtype to avoid overflows
            cumw = numpy.cumsum(sorted_w, dtype=float)
            cumxw = numpy.cumsum(sorted_x * sorted_w, dtype=float)
            return numpy.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (
                cumxw[-1] * cumw[-1]
            )
        else:
            sorted_x = numpy.sort(x)
            n = len(x)
            cumx = numpy.cumsum(sorted_x, dtype=float)
            # The above formula, with all weights equal to 1 simplifies to:
            return (n + 1 - 2 * numpy.sum(cumx) / cumx[-1]) / n
