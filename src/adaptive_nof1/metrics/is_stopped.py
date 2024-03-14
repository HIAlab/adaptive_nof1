from __future__ import annotations

from adaptive_nof1.metrics.metric import Metric

from typing import List


class IsStopped(Metric):
    def score(self, data: "SimulationData") -> List[float]:
        debug_data = data.history.debug_data()
        return [1 - d["is_stopped"] for d in debug_data]

    def __str__(self) -> str:
        return f"IsStopped"
