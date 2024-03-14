from __future__ import annotations

from adaptive_nof1.metrics.metric import Metric

from typing import List


class Length(Metric):
    def score(self, data: SimulationData) -> List[float]:
        df = data.history.to_df()
        return [len(df)] * len(df)

    def __str__(self) -> str:
        return f"Length"
