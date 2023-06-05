from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

import matplotlib
import pandas as pd
import seaborn as sb


def get_member_as_list(list, member_name):
    return [getattr(item, member_name) for item in list]


Context = Dict[str, float]


Treatment = Dict[str, float]

Outcome = Dict[str, float]


@dataclass
class Observation:
    context: Context
    treatment: Treatment
    outcome: Outcome
    counterfactual_outcomes: List[Outcome] = field(default_factory=lambda: [])


@dataclass
class History:
    observations: List[Observation]

    def __len__(self):
        return len(self.observations)

    def add_observation(self, observation):
        self.observations.append(observation)

    def plot(self, ax) -> matplotlib.axis:
        primary_outcomes = [
            observation.outcome.outcome for observation in self.observations
        ]
        treatments = [observation.treatment.i for observation in self.observations]
        df = pd.DataFrame(
            data={"primary_outcomes": primary_outcomes, "treatments": treatments}
        )
        return sb.scatterplot(
            df, x=df.index, y="primary_outcomes", ax=ax, hue="treatments"
        )

    def to_df(self):
        if len(self.observations) == 0:
            return pd.DataFrame(columns=["treatment", "outcome"])

        dict_list = [
            {
                **observation.context,
                **observation.treatment,
                **observation.outcome,
            }
            for observation in self.observations
        ]
        return pd.DataFrame(dict_list)

    def __getitem__(self, index) -> History:
        if isinstance(index, slice):
            return History(
                observations=[
                    self.observations[i]
                    for i in range(*index.indices(len(self.observations)))
                ]
            )
        else:
            return History(observations=[self.observations[index]])
