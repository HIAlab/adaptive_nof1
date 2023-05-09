from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import matplotlib
import seaborn as sb


def get_member_as_list(list, member_name):
    return [getattr(item, member_name) for item in list]


Context = Dict[str, float]


@dataclass
class Treatment:
    i: int

    def __dict__(self):
        return {"treatment": self.i}


@dataclass
class Outcome(Dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def primary_outcome(self):
        return self["outcome"]


@dataclass
class Observation:
    context: Context
    treatment: Treatment
    outcome: Outcome


@dataclass
class History:
    observations: List[Observation]

    def __len__(self):
        return len(self.observations)

    def add_outcome(self, outcome):
        self.observations.append(outcome)

    def plot(self, ax) -> matplotlib.axis:
        primary_outcomes = [
            observation.outcome.primary_outcome() for observation in self.observations
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
                **observation.treatment.__dict__(),
                **observation.outcome,
            }
            for observation in self.observations
        ]
        return pd.DataFrame(dict_list)
