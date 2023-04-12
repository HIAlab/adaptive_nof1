from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class Context:
    c: List[float]


@dataclass
class Treatment:
    i: int


@dataclass
class Outcome:
    y: List[float]


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

    # TODO:
    def linear_coefficients(self):
        treatments = [observation.treatment for observation in observations]
