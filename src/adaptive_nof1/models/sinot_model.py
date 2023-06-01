import json
from dataclasses import dataclass, field

import numpy
import pandas as pd

from adaptive_nof1.basic_types import Observation, Treatment, Outcome
from adaptive_nof1.models.model import Model
from sinot.simulation import Simulation as SinotSimulation


@dataclass
class SinotModel(Model):
    parameter_file_path: str
    sinotSimulation: SinotSimulation = field(init=False)
    pat_complete: pd.DataFrame = field(init=False)
    patient_id: int
    days_per_period: int = 1
    patient_id_in_str = False

    def __post_init__(self):
        # Load example params
        with open(self.parameter_file_path) as fp:
            study_params = json.load(fp)

        self.sinotSimulation = SinotSimulation(
            study_params, random_generator=numpy.random.default_rng(self.patient_id)
        )
        self.pat_complete = SinotSimulation.empty_dataframe()
        self.days_per_period = 1

    def __str__(self):
        if self.patient_id_in_str:
            return f"SinotModel(id:{self.patient_id})"
        return f"SinotModel()"

    def generate_context(self):
        return {}

    def best_treatment(self):
        return 2

    def observe_outcome(self, action, context) -> Observation:
        action_to_treatment = {1: "Treatment_1", 2: "Treatment_2"}
        first_day = None if len(self.pat_complete) > 0 else "2018-01-01"
        self.pat_complete = self.sinotSimulation.step_patient(
            action_to_treatment[action],
            self.days_per_period,
            data=self.pat_complete,
            first_day=first_day,
        )
        last_row = self.pat_complete.iloc[-1]
        return Observation(
            **{
                "context": context,
                "treatment": Treatment(i=action),
                "outcome": {"outcome": -last_row["Uncertain_Low_Back_Pain"]},
            }
        )
