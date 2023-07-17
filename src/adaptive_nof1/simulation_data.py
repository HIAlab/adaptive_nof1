from adaptive_nof1.basic_types import History

from dataclasses import dataclass


@dataclass
class SimulationData:
    history: History
    model: str
    policy: str
    patient_id: int

    def __str__(self):
        return f"SimulationData"
