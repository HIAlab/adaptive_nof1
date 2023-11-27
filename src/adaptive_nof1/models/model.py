from abc import ABC, abstractmethod

from adaptive_nof1.basic_types import Observation, History, Context


class Model(ABC):
    def __init__(self, patient_id=None):
        self.patient_id = patient_id
        pass

    @property
    def additional_config(self):
        return {}

    @abstractmethod
    def generate_context(self, history: History) -> Context:
        pass

    @abstractmethod
    def observe_outcome(self, action, context) -> Observation:
        pass

    def reset(self):
        pass
