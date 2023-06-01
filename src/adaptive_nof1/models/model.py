from abc import ABC, abstractmethod

from adaptive_nof1.basic_types import Observation, History, Context


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_context(self, history: History) -> Context:
        pass

    @abstractmethod
    def observe_outcome(self, action, context) -> Observation:
        pass
