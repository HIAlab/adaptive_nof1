from abc import ABC, abstractmethod
from typing import Dict

from adaptive_nof1.basic_types import Observation, History


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_context(self, history: History) -> Dict:
        pass

    @abstractmethod
    def observe_outcome(self, action, context) -> Observation:
        pass
