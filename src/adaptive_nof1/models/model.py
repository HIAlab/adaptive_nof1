from abc import ABC, abstractmethod
from typing import Dict, List

from adaptive_nof1.basic_types import Observation, History, Context


def merge_with_postfix(dicts: List[Dict]) -> Dict:
    merged = {}
    for index, dict in enumerate(dicts):
        merged.update({f"{key}_{index}": value for key, value in dict.items()})
    return merged


def split_with_postfix(dict: Dict) -> List[Dict]:
    reconstruction = []
    for key, value in dict.items():
        name, index_string, *_ = key.split("_")
        index = int(index_string)
        while len(reconstruction) < index + 1:
            reconstruction += [{}]
        reconstruction[index].update({name: value})
    return reconstruction


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_context(self, history: History) -> Context:
        pass

    @abstractmethod
    def observe_outcome(self, action, context) -> Observation:
        pass


class CombinedModel(Model):
    def __init__(self, models: List[Model]):
        self.models = models

    def generate_context(self, history: History) -> Context:
        contexts = [model.generate_context(history) for model in self.models]
        return merge_with_postfix(contexts)

    def observe_outcome(self, action: List[int], context) -> Observation:
        outcomes = [
            model.observe_outcome(action, context)
            for action, model in zip(action, self.models)
        ]
        return merge_with_postfix(outcomes)
