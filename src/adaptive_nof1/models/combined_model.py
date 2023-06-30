from typing import List

from adaptive_nof1.basic_types import Observation, History, Context
from adaptive_nof1.helpers import merge_with_postfix, split_with_postfix
from adaptive_nof1.models.model import Model


class CombinedModel(Model):
    def __init__(self, models: List[Model]):
        self.models = models

    def generate_context(self, history: History) -> Context:
        contexts = [model.generate_context(history) for model in self.models]
        return merge_with_postfix(contexts)

    def observe_outcome(self, action: List[int], context) -> Observation:
        contexts = split_with_postfix(context)
        outcomes = [
            model.observe_outcome(action, context)
            for action, model, context in zip(action, self.models, contexts)
        ]
        return merge_with_postfix(outcomes)

    def __str__(self):
        return f"CombinedModel({[str(model) for model in self.models]})"
