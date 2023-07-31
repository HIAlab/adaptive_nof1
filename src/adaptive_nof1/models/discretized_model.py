from adaptive_nof1.basic_types import History

from adaptive_nof1.models.model import Model


class DiscretizedModel(Model):
    def __init__(self, continuous_model, outcome_name="outcome", **kwargs):
        self.continuous_model = continuous_model
        self.outcome_name = outcome_name
        super().__init__(**kwargs)

    def generate_context(self, history: History):
        return self.continuous_model.generate_context(history)

    def observe_outcome(self, action, context):
        continuous_outcome_value = self.continuous_model.observe_outcome(
            action, context
        )[self.outcome_name]
        return {
            f"discrete_{self.outcome_name}": 1 if continuous_outcome_value > 0 else -1
        }

    def __str__(self):
        return f"DiscretizeModel({self.continuous_model})"
