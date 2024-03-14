from adaptive_nof1.helpers import index_to_values
from adaptive_nof1.models.model import Model
from adaptive_nof1.policies.policy import Policy


class FixedIndexedPolicy(Policy):
    def __init__(self, number_of_actions: int, inference):
        self.inference = inference
        super().__init__(number_of_actions)

    def __str__(self):
        return f"FixedIndexedPolicy"

    def choose_action(self, history, context, block_length=None):
        block_length = 1 if block_length is None else block_length
        self._debug_information += ["Fixed Indexed Schedule"]
        self.inference.update_posterior(history, self.number_of_actions)
        self.inference.approximate_max_probabilities(self.number_of_actions, context)
        self._debug_data.append(self.inference.debug_data)

        if context["t"] < self.number_of_actions:
            return {self.treatment_name: context["t"]}

        # context["patient_id"] will guarantee that the dimension array is long enough (but will be too long)
        dimensions = [self.number_of_actions] * context["t"]
        values = index_to_values(dimensions, context["patient_id"])
        # We do not have the correct length, so we just start from the other side
        values.reverse()

        return {self.treatment_name: values[context["t"] - self.number_of_actions]}
