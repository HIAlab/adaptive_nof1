from adaptive_nof1.policies.policy import Policy

import numpy


class FixedPolicy(Policy):
    def __init__(self, inference_model=None, block_length=1, randomize=False, **kwargs):
        self.inference = inference_model
        self.block_length = block_length
        self.randomize = randomize
        self.treatment_sequence = None
        super().__init__(**kwargs)

    def __str__(self):
        return f"FixedPolicy"

    @property
    def additional_config(self):
        return {"inference": f"{self.inference}"}

    def choose_action(self, history, context):
        if self.treatment_sequence is None:
            self.treatment_sequence = list(range(self.number_of_actions))
            if self.randomize:
                numpy.random.default_rng().shuffle(self.treatment_sequence)

        if self.inference is not None:
            self.inference.update_posterior(history, self.number_of_actions)

            probability_array = self.inference.approximate_max_probabilities(
                self.number_of_actions, context
            )
            self._debug_information += [
                f"Fixed Schedule, hypothetical probabilities for picking: {numpy.array_str(numpy.array(probability_array), precision=2, suppress_small=True)}"
            ]
            debug_data_from_model = self.inference.debug_data
            self._debug_data.append(
                {**{"probabilities": probability_array}, **debug_data_from_model}
            )
        else:
            self._debug_information += ["Fixed Schedule"]
            self._debug_data.append({})
        return {
            self.treatment_name: self.treatment_sequence[
                (context["t"] // self.block_length) % self.number_of_actions
            ]
        }
