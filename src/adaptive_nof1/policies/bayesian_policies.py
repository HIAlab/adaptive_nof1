from adaptive_nof1.inference.bayes import GaussianAverageTreatmentEffect
from adaptive_nof1.policies.policy import Policy


class UpperConfidenceBound(Policy):
    def __init__(self, number_of_actions: int, epsilon: float):
        self.epsilon = epsilon
        self.inference = GaussianAverageTreatmentEffect()
        super().__init__(number_of_actions)

    def __str__(self):
        return f"UpperConfidenceBound: {self.epsilon} epsilon"

    def choose_best_action(self, history):
        outcome_groupby = history.to_df().groupby("treatment")["outcome"].mean()
        best_row = outcome_groupby.idxmin()
        return best_row

    def choose_action(self, history, _, block_length):
        self.inference.update_posterior(history, self.number_of_actions)
        bounds = (
            self.inference.get_upper_confidence_bounds("average_treatment_effect")
            .get("average_treatment_effect")
            .data
        )
        upper_bounds_array = [bound[1] for bound in bounds]
        self.debug_information += [
            f"Round {len(history)}: Upper Bounds Array: {upper_bounds_array}"
        ]
        return np.argmax(upper_bounds_array) + 1


class ThompsonSampling(Policy):
    def __init__(self, number_of_actions: int):
        self.inference = GaussianAverageTreatmentEffect()
        super().__init__(number_of_actions)

    def __str__(self):
        return f"ThompsonSampling\n"

    def choose_action(self, history, _, block_length):
        self.inference.update_posterior(history, self.number_of_actions)
        probability_array = self.inference.approximate_max_probabilities()
        action = (
            random.choices(range(self.number_of_actions), weights=probability_array)[0]
            + 1
        )
        self.debug_information += [
            f"Probabilities for picking: {probability_array}, chose {action}"
        ]
        return action
