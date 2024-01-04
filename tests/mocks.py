
class MockBayesianModel:
    def __init__(
        self,
        treatment_name="treatment",
        outcome_name="outcome",
        max_probabilities=[0.5, 0.5],
        upper_confidence_bouns=[1, 1],
    ):
        self.trace = None
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name
        self.max_probabilities = max_probabilities
        self.upper_confidence_bounds = upper_confidence_bouns

    def get_upper_confidence_bounds(self, variable_name, epsilon: float = 0.05):
        return self.upper_confidence_bounds

    def approximate_max_probabilities(self, number_of_treatments, context):
        return self.max_probabilities

    def update_posterior(self, history, number_of_treatments):
        pass

