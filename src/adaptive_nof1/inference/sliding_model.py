# TODO: create abstract base class for model


class SlidingModel:
    def __init__(self, sliding_window_size=None, model=None, **kwargs):
        assert sliding_window_size is not None, "You must specify a sliding window size"
        assert model is not None, "You must specify an underlying model to use"
        self.sliding_window_size = sliding_window_size
        self.model = model
        super().__init__(**kwargs)

    def update_posterior(self, history, number_of_treatments):
        return self.model.update_posterior(
            history[-self.sliding_window_size :], number_of_treatments
        )

    def __str__(self):
        return f"SlidingWindow({self.sliding_window_size}, {self.model})"

    def approximate_max_probabilities(self, number_of_treatments, context):
        return self.model.approximate_max_probabilities(number_of_treatments, context)
