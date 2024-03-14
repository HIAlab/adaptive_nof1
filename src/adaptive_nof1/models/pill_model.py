from abc import ABC, abstractmethod

import numpy
from adaptive_nof1.models.model import Model


class PillModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rng = numpy.random.default_rng(self.patient_id)
        self.red_pill_slope = self.rng.normal(0, 1)

    @property
    def number_of_interventions(self):
        return 2

    def generate_context(self, history):
        return {"time_of_the_day": numpy.sin(len(history) / 24 / 3)}

    def observe_outcome(self, action, context):
        if action["treatment"] == 0:
            return {"enlightment": context["time_of_the_day"] * self.red_pill_slope}
        else:
            return {"enlightment": self.rng.normal(0.3, 1)}

    def __str__(self):
        return f"PillModel"
