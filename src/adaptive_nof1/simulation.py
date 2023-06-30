import copy
import dataclasses
from dataclasses import dataclass

import matplotlib.pyplot as plt

from adaptive_nof1.basic_types import History, Treatment, Observation
from adaptive_nof1.models.model import Model
from adaptive_nof1.policies.policy import Policy


@dataclass
class Simulation:
    history: History
    policy: Policy
    model: Model

    @staticmethod
    def from_model_and_policy_with_copy(model: Model, policy: Policy):
        return Simulation(
            history=History(observations=[]),
            model=copy.deepcopy(model),
            policy=copy.deepcopy(policy),
        )

    @staticmethod
    def from_model_and_policy(model: Model, policy: Policy):
        return Simulation(history=History(observations=[]), model=model, policy=policy)

    def plot(self):
        axes = plt.axes()
        self.history.plot(axes)
        plt.title(str(self.policy) + str(self.model))

    def __str__(self):
        return f"Simulation[{self.policy},{self.model}]"

    def step(self):
        context = self.model.generate_context(self.history)
        action = self.policy.choose_action(self.history, context)
        outcome = self.model.observe_outcome(action, context)
        counterfactual_outcomes = [
            self.model.observe_outcome(counterfactual_action, context)
            for counterfactual_action in self.policy.available_actions()
        ]
        observation = Observation(
            **{
                "context": context,
                "treatment": action,
                "outcome": outcome,
                "counterfactual_outcomes": counterfactual_outcomes,
            }
        )
        self.history.add_observation(observation)
        return self

    def __getitem__(self, index):
        return dataclasses.replace(self, history=self.history[index])
