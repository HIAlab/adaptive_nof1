from adaptive_nof1.basic_types import History
from adaptive_nof1.policies.fixed_policy import FixedPolicy
from adaptive_nof1.policies.block_policy import BlockPolicy
from adaptive_nof1.simulation import Simulation, Model


def load_basic_simulation():
    simulation = Simulation(
        **{
            "history": History(observations=[]),
            "model": Model(
                **{
                    "mu_p": 1,
                    "epsilon_i_sigma": 0.1,
                    "mu_T": [-2, -3],
                    "alpha": [-0.05, 0.05],
                    "c_sigma": 2,  # "Activity"
                }
            ),
            "policy": BlockPolicy(FixedPolicy(number_of_actions=2), block_length=5),
        }
    )
    return simulation
