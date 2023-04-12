from src.simulation import Model, Simulation
from src.observation import History
from src.policy import FixedPolicy


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
            "policy": FixedPolicy(number_of_actions=2),
        }
    )
    return simulation
