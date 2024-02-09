import pytest
from adaptive_nof1 import SeriesOfSimulationsRunner
from adaptive_nof1 import SimulationData
from adaptive_nof1.models.pill_model import PillModel
from adaptive_nof1.policies import CombinedPolicy, FixedPolicy
from adaptive_nof1.basic_types import History, Observation

NUMBER_OF_ACTIONS = 5
NUMBER_OF_PATIENTS = 5


@pytest.fixture
def fixed_policy():
    return CombinedPolicy(
        [
            FixedPolicy(number_of_actions=NUMBER_OF_ACTIONS),
        ],
        number_of_actions=NUMBER_OF_ACTIONS,
    )


@pytest.fixture
def runner(fixed_policy):
    runner = SeriesOfSimulationsRunner(
        model_from_patient_id=lambda patient_id: PillModel(patient_id=patient_id),
        n_patients=NUMBER_OF_PATIENTS,
        policy=fixed_policy,
    )
    return runner


@pytest.fixture
def simple_history():
    return History(
        observations=[
            Observation(
                context={"activity": 10},
                treatment={"treatment": 0},
                outcome={"outcome": 2},
                t=0,
                patient_id=0,
                debug_data={"means": [2, 0, 0]},
            ),
            Observation(
                context={"activity": 20},
                treatment={"treatment": 1},
                outcome={"outcome": 3},
                t=1,
                patient_id=0,
                debug_data={"means": [2, 3, 0]},
            ),
            Observation(
                context={"activity": 30},
                treatment={"treatment": 2},
                outcome={"outcome": 3},
                t=2,
                patient_id=0,
                debug_data={"means": [2, 3, 3]},
            ),
        ]
    )


@pytest.fixture
def simulation_data(simple_history):
    return SimulationData(
        history=simple_history,
        model="Model",
        policy="Policy",
        patient_id=0,
        additional_config={"expectations_of_interventions": [1, 2, 3]},
    )
