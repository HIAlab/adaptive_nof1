from adaptive_nof1.basic_types import *


def outcomes_to_history(treatments, outcomes):
    assert len(treatments) == len(outcomes)
    return History(
        observations=[
            Observation(
                context={},
                treatment={"treatment": treatment},
                outcome={"outcome": outcome},
                t=t,
                patient_id=0,
            )
            for treatment, outcome, t in zip(
                treatments, outcomes, range(len(treatments))
            )
        ]
    )
