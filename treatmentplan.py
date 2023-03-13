from dataclasses import dataclass
from typing import List

import numpy as np

@dataclass
class TreatmentPlan:
    n_observations: int
    treatment_window_length: int
    treatment_order: List[int]
    
    def treatments(self):
        return np.repeat(self.treatment_order, self.n_observations / len(self.treatment_order))