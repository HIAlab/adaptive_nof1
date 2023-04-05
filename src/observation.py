from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class Observation:
    c: List[float]
    treatments: List[float]
    y: List[float]

    def linear_coefficients(self):
        treatments = self.treatments
        return pd.DataFrame(
            {
                "T_1": (treatments == 1) * 1,
                "T_2": (treatments == 2) * 1,
                "c_1": self.c * (treatments == 1),
                "c_2": self.c * (treatments == 2),
            }
        )

    def dataFrame(self):
        lc = self.linear_coefficients()
        lc["y"] = self.y
        return lc
