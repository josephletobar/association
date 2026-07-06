from abc import ABC, abstractmethod

import numpy as np


class AssociationScorer(ABC):
    name = "association"

    def __init__(self, weight=1.0):
        self.weight = weight

    @abstractmethod
    def score(self, current, previous) -> float:
        pass

    def score_many(self, current, previous_objects):
        return np.array(
            [self.score(current, previous) for previous in previous_objects],
            dtype=np.float32,
        )
