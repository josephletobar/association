import numpy as np

from .base import AssociationScorer


class SpatialSimilarity(AssociationScorer):
    """Compare positions stored on each object under ``position_attr``."""

    name = "spatial"

    def __init__(self, weight=1.0, sigma=2.0, position_attr="position"):
        super().__init__(weight=weight)
        self.sigma = sigma
        self.position_attr = position_attr

    def score(self, current, previous):
        current_position = np.asarray(
            getattr(current, self.position_attr), dtype=np.float32
        )
        previous_position = np.asarray(
            getattr(previous, self.position_attr), dtype=np.float32
        )
        distance = np.linalg.norm(previous_position - current_position)
        return float(np.exp(-distance / self.sigma))
