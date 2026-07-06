import numpy as np

from .base import AssociationScorer


class SpatialSimilarity(AssociationScorer):
    name = "spatial"

    def __init__(self, weight=1.0, sigma=2.0, position_attr="world_pos"):
        super().__init__(weight=weight)
        self.sigma = sigma
        self.position_attr = position_attr

    def score(self, current, previous) -> float:
        return float(self.score_many(current, [previous])[0])

    def score_many(self, current, previous_objects):
        current_pos = getattr(current, self.position_attr, None)
        if current_pos is None:
            return np.zeros(len(previous_objects), dtype=np.float32)

        current_pos = np.asarray(current_pos, dtype=np.float32)
        previous_positions = [
            getattr(previous, self.position_attr, None)
            for previous in previous_objects
        ]

        scores = np.zeros(len(previous_objects), dtype=np.float32)
        valid_indices = [
            idx
            for idx, position in enumerate(previous_positions)
            if position is not None
        ]
        if not valid_indices:
            return scores

        position_matrix = np.asarray(
            [previous_positions[idx] for idx in valid_indices],
            dtype=np.float32,
        )
        distances = np.linalg.norm(position_matrix - current_pos, axis=1)
        scores[valid_indices] = np.exp(-distances / self.sigma)

        return scores
