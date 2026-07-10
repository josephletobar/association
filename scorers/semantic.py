import numpy as np

from .base import AssociationScorer


class SemanticSimilarity(AssociationScorer):
    """Compare images stored on each object under ``image_attr``."""

    name = "semantic"

    def __init__(self, weight=1.0, image_attr="image"):
        super().__init__(weight=weight)
        self.image_attr = image_attr

    def score(self, current, previous):
        current_embedding = self._get_embedding(current)
        previous_embedding = self._get_embedding(previous)
        current_embedding = np.asarray(current_embedding, dtype=np.float32).reshape(-1)
        previous_embedding = np.asarray(previous_embedding, dtype=np.float32).reshape(-1)
        current_norm = np.linalg.norm(current_embedding)
        previous_norm = np.linalg.norm(previous_embedding)
        if current_norm == 0 or previous_norm == 0:
            return 0.0

        similarity = np.dot(current_embedding, previous_embedding)
        similarity /= current_norm * previous_norm
        return float(np.clip(similarity, 0.0, 1.0))

    def _get_embedding(self, obj):
        from helpers import get_dino_embedding

        return get_dino_embedding(getattr(obj, self.image_attr))
