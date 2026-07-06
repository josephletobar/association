import numpy as np

from .base import AssociationScorer


class SemanticSimilarity(AssociationScorer):
    name = "semantic"

    def __init__(
        self,
        weight=1.0,
        embedding_attr="img_embedding",
        image_attr="segmented_rgb",
        compute_missing=True,
    ):
        super().__init__(weight=weight)
        self.embedding_attr = embedding_attr
        self.image_attr = image_attr
        self.compute_missing = compute_missing

    def score(self, current, previous) -> float:
        return float(self.score_many(current, [previous])[0])

    def score_many(self, current, previous_objects):
        current_embedding = self._get_embedding(current)
        if current_embedding is None:
            return np.zeros(len(previous_objects), dtype=np.float32)

        current_embedding = np.asarray(current_embedding, dtype=np.float32).reshape(-1)
        current_norm = np.linalg.norm(current_embedding)
        if current_norm == 0:
            return np.zeros(len(previous_objects), dtype=np.float32)

        previous_embeddings = [
            self._get_embedding(previous)
            for previous in previous_objects
        ]
        valid_embeddings = [
            np.asarray(embedding, dtype=np.float32).reshape(-1)
            if embedding is not None
            else None
            for embedding in previous_embeddings
        ]

        scores = np.zeros(len(previous_objects), dtype=np.float32)
        valid_indices = [
            idx
            for idx, embedding in enumerate(valid_embeddings)
            if embedding is not None and np.linalg.norm(embedding) > 0
        ]
        if not valid_indices:
            return scores

        matrix = np.stack([valid_embeddings[idx] for idx in valid_indices])
        matrix_norms = np.linalg.norm(matrix, axis=1)
        similarities = matrix @ current_embedding
        similarities /= matrix_norms * current_norm
        scores[valid_indices] = np.clip(similarities, 0.0, 1.0)

        return scores

    def _get_embedding(self, obj):
        embedding = getattr(obj, self.embedding_attr, None)
        if embedding is not None:
            return embedding

        if not self.compute_missing:
            return None

        image = getattr(obj, self.image_attr, None)
        if image is None:
            return None

        from dino import get_dino_embedding

        embedding = get_dino_embedding(image)
        setattr(obj, self.embedding_attr, embedding)
        return embedding
