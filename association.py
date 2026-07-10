import cv2
import numpy as np
from scorers import SemanticSimilarity, SpatialSimilarity

class Association():
    """Match a current object against previously observed objects.

    Candidate matches are ranked using configurable, independently weighted
    scorers. The result identifies the best candidate and whether it meets the
    association threshold, allowing downstream code to decide whether to merge
    the current object with an existing object.
    """

    def __init__(self, threshold=0.95, scorers=None):

        self.scorers = scorers or [
            SemanticSimilarity(weight=0.8),
            SpatialSimilarity(weight=0.2),
        ]
        self.threshold = threshold

    def associate(self, current_object, previous_objects):
        if not previous_objects:
            return {
                "matched": False,
                "object": None,
                "score": 0.0,
                "details": [],
                "scores": np.array([], dtype=np.float32),
            }

        score_result = self._score_candidates(current_object, previous_objects)
        best_idx = int(np.argmax(score_result["scores"]))
        best_match = {
            "matched": bool(score_result["scores"][best_idx] >= self.threshold),
            "object": previous_objects[best_idx],
            "score": float(score_result["scores"][best_idx]),
            "details": [
                {
                    "name": detail["name"],
                    "score": float(detail["scores"][best_idx]),
                    "weight": detail["weight"],
                    "weighted_score": float(detail["weighted_scores"][best_idx]),
                }
                for detail in score_result["details"]
            ],
            "scores": score_result["scores"],
            "best_idx": best_idx,
        }

        return best_match

    def _score_candidates(self, current_object, previous_objects):
        total_scores = np.zeros(len(previous_objects), dtype=np.float32)
        total_weight = 0.0
        details = []

        for scorer in self.scorers:
            scores = np.array(
                [
                    scorer.score(current_object, previous_object)
                    for previous_object in previous_objects
                ],
                dtype=np.float32,
            )
            weight = float(scorer.weight)
            weighted_scores = scores * weight

            total_scores += weighted_scores
            total_weight += weight
            details.append({
                "name": scorer.name,
                "scores": scores,
                "weight": weight,
                "weighted_scores": weighted_scores,
            })

        final_scores = total_scores / total_weight if total_weight else total_scores
        return {
            "scores": final_scores,
            "details": details,
        }

    def _show_debug_comparison(self, new_object, existing_object, title, score_text):
        new_panel = self._fit_debug_image(new_object.segmented_rgb)
        existing_panel = self._fit_debug_image(existing_object.segmented_rgb)
        new_depth_panel = self._fit_debug_image(new_object.segmented_depth)
        existing_depth_panel = self._fit_debug_image(existing_object.segmented_depth)

        rgb_row = np.hstack((new_panel, existing_panel))
        depth_row = np.hstack((new_depth_panel, existing_depth_panel))
        combined = np.vstack((rgb_row, depth_row))

        cv2.putText(
            combined,
            f"new: {new_object.label}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.putText(
            combined,
            f"existing: {existing_object.node_id}",
            (372, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.putText(
            combined,
            score_text,
            (12, combined.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        cv2.putText(
            combined,
            "depth",
            (12, 388),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(title, combined)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 32:
                break
        cv2.destroyWindow(title)
