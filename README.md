# Association

`Association` matches a current object against a list of previously observed
objects. It ranks the candidates with one or more weighted scorers and returns
the best candidate plus whether it passed the threshold. Downstream code can
then decide whether to merge the objects.

## Basic use

```python
from association import Association

association = Association(threshold=0.95)
result = association.associate(current_object, previous_objects)

if result["matched"]:
    previous_object = result["object"]
    # Decide how to merge current_object into previous_object.
```

By default, association uses semantic similarity with weight `0.8` and spatial
similarity with weight `0.2`. Semantic similarity reads `img_embedding` (or
computes it from `segmented_rgb`), while spatial similarity reads `world_pos`.

You can choose the scorers and weights yourself:

```python
from association import Association
from scorers import SemanticSimilarity, SpatialSimilarity

association = Association(
    threshold=0.8,
    scorers=[
        SemanticSimilarity(weight=0.6),
        SpatialSimilarity(weight=0.4, sigma=3.0),
    ],
)
```

## Add your own scorer

Subclass `AssociationScorer` and implement `score`. Scores should normally be
between `0.0` (not similar) and `1.0` (very similar).

```python
from scorers import AssociationScorer


class LabelSimilarity(AssociationScorer):
    name = "label"

    def score(self, current, previous):
        return 1.0 if current.label == previous.label else 0.0


association = Association(
    threshold=0.8,
    scorers=[LabelSimilarity(weight=1.0)],
)
```

The scorer's `weight` controls how much it contributes to the final score.
`AssociationScorer` automatically applies `score` to every previous object;
for faster batch scoring, a scorer can override `score_many` instead.
