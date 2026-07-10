# Association

`Association` matches a current object against previously observed objects. It
ranks the candidates with one or more weighted scorers and returns
the best candidate plus whether it passed the threshold. Downstream code can
then decide whether to merge the objects.

## Basic use

```python
from association import Association

association = Association(threshold=0.95)
result = association.associate(current_object, previous_objects)

if result["matched"]:
    previous_object = result["object"]
    # Decide what to do with the match.
```

Each scorer reads an attribute from those objects. By default,
`SemanticSimilarity` reads `object.image` and `SpatialSimilarity` reads
`object.position`. Your object class must provide those attributes, or you can
tell the scorers which attributes your class uses:

```python
from association import Association
from scorers import SemanticSimilarity, SpatialSimilarity

association = Association(
    threshold=0.8,
    scorers=[
        SemanticSimilarity(weight=0.6, image_attr="photo"),
        SpatialSimilarity(
            weight=0.4,
            sigma=3.0,
            position_attr="coordinates",
        ),
    ],
)
```

You can choose the scorers and weights yourself:

```python
from association import Association
from scorers import SemanticSimilarity

association = Association(
    threshold=0.8,
    scorers=[SemanticSimilarity(weight=1.0)],
)
```

## Add your own scorer

Subclass `AssociationScorer` and implement `score`. Scores should normally be
between `0.0` (not similar) and `1.0` (very similar).
For example:

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
Every scorer implements `score(current, previous)` for one pair of objects.
`Association` handles applying it to all previous objects.
