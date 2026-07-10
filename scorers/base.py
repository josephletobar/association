from abc import ABC, abstractmethod


class AssociationScorer(ABC):
    """Base class for scorers that compare attributes on candidate objects.

    Each scorer defines which attribute it needs. Objects passed to
    ``Association.associate`` must provide that attribute.
    """

    name = "association"

    def __init__(self, weight=1.0):
        self.weight = weight

    @abstractmethod
    def score(self, current, previous) -> float:
        """Score one current object against one previous object."""
        pass
