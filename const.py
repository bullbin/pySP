from enum import Enum, auto

class QualityDemosaic(Enum):
    Draft   = auto()
    Fast    = auto()
    Best    = auto()

class PatternDemosaic(Enum):
    Rgbg    = auto()