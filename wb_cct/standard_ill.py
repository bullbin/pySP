from enum import IntEnum, auto
from typing import Dict, Tuple

# Source for all this info is https://en.wikipedia.org/wiki/Standard_illuminant
#     as of 22/1/25.
# Yes, citing a wiki is bad, the numbers seem fine!

class StandardIlluminant(IntEnum):
    A = auto()
    B = auto()
    C = auto()
    D50 = auto()
    D55 = auto()
    D65 = auto()
    D75 = auto()
    F1 = auto()
    F2 = auto()
    F3 = auto()
    F4 = auto()
    F5 = auto()

STANDARD_ILLUMINANT_TO_XY : Dict[StandardIlluminant, Tuple[float,float]] = {
    
    StandardIlluminant.A : (0.44758, 0.40745),
    StandardIlluminant.B : (0.34842, 0.35161),
    StandardIlluminant.C : (0.31006, 0.31616),
    StandardIlluminant.D50 : (0.34567, 0.35850),
    StandardIlluminant.D55 : (0.33242, 0.34743),
    StandardIlluminant.D65 : (0.31272, 0.32903),
    StandardIlluminant.D75 : (0.29902, 0.31485),
    StandardIlluminant.F1 : (0.31310, 0.33727),
    StandardIlluminant.F2 : (0.37208, 0.37529),
    StandardIlluminant.F3 : (0.40910, 0.39430),
    StandardIlluminant.F4 : (0.44018, 0.40329),
    StandardIlluminant.F5 : (0.31379, 0.34531)
}

LIGHTSOURCE_TO_STANDARD_ILLUMINANT : Dict[int, StandardIlluminant] = {
    12:StandardIlluminant.F1,
    13:StandardIlluminant.F5,
    14:StandardIlluminant.F2,
    15:StandardIlluminant.F3,
    16:StandardIlluminant.F4,
    17:StandardIlluminant.A,
    18:StandardIlluminant.B,
    19:StandardIlluminant.C,
    20:StandardIlluminant.D55,
    21:StandardIlluminant.D65,
    22:StandardIlluminant.D75,
    23:StandardIlluminant.D50
}

def get_chromacity_from_illuminant(ill : StandardIlluminant) -> Tuple[float,float]:
    """Get CIE 1931 chromacities for a standard illuminant.

    Args:
        ill (StandardIlluminant): Standard illuminant.

    Raises:
        KeyError: Raised if illuminant has no defined chromacity.

    Returns:
        Tuple[float,float]: (x,y) white point.
    """
    if ill in STANDARD_ILLUMINANT_TO_XY:
        return STANDARD_ILLUMINANT_TO_XY[ill]
    raise KeyError("Illuminant", StandardIlluminant.A.name, "has no defined chromacity value!")

def get_illuminant_from_lightsource(id : int) -> StandardIlluminant:
    """Get the equivalent StandardIlluminant enum member for an EXIF LightSource tag.

    Args:
        id (int): EXIF LightSource tag value.

    Raises:
        KeyError: Raised if LightSource has no mapping. Many IDs do not have a corresponding
        standard illuminant as they are vague so other EXIF tags may be needed to find either
        predicted CCT or chromacity.

    Returns:
        StandardIlluminant: Equivalent standard illuminant.
    """
    if id in LIGHTSOURCE_TO_STANDARD_ILLUMINANT:
        return LIGHTSOURCE_TO_STANDARD_ILLUMINANT[id]
    raise KeyError("ID", id, "either unimplemented or has no corresponding standard illuminant.")