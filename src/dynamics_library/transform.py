from enum import Enum
import numpy as np


class SupportedTransformMode(Enum):
    """Support transform methods

    Args:
        Enum (String): Transform method
    """
    AFFINE = 'affine'
    UNSCENTED = 'unscented'
    PARTICLE = 'particle'


class AffineTransformer:
    ...
