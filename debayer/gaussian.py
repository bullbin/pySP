from enum import Enum
from typing import List, Tuple

import numpy as np

class BayerPatternPosition(Enum):
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3

def get_rgbg_kernel(kernel : np.ndarray, base_position : BayerPatternPosition) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-photosite low pass filter kernels for performing photosite-aware upsampling. Supports square Bayer
    patterns only, i.e., RGGB or RGBG.

    Args:
        kernel (np.ndarray): Input kernel. Must be square and of odd length. Typically Gaussian. Will be normalized by sum.
        base_position (BayerPatternPosition): Position of real photosite within one Bayer pattern arrangement.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Kernels corresponding to [TopLeft, TopRight, BottomLeft, BottomRight]
        photosites.
    """
    assert len(kernel.shape) == 2 or len(kernel.shape) == 3 and kernel.shape[2] == 1
    assert kernel.shape[0] == kernel.shape[1]
    assert kernel.shape[0] % 2 == 1

    is_base_left = base_position == BayerPatternPosition.TOP_LEFT or base_position == BayerPatternPosition.BOTTOM_LEFT
    is_base_bottom = base_position == BayerPatternPosition.BOTTOM_LEFT or base_position == BayerPatternPosition.BOTTOM_RIGHT

    output : List[np.ndarray] = []

    for idx in range(4):
        target = BayerPatternPosition(idx)
        is_left = target == BayerPatternPosition.TOP_LEFT or target == BayerPatternPosition.BOTTOM_LEFT
        is_bottom = target == BayerPatternPosition.BOTTOM_LEFT or target == BayerPatternPosition.BOTTOM_RIGHT
        
        base_kernel = kernel[0::2] if is_base_bottom == is_bottom else kernel[1::2]
        base_kernel = base_kernel[:, 0::2] if is_base_left == is_left else base_kernel[:, 1::2]
        if is_left != is_base_left:
            base_kernel = np.c_[base_kernel, np.zeros(base_kernel.shape[0])] if is_left else np.c_[np.zeros(base_kernel.shape[0]), base_kernel]
        if is_bottom != is_base_bottom:
            base_kernel = np.r_[np.zeros((1,base_kernel.shape[1])), base_kernel] if is_bottom else np.r_[base_kernel, np.zeros((1,base_kernel.shape[1]))]

        output.append(base_kernel / base_kernel.sum())
    
    return (output[0], output[1], output[2], output[3])