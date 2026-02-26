from typing import List, Tuple
import cv2
import numpy as np
from pySP.bayer_chan_mixer import bayer_to_rgbg, rgbg_to_bayer
from pySP.image import RawRgbgData

def compute_structural_instability(image : RawRgbgData) -> np.ndarray:
    # Compute simple WB using stored WB. Recommended to set to good value prior to everything
    wb_coeff = image.cam_wb.get_reciprocal_multipliers()

    # Paper - apply wb coeff NOW
    r,g0,b,g1 = bayer_to_rgbg(np.copy(image.bayer_data_scaled))
    r *= wb_coeff[0]
    g0 *= wb_coeff[1]
    g1 *= wb_coeff[1]
    b *= wb_coeff[2]

    # Pad bayer data to give enough pixels, furthest outreach is 3 but pad 4 to keep pixel structure
    bayer_data_padded = cv2.copyMakeBorder(rgbg_to_bayer(r,g0,b,g1), 4,4,4,4, cv2.BORDER_REFLECT)

    def compute_structural_instability_bayer(b_arr : np.ndarray, offsets : List[Tuple[int,int]], b_pad : int = 4, bayer_offset : Tuple[int,int] = (0,0)) -> np.ndarray:

        stack = []

        max_y = (b_arr.shape[0] - (b_pad * 2)) // 2
        max_x = (b_arr.shape[1] - (b_pad * 2)) // 2

        for offset in offsets:
            x_start, y_start = offset
            x_start = x_start + b_pad + bayer_offset[0]
            y_start = y_start + b_pad + bayer_offset[1]

            # Skip over every other subpixel because of 2x2 structure
            masked = b_arr[y_start::2,x_start::2][:max_y,:max_x]
            stack.append(masked)

        stack = np.dstack(stack)
        return np.max(stack, axis=2) - np.min(stack, axis=2)

    rr_r = compute_structural_instability_bayer(bayer_data_padded, [(0,0), (0,-2), (0,2), (-2,0), (2,0)], bayer_offset=(0,0))    # good
    rr_g = compute_structural_instability_bayer(bayer_data_padded, [(-1,0),(1,0),(0,-1),(0,1)], bayer_offset=(0,0))
    rr_b = compute_structural_instability_bayer(bayer_data_padded, [(-1,-1),(1,-1),(1,1),(-1,1)], bayer_offset=(0,0))

    g0_r = compute_structural_instability_bayer(bayer_data_padded, [(-1,0),(-1,-2),(-1,2),(1,-2),(1,0),(1,2)], bayer_offset=(1,0))
    g0_g = compute_structural_instability_bayer(bayer_data_padded, [(0,0),(-1,-1),(-1,1),(1,-1),(1,1)], bayer_offset=(1,0))
    g0_b = compute_structural_instability_bayer(bayer_data_padded, [(0,-1),(0,1),(-2,-1),(-2,1),(2,-1),(2,1)], bayer_offset=(1,0))

    g1_r = compute_structural_instability_bayer(bayer_data_padded, [(0,-1),(-2,-1),(2,-1),(0,1),(-2,1),(2,1)], bayer_offset=(0,1))
    g1_g = compute_structural_instability_bayer(bayer_data_padded, [(0,0),(-1,1),(1,1),(-1,-1),(1,-1)], bayer_offset=(0,1))
    g1_b = compute_structural_instability_bayer(bayer_data_padded, [(-1,0),(1,0),(-1,-2),(1,-2),(-1,2),(1,2)], bayer_offset=(0,1))

    b_r = compute_structural_instability_bayer(bayer_data_padded, [(-1,-1),(1,-1),(-1,1),(1,1)], bayer_offset=(1,1))
    b_g = compute_structural_instability_bayer(bayer_data_padded, [(-1,0),(1,0),(0,-1),(0,1)], bayer_offset=(1,1))
    b_b = compute_structural_instability_bayer(bayer_data_padded, [(0,0),(-2,0),(2,0),(0,-2),(0,2)], bayer_offset=(1,1))

    si_r = rgbg_to_bayer(rr_r, g0_r, b_r, g1_r)
    si_g = rgbg_to_bayer(rr_g, g0_g, b_g, g1_g)
    si_b = rgbg_to_bayer(rr_b, g0_b, b_b, g1_b)

    return np.dstack((si_r, si_g, si_b))