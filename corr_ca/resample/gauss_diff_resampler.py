# Resample R/B channels using green estimate
# Green diff to other channels should be lower frequency because of cross-correlation. Therefore,
#     resampling the difference and adding it to our green estimate (which started from a higher
#     resolution souce) should provide the cleanest result

import cv2
import numpy as np

from pySP.bayer_chan_mixer import bayer_to_rgbg, rgbg_to_bayer
from pySP.debayer.gaussian import BayerPatternPosition, get_rgbg_kernel

# TODO - This is the same resampler used in AHD. After some testing against bilinear, here's some results:
#        - This technique doesn't really add any major tint (after blurring, images are broadly similar)
#        - The appearance of B/R does look higher resolution
#        - The act of blurring B/R does end up removing some HF detail from source. It's not major but something that should be looked into.
#               Maybe shrink the kernel a bit.

# TODO - Find a better way to restore B/R extreme HF detail.
cv2_default_gauss = np.array([[1, 4, 6, 4,1],
                              [4,16,24,16,4],
                              [6,24,36,24,6],
                              [4,16,24,16,4],
                              [1, 4, 6, 4,1]])

def resample_b(b : np.ndarray, g_upscaled : np.ndarray):
    g_hf_cut = g_upscaled - cv2.GaussianBlur(g_upscaled, (3,3), 1.0)
    _g_r, _g_g1, g_b, _g_g2 = bayer_to_rgbg(g_upscaled)

    k_r, k_g, k_g2, k_b = get_rgbg_kernel(cv2_default_gauss, BayerPatternPosition.BOTTOM_RIGHT)

    g_b_upscaled = rgbg_to_bayer(cv2.filter2D(g_b, -1, k_r), cv2.filter2D(g_b, -1, k_g), cv2.filter2D(g_b, -1, k_b), cv2.filter2D(g_b, -1, k_g2)) + g_hf_cut

    b_diff = b - g_b

    b_upscaled = rgbg_to_bayer(cv2.filter2D(b_diff, -1, k_r), cv2.filter2D(b_diff, -1, k_g), cv2.filter2D(b_diff, -1, k_b), cv2.filter2D(b_diff, -1, k_g2)) + g_b_upscaled
    return b_upscaled

def resample_r(r : np.ndarray, g_upscaled : np.ndarray):
    g_hf_cut = g_upscaled - cv2.GaussianBlur(g_upscaled, (3,3), 1.0)
    g_r, _g_g1, _g_b, _g_g2 = bayer_to_rgbg(g_upscaled)

    k_r, k_g, k_g2, k_b = get_rgbg_kernel(cv2_default_gauss, BayerPatternPosition.TOP_LEFT)

    g_r_upscaled = rgbg_to_bayer(cv2.filter2D(g_r, -1, k_r), cv2.filter2D(g_r, -1, k_g), cv2.filter2D(g_r, -1, k_b), cv2.filter2D(g_r, -1, k_g2)) + g_hf_cut

    r_diff = r - g_r

    r_upscaled = rgbg_to_bayer(cv2.filter2D(r_diff, -1, k_r), cv2.filter2D(r_diff, -1, k_g), cv2.filter2D(r_diff, -1, k_b), cv2.filter2D(r_diff, -1, k_g2)) + g_r_upscaled
    return r_upscaled