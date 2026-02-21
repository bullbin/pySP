from typing import Optional, Tuple, Union

import cv2
import numpy as np

from pySP.base_types.image_base import RawDebayerData, RawRgbgData_BaseType
from pySP.bayer_chan_mixer import bayer_to_rgbg, rgbg_to_bayer
from pySP.debayer.gaussian import CV2_DEFAULT_KERNEL_SIGMA, CV2_DEFAULT_UNNORM_GAUSSIAN_KERNEL, BayerPatternPosition, get_rgbg_kernel

def simple_delta_mix_bilinear_kernel(top : np.ndarray, bottom : np.ndarray, left : np.ndarray, right : np.ndarray) -> np.ndarray:
    """A simple bilinear in-fill kernel for interpolating a pixel surrounded by 4 cardinal pixels.

    This kernel attempts to preserve directionality of the pixels by modifying the bilinear weights. If there is more change in
    the up-down direction, the kernel will prefer interpolating the left-right direction to preserve that contrast (and vice
    versa).

    Args:
        top (np.ndarray): Pixel above new sample
        bottom (np.ndarray): Pixel below new sample
        left (np.ndarray): Pixel to left of new sample
        right (np.ndarray): Pixel to right of new sample

    Returns:
        np.ndarray: Interpolated new samples
    """
    assert top.shape == bottom.shape == left.shape == right.shape

    # To try to fix smoothness we'll attempt to follow the local gradient using the neighbours. We don't have much width to use.
    #    T     If LR difference is bigger than TB, there's probably some edge or detail travelling through TB.
    #  L . R   We'll therefore blend more in the TB direction to try to preserve directionality (rotate this for LR).
    #    B     TODO - We have full soft edge map from Bayer instability. We could make some modifications to make it directional.

    # TODO - Experiment with blending part of pure bilinear back in, where overall delta is quite low sometimes this could be
    #            used to avoid adding cross artifact (mazes).

    delta_y = np.abs(top - bottom)
    delta_x = np.abs(left - right)
    sum_delta = delta_y + delta_x

    # Ultimately the weights are always constrained with XY axis, we are still sampling pixel center. Pre-weight axis now.
    avg_x = (left + right) / 2
    avg_y = (top + bottom) / 2

    # Prevent divide by zero - where there is no change in the neighbourhood, weight as equal. Doesn't really matter.
    # TODO - Check if we need to transform the weight, it is like an angle. In practice it seems to look fine though.
    strength_y = np.divide(delta_y, sum_delta, out=np.ones_like(sum_delta) * 0.5, where=sum_delta != 0)
    strength_x = 1 - strength_y

    return avg_y * strength_x + avg_x * strength_y

def resample_g_to_full_resolution(g1 : np.ndarray, g2 : np.ndarray, use_bilinear_weighting : bool = True) -> np.ndarray:
    """Resample G using bilinear interpolation to get a full-resolution channel. This is only compatible with RGGB arrays.

    This method is not intended for demosaicing and will produce a softer resolve because it does not use any channel
    cross-correlation (shared features between channels) to reconstruct features in missing pixels. Instead, it aims
    to interpolate the channel without needing to blend or modify the original green data.

    Bilinear weighting is provided as a quality enhancement to improve edge flow in the output. Where bilinear interpolation
    considers each neighbour equally, our weighting approach tries to interpolate in such a way that horizontal edges
    are interpolated horizontally, avoiding excessive blurring of local detail (and vice versa). Note that this is not the
    same as approaches like AHD which evaluates and recorrects local smoothness - this is strictly a kernel shaping trick
    during resampling.

    The output of this will have the same shape as the sensor input. Padding is used to generate valid data for edge pixels.

    Args:
        g1 (np.ndarray): G pixel, top-right
        g2 (np.ndarray): G pixel, bottom-left
        use_bilinear_weighting (bool, optional): Adjust kernel to try to follow edges. <b>Recommended, no real downside</b>. Defaults to True.

    Returns:
        np.ndarray: Resampled G channel at input sensor resolution.
    """

    assert g1.shape == g2.shape

    # R/B are half the resolution of G. To avoid killing resolution during resampling, we'll resample (R/B - G) which should be lower
    #     frequency because of channel correlation so we won't damage the details as much (this the principle of AHD, for example).
    # To do this, we need to get full resolution G without using contaminated R/B.

    # TODO - We could try a AHD-like approach, interpolating H/V using resampled (undistorted) R/B and smoothing after. Maybe more details
    #        in green could be extracted. This method already produces very smooth G, maybe cross correlation is overkill. Also a
    #        little worried about adding microcontrast in R/B if we taint G with R/B detail when using G delta for correcting R/B CA.

    # Pad both channels so we don't need to think about out of bounds
    g1_padded = cv2.copyMakeBorder(g1, 1,1,1,1, cv2.BORDER_REFLECT)
    g2_padded = cv2.copyMakeBorder(g2, 1,1,1,1, cv2.BORDER_REFLECT)

    #  r g1 |  r g1 |  r g1
    # g2  b | g2  b | g2  b
    # ----------------------
    #  r g1 |  r g1 |  r g1    For better visualization, when indexing we are targetting the center pixel corresponding to the
    # g2  b | g2  b | g2  b    top-left of the original. We have padded a 1px edge to hide indexing issues.
    # ----------------------
    #  r g1 |  r g1 | r g1
    # g2  b | g2  b | g2  b

    # When indexing blue, the bottom and right edges do not have enough pixels.
    b_t = g1_padded[1:-1, 1:-1]
    b_b = g1_padded[2:,   1:-1]
    b_l = g2_padded[1:-1, 1:-1]
    b_r = g2_padded[1:-1, 2:  ]

    # When indexing red, the top and left edges do not have enough pixels.
    r_t = g2_padded[ :-2, 1:-1]
    r_b = g2_padded[1:-1, 1:-1]
    r_l = g1_padded[1:-1,  :-2]
    r_r = g1_padded[1:-1, 1:-1]

    # When no strategy is used, fallback to pure bilinear sampling. Average the neighbours.
    # This works but you can see a slight structural pattern in the output. It's not pleasant noise.
    if not(use_bilinear_weighting):
        r = (r_t + r_b + r_l + r_r) / 4
        b = (b_t + b_b + b_l + b_r) / 4
        return rgbg_to_bayer(r, g1, b, g2)
    
    # An alternative strategy weighs the local structure to favor sampling along the gradient. This stretches
    #     the edges a bit through the missing data, effectively hiding the checkerboard pattern.
    # This doesn't reconstruct any detail like demosaicing but makes the output more usable.
    r = simple_delta_mix_bilinear_kernel(r_t, r_b, r_l, r_r)
    b = simple_delta_mix_bilinear_kernel(b_t, b_b, b_l, b_r)

    # The output will always retain source data. Hopefully the interpolation looks good.
    return rgbg_to_bayer(r, g1, b, g2)

def resample_channel(subpixel : np.ndarray, g_at_subpixel : np.ndarray, g_hf_pass : np.ndarray, bayer_position : BayerPatternPosition) -> np.ndarray:
    """Resample filtered pixel to full resolution by upscaling green difference using Gaussian filtering.

    Args:
        subpixel (np.ndarray): Low-res channel, no gaps.
        g_at_subpixel (np.ndarray): Green channel at corresponding position to low-res channel.
        g_hf_pass (np.ndarray): Full resolution green channel with high pass filter, sigma = 1.0 (match default kernel).
        bayer_position (BayerPatternPosition): Position corresponding to input subpixel in Bayer array.

    Returns:
        np.ndarray: Resampled channel.
    """
    assert subpixel.shape == g_at_subpixel.shape

    k_r, k_g, k_g2, k_b = get_rgbg_kernel(CV2_DEFAULT_UNNORM_GAUSSIAN_KERNEL, bayer_position)
    g_channel_upscaled = rgbg_to_bayer(cv2.filter2D(g_at_subpixel, -1, k_r), cv2.filter2D(g_at_subpixel, -1, k_g), cv2.filter2D(g_at_subpixel, -1, k_b), cv2.filter2D(g_at_subpixel, -1, k_g2)) + g_hf_pass
    channel_diff = subpixel - g_at_subpixel
    return rgbg_to_bayer(cv2.filter2D(channel_diff, -1, k_r), cv2.filter2D(channel_diff, -1, k_g), cv2.filter2D(channel_diff, -1, k_b), cv2.filter2D(channel_diff, -1, k_g2)) + g_channel_upscaled

def resample_rb(r : np.ndarray, b : np.ndarray, g_upscaled : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Resample R and B to full resolution by upscaling green difference using Gaussian filtering.

    Args:
        r (np.ndarray): Low-res red channel, no gaps.
        b (np.ndarray): Low-res blue channel, no gaps.
        g_upscaled (np.ndarray): Full resolution green channel.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Resampled red channel, resampled blue channel)
    """
    g_hf_cut = g_upscaled - cv2.GaussianBlur(g_upscaled, (3,3), CV2_DEFAULT_KERNEL_SIGMA)
    g_r, _g_g1, g_b, _g_g2 = bayer_to_rgbg(g_upscaled)
    return (resample_channel(r, g_r, g_hf_cut, BayerPatternPosition.TOP_LEFT), resample_channel(b, g_b, g_hf_cut, BayerPatternPosition.BOTTOM_RIGHT))

def resample_b(b : np.ndarray, g_upscaled : np.ndarray) -> np.ndarray:
    """Resample B to full resolution by upscaling green difference using Gaussian filtering.

    Args:
        b (np.ndarray): Low-res blue channel, no gaps.
        g_upscaled (np.ndarray): Full resolution green channel.

    Returns:
        np.ndarray: Resampled blue channel.
    """
    g_hf_cut = g_upscaled - cv2.GaussianBlur(g_upscaled, (3,3), CV2_DEFAULT_KERNEL_SIGMA)
    _g_r, _g_g1, g_b, _g_g2 = bayer_to_rgbg(g_upscaled)
    return resample_channel(b, g_b, g_hf_cut, BayerPatternPosition.BOTTOM_RIGHT)

def resample_r(r : np.ndarray, g_upscaled : np.ndarray) -> np.ndarray:
    """Resample R to full resolution by upscaling green difference using Gaussian filtering.

    Args:
        r (np.ndarray): Low-res red channel, no gaps.
        g_upscaled (np.ndarray): Full resolution green channel.

    Returns:
        np.ndarray: Resampled red channel.
    """
    g_hf_cut = g_upscaled - cv2.GaussianBlur(g_upscaled, (3,3), CV2_DEFAULT_KERNEL_SIGMA)
    g_r, _g_g1, _g_b, _g_g2 = bayer_to_rgbg(g_upscaled)
    return resample_channel(r, g_r, g_hf_cut, BayerPatternPosition.TOP_LEFT)

def debayer(image : Union[RawRgbgData_BaseType]) -> Optional[RawDebayerData]:
    if not(image.is_valid()):
        return None
    
    r, g1, b, g2 = bayer_to_rgbg(image.bayer_data_scaled)
    wb_coeff = image.cam_wb.get_reciprocal_multipliers()

    g_up = resample_g_to_full_resolution(g1, g2)
    r_up, b_up = resample_rb(r, b, g_up)

    debayered =  np.dstack((r_up  * wb_coeff[0],
                            g_up  * wb_coeff[1],
                            b_up  * wb_coeff[2]))

    output = RawDebayerData(debayered, wb_coeff, wb_norm=False)
    output.mat_xyz = image.cam_wb.get_matrix()
    output.current_ev = image.current_ev
    return output