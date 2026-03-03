from typing import Optional, Tuple, Union

import cv2
import numpy as np

from pySP.base_types.image_base import RawDebayerData, RawRgbgData_BaseType
from pySP.bayer_chan_mixer import bayer_to_rgbg, rgbg_to_bayer
from pySP.debayer.gaussian import CV2_DEFAULT_KERNEL_SIGMA, CV2_DEFAULT_UNNORM_GAUSSIAN_KERNEL, BayerPatternPosition, get_rgbg_kernel
from pySP.debayer.edge_assisted_bilinear import resample_g_to_full_resolution

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

    g_up = resample_g_to_full_resolution(g1, g2) * wb_coeff[1]
    r_up, b_up = resample_rb(r * wb_coeff[0], b * wb_coeff[2], g_up)

    debayered =  np.dstack((r_up, g_up, b_up))

    output = RawDebayerData(debayered, wb_coeff, wb_norm=False)
    output.mat_xyz = image.cam_wb.get_matrix()
    output.current_ev = image.current_ev
    return output