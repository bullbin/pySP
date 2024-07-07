from typing import Optional, Union

import cv2
import numpy as np

from .ahd_homogeneity_cython import build_map
from ..bayer_chan_mixer import bayer_to_rgbg, rgbg_to_bayer
from ..colorize import cam_to_lin_srgb
from ..image import RawDebayerData, RawRgbgData, HdrRgbgData

def debayer(image : Union[RawRgbgData, HdrRgbgData], deartifact : bool = True, postprocess_stages : int = 1) -> Optional[RawDebayerData]:
    """Debayer using the Adaptive Homogeneity-Directed Demosaicing algorithm by Hirakawa and Parks (2005).

    This is slow but produces high-quality results with reduced zippering and smooth graduation.
    
    This works by using recreating the green channel for red and blue in two directions and using a
    smoothness metric to pick the resulting direction that produces the least artifacts.

    If a HDR image is provided, internal tonemapping will be performed while reducing zipper artifacts.

    Args:
        image (Union[RawRgbgData, HdrRgbgData]): Bayer image with RGBG pattern.
        deartifact (bool, optional): Restrict interpolation to within surrounding channels. Defaults to True.
        postprocess (bool, optional): Reduce color bleeding. Defaults to True.

    Returns:
        Optional[RawDebayerData]: Debayered image; None if image is not valid.
    """

    def build_homogeneity_map(r : np.ndarray, g : np.ndarray, b : np.ndarray, is_vertical : bool, domain_k : int = 3) -> np.ndarray:

        # Domain ball neighborhood - pixels within d distance of x
        # Level neighborhood - pixels within epsilon brightness of x
        # Color neighborhood - pixels within epsilon color of x
        # Metric neighborhood - Intersection of all
        # Homogeneity = Len(metric neighborhood) / Len(domain)
        assert domain_k % 2 == 1
        k_pad = domain_k // 2

        # Use camera matrix to convert internal to CIELAB
        # TODO - Can just use XYZ values directly
        # TODO - Don't know what OpenCV is doing to these values. Colorspace? Gamma correction?
        im_rgb = cam_to_lin_srgb(np.dstack((r * image.wb_coeff[0],
                                            g * image.wb_coeff[1],
                                            b * image.wb_coeff[2])), image.mat_xyz, clip_highlights=False)
        
        if isinstance(image, HdrRgbgData):
            # If the image is HDR, we can't formulate a CIELAB representation
            # Instead, use luma for L* and tonemap to get A*B*.
            luma = 0.2126 * im_rgb[:,:,0] + 0.7152 * im_rgb[:,:,1] + 0.0722 * im_rgb[:,:,2]
            
            im_rgb = im_rgb / (1 + im_rgb)
            lab = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = luma
        else:
            # TODO - Maybe clamp for safety
            lab = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2LAB)

        lab = cv2.copyMakeBorder(lab, k_pad, k_pad, k_pad, k_pad, cv2.BORDER_REFLECT)
        
        homogeneity = build_map(lab, k_pad, domain_k, is_vertical)
        return homogeneity

    if not(image.is_valid()):
        return None

    r, g1, b, g2 = bayer_to_rgbg(image.bayer_data_scaled)

    # Pad photosites to make wrapping a bit easier
    #     Interpolation needs the photosites at edges as well as current, so we need to pad to let this work for edge pixels
    r = cv2.copyMakeBorder(r, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    g1 = cv2.copyMakeBorder(g1, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    b = cv2.copyMakeBorder(b, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    g2 = cv2.copyMakeBorder(g2, 1, 1, 1, 1, cv2.BORDER_REFLECT)

    # Blend h, h_optimal is the optimal solution presented in paper. Produces no mazes but leaves aliased crosses instead
    #     h_fast is their power-of-two version. Smoother appearance but produces mazes
    # Blending them can improve the naturalness a bit
    h_optimal   = np.array([-0.2569, 0.4339, 0.5138, 0.4339, -0.2569])
    h_fast      = np.array([-0.25, 0.5, 0.5, 0.5, -0.25])
    ratio_optimal = 0.875
    h = (h_optimal * ratio_optimal) + (h_fast * (1 - ratio_optimal))

    # Artifacts can be created from hot pixel removal, this can reduce the appearance of them
    def deartifact(chan : np.ndarray, lim_chan_0 : np.ndarray, lim_chan_1 : np.ndarray) -> np.ndarray:
        # TODO - Try median
        if deartifact:
            stacked = np.dstack((lim_chan_0, lim_chan_1))
            min_chan = np.min(stacked, axis=2)
            max_chan = np.max(stacked, axis=2)
            average = (min_chan + max_chan) / 2
            return np.where(np.logical_or(chan < min_chan, chan > max_chan), average, chan)
        
        return chan

    # Interpolate green channel from red
    gh_r = (r[1:-1, :-2] * h[0]) + (g1[1:-1, :-2] * h[1]) + (r[1:-1, 1:-1] * h[2]) + (g1[1:-1, 1:-1] * h[3]) + (r[1:-1, 2:] * h[4])
    gv_r = (r[:-2, 1:-1] * h[0]) + (g2[:-2, 1:-1] * h[1]) + (r[1:-1, 1:-1] * h[2]) + (g2[1:-1, 1:-1] * h[3]) + (r[2:, 1:-1] * h[4])

    # Interpolate green channel from blue
    gh_b = (b[1:-1, :-2] * h[0]) + (g2[1:-1, 1:-1] * h[1]) + (b[1:-1, 1:-1] * h[2]) + (g2[1:-1, 2:] * h[3]) + (b[1:-1, 2:] * h[4])
    gv_b = (b[:-2, 1:-1] * h[0]) + (g1[1:-1, 1:-1] * h[1]) + (b[1:-1, 1:-1] * h[2]) + (g1[2:, 1:-1] * h[3]) + (b[2:, 1:-1] * h[4])

    gh_r = deartifact(gh_r, g1[1:-1, :-2], g1[1:-1, 1:-1])
    gv_r = deartifact(gv_r, g2[:-2, 1:-1], g2[1:-1, 1:-1])
    gh_b = deartifact(gh_b, g2[1:-1, 1:-1], g2[1:-1, 2:])
    gv_b = deartifact(gv_b, g1[1:-1, 1:-1], g1[2:, 1:-1])

    # Reconstruct full resolution green channels
    g_h = rgbg_to_bayer(gh_r, g1[1:-1, 1:-1], gh_b, g2[1:-1, 1:-1])
    g_v = rgbg_to_bayer(gv_r, g1[1:-1, 1:-1], gv_b, g2[1:-1, 1:-1])

    # Reconstruct full resolution other channels
    r_h = cv2.pyrUp(r[1:-1, 1:-1] - gh_r) + g_h
    r_v = cv2.pyrUp(r[1:-1, 1:-1] - gv_r) + g_v
    b_h = cv2.pyrUp(b[1:-1, 1:-1] - gh_b) + g_h
    b_v = cv2.pyrUp(b[1:-1, 1:-1] - gv_b) + g_v
    
    map_h = build_homogeneity_map(r_h, b_h, g_h, False)
    map_v = build_homogeneity_map(r_v, b_v, g_v, True)

    rgb_h = np.dstack((r_h, g_h, b_h))
    rgb_v = np.dstack((r_v, g_v, b_v))

    # Blur the homogeneity maps to consider wider homogeneity when merging
    map_h = cv2.blur(map_h, (3,3))
    map_v = cv2.blur(map_v, (3,3))

    combination = (map_h < map_v).astype(np.float32)
    combination = np.reshape(combination, (combination.shape[0], combination.shape[1], 1))

    rgb_h *= combination
    rgb_v *= (1 - combination)

    def postprocess_color(image_prev : np.ndarray) -> np.ndarray:

        def median(im : np.ndarray) -> np.ndarray:
            return cv2.medianBlur(im, 5)    # Small median results in extreme sharpening and overshoot, use 3 or higher to eliminate overshoot

        image_next = np.copy(image_prev)
        r = image_next[:,:,0]
        g = image_next[:,:,1]
        b = image_next[:,:,2]

        r = median(r - g) + g
        b = median(b - g) + g
        g = (median(g - r) + median(g - b) + r + b) / 2
        return np.dstack((r,g,b))

    # 3 iterations of postprocessing
    debayered = rgb_h + rgb_v

    # White balance prior to postprocessing to reduce highlight shifting
    debayered[:,:,0] = debayered[:,:,0] * image.wb_coeff[0]
    debayered[:,:,1] = debayered[:,:,1] * image.wb_coeff[1]
    debayered[:,:,2] = debayered[:,:,2] * image.wb_coeff[2]

    postprocess_stages = max(postprocess_stages, 0)
    for _i in range(postprocess_stages):
        debayered = postprocess_color(debayered)

    debayered = RawDebayerData(debayered, np.copy(image.wb_coeff), wb_norm=False)
    debayered.mat_xyz = np.copy(image.mat_xyz)
    debayered.current_ev = image.current_ev
    return debayered