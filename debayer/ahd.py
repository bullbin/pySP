from typing import Optional, Union

import cv2
import numpy as np

from pySP.wb_cct.helpers_cam_mat import MatXyzToCamera

from ..base_types.image_base import RawDebayerData, RawRgbgData_BaseType
from ..bayer_chan_mixer import bayer_to_rgbg, rgbg_to_bayer
from ..colorize import cam_to_lin_srgb
from .ahd_homogeneity_cython import build_map
from .gaussian import BayerPatternPosition, get_rgbg_kernel

def debayer(image : Union[RawRgbgData_BaseType], postprocess_stages : int = 1) -> Optional[RawDebayerData]:
    """Debayer using the Adaptive Homogeneity-Directed Demosaicing algorithm by Hirakawa and Parks (2005).

    This is slow but produces high-quality results with reduced zippering and smooth graduation.
    
    This works by using recreating the green channel for red and blue in two directions and using a
    smoothness metric to pick the resulting direction that produces the least artifacts.

    If a HDR image is provided, internal tonemapping will be performed while reducing zipper artifacts.

    Args:
        image (RawRgbgData_BaseType): Bayer image with RGBG pattern.
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
        wb_coeff = image.cam_wb.get_reciprocal_multipliers()
        im_rgb = cam_to_lin_srgb(np.dstack((r * wb_coeff[0],
                                            g * wb_coeff[1],
                                            b * wb_coeff[2])), image.cam_wb.get_matrix(), clip_highlights=False)
        
        if image.get_hdr():
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

    # Note - h_optimal is slightly less blurry but tints harder with false colors. h_fast is better in this regard.
    #        Testing at 0.875 (old value) would reveal strong pink edge fringing caused entirely by these weights.
    #        Do not set this value too high or you'll experience debugging hell
    h_optimal   = np.array([-0.2569, 0.4339, 0.5138, 0.4339, -0.2569], dtype=np.float32)
    h_fast      = np.array([-0.25, 0.5, 0.5, 0.5, -0.25], dtype=np.float32)
    ratio_optimal = 0.0
    h = (h_optimal * ratio_optimal) + (h_fast * (1 - ratio_optimal))

    # Interpolate green channel from red
    gh_r = (r[1:-1, :-2] * h[0]) + (g1[1:-1, :-2] * h[1]) + (r[1:-1, 1:-1] * h[2]) + (g1[1:-1, 1:-1] * h[3]) + (r[1:-1, 2:] * h[4])
    gv_r = (r[:-2, 1:-1] * h[0]) + (g2[:-2, 1:-1] * h[1]) + (r[1:-1, 1:-1] * h[2]) + (g2[1:-1, 1:-1] * h[3]) + (r[2:, 1:-1] * h[4])

    # Interpolate green channel from blue
    gh_b = (b[1:-1, :-2] * h[0]) + (g2[1:-1, 1:-1] * h[1]) + (b[1:-1, 1:-1] * h[2]) + (g2[1:-1, 2:] * h[3]) + (b[1:-1, 2:] * h[4])
    gv_b = (b[:-2, 1:-1] * h[0]) + (g1[1:-1, 1:-1] * h[1]) + (b[1:-1, 1:-1] * h[2]) + (g1[2:, 1:-1] * h[3]) + (b[2:, 1:-1] * h[4])

    # Reconstruct full resolution green channels
    g_h = rgbg_to_bayer(gh_r, g1[1:-1, 1:-1], gh_b, g2[1:-1, 1:-1])
    g_v = rgbg_to_bayer(gv_r, g1[1:-1, 1:-1], gv_b, g2[1:-1, 1:-1])

    # Reconstruct full resolution other channels
    # Paper uses bilinear filter for low-pass. Most current implementations use a Gaussian filter.
    # Below is a photosite-aware implementation of cv2.pyrUp that performs Gaussian upsampling
    #     without introducing plane decentering.

    # We use cv2's default approximate 5x5 Gaussian. This is normalized to float internally so
    #     doesn't actually speed anything up :)
    # This is an approximation of a 5x5 Gaussian kernel with sigma 1.0
    cv2_default_gauss = np.array([[1, 4, 6, 4,1],
                                  [4,16,24,16,4],
                                  [6,24,36,24,6],
                                  [4,16,24,16,4],
                                  [1, 4, 6, 4,1]])
    
    # When using photosite operations, the kernel acts more like a 3x3 kernel.
    # TODO - There's a bug with aligment (or scaling) that causes worsening of friging when using original channels. This is unintended,
    #            the paper just adds the original green channel back which should work. It doesn't in this implementation for some reason.
    #            Demosaicing is broadly matched with dcraw when using low-pass for r,b but this shouldn't happen since we want high frequency
    #            detail. We can readd high frequency by cutting it from original green channels and adding it back to upscaled green.
    #        There is probably some error with the Gaussian upsampling function but this works.......
    #        I have tried bilinear kernels as well and they don't do as good a job. There's probably some bigger issue here :)
    delta_gh_hf = g_h - cv2.GaussianBlur(g_h, (3,3), 1.0)
    delta_gv_vf = g_v - cv2.GaussianBlur(g_v, (3,3), 1.0)
    
    k_r, k_g, k_g2, k_b = get_rgbg_kernel(cv2_default_gauss, BayerPatternPosition.TOP_LEFT)

    hg_r = rgbg_to_bayer(cv2.filter2D(gv_r, -1, k_r), cv2.filter2D(gv_r, -1, k_g), cv2.filter2D(gv_r, -1, k_b), cv2.filter2D(gv_r, -1, k_g2)) + delta_gh_hf
    vg_r = rgbg_to_bayer(cv2.filter2D(gh_r, -1, k_r), cv2.filter2D(gh_r, -1, k_g), cv2.filter2D(gh_r, -1, k_b), cv2.filter2D(gh_r, -1, k_g2)) + delta_gv_vf

    r_h = r[1:-1, 1:-1] - gh_r
    r_h = rgbg_to_bayer(cv2.filter2D(r_h, -1, k_r), cv2.filter2D(r_h, -1, k_g), cv2.filter2D(r_h, -1, k_b), cv2.filter2D(r_h, -1, k_g2)) + hg_r
    r_v = r[1:-1, 1:-1] - gv_r
    r_v = rgbg_to_bayer(cv2.filter2D(r_v, -1, k_r), cv2.filter2D(r_v, -1, k_g), cv2.filter2D(r_v, -1, k_b), cv2.filter2D(r_v, -1, k_g2)) + vg_r 

    k_r, k_g, k_g2, k_b = get_rgbg_kernel(cv2_default_gauss, BayerPatternPosition.BOTTOM_RIGHT)

    hg_b = rgbg_to_bayer(cv2.filter2D(gh_b, -1, k_r), cv2.filter2D(gh_b, -1, k_g), cv2.filter2D(gh_b, -1, k_b), cv2.filter2D(gh_b, -1, k_g2)) + delta_gh_hf
    vg_b = rgbg_to_bayer(cv2.filter2D(gv_b, -1, k_r), cv2.filter2D(gv_b, -1, k_g), cv2.filter2D(gv_b, -1, k_b), cv2.filter2D(gv_b, -1, k_g2)) + delta_gv_vf

    b_h = b[1:-1, 1:-1] - gh_b
    b_h = rgbg_to_bayer(cv2.filter2D(b_h, -1, k_r), cv2.filter2D(b_h, -1, k_g), cv2.filter2D(b_h, -1, k_b), cv2.filter2D(b_h, -1, k_g2)) + hg_b
    b_v = b[1:-1, 1:-1] - gv_b
    b_v = rgbg_to_bayer(cv2.filter2D(b_v, -1, k_r), cv2.filter2D(b_v, -1, k_g), cv2.filter2D(b_v, -1, k_b), cv2.filter2D(b_v, -1, k_g2)) + vg_b

    map_h = build_homogeneity_map(r_h, g_h, b_h, False)
    map_v = build_homogeneity_map(r_v, g_v, b_v, True)

    # Blur the homogeneity maps to consider wider homogeneity when merging
    map_h = cv2.blur(map_h, (3,3))
    map_v = cv2.blur(map_v, (3,3))

    rgb_h = np.dstack((r_h, g_h, b_h))
    rgb_v = np.dstack((r_v, g_v, b_v))

    combination = (map_h < map_v).astype(np.float32)
    combination = np.reshape(combination, (combination.shape[0], combination.shape[1], 1))

    rgb_h *= combination
    rgb_v *= (1 - combination)
    
    debayered = rgb_h + rgb_v

    # 3 iterations of postprocessing used in paper. Can kill small details but is effective at suppressing chroma errors
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

    # White balance prior to postprocessing to reduce highlight shifting
    wb_coeff = image.cam_wb.get_reciprocal_multipliers()
    debayered[:,:,0] = debayered[:,:,0] * wb_coeff[0]
    debayered[:,:,1] = debayered[:,:,1] * wb_coeff[1]
    debayered[:,:,2] = debayered[:,:,2] * wb_coeff[2]

    postprocess_stages = max(postprocess_stages, 0)
    for _i in range(postprocess_stages):
        debayered = postprocess_color(debayered)

    debayered = RawDebayerData(debayered, wb_coeff, wb_norm=False)
    debayered.mat_xyz = image.cam_wb.get_matrix()
    debayered.current_ev = image.current_ev
    return debayered