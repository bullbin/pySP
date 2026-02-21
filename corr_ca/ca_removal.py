from typing import Optional, Tuple

import cv2
import numpy as np

from pySP.base_types.image_base import RawRgbgData_BaseType
from pySP.bayer_chan_mixer import bayer_to_rgbg, rgbg_to_bayer
from pySP.corr_ca.instability import compute_structural_instability
from pySP.corr_ca.model.generic import CaCorrectionModel, ReversibleModelMixin
from pySP.corr_ca.model.poly5 import Poly5CorrectionModel
from pySP.debayer.edge_assisted_gaussian import resample_g_to_full_resolution
from pySP.debayer.edge_assisted_gaussian import resample_b, resample_r
from pySP.corr_ca.solver.radial_offset_solver import get_scale_pairs_using_pooled_tiler

def compute_ca_lens_models_for_raw(raw : RawRgbgData_BaseType, init_model_r : Optional[CaCorrectionModel] = Poly5CorrectionModel(), init_model_b : Optional[CaCorrectionModel] = Poly5CorrectionModel(),
                                   max_distortion_additional_scale : float = 0.004) -> Tuple[Optional[CaCorrectionModel], Optional[CaCorrectionModel]]:
    """Fit lens distortion models for removing chromatic abberation by aligning the red and blue channels onto green. This method makes the following assumptions:
    - The chromatic abberation is lateral, i.e., fringing that worsens towards the edges of the image
    - The lens center perfectly aligns with the imaging center
    - CA isn't too extreme

    This a blind method designed for applying CA removal in-raw. Lens models are fitted on a per-image basis with no prior knowledge of the lens. It does not correct for general distortions related to
    perspective or focal length but makes those easier.

    Structural instability is used as a proxy for demosaicing which is similar to an edge map but computed directly from Bayer data. Fitting is done using cross-correlation - edges within the red/blue
    channels should have corresponding edges within the green channel. For further information on method, this roughly follows 10.1109/ACCESS.2021.3096201.

    Args:
        raw (RawRgbgData_BaseType): Input raw image.
        init_model_r (Optional[CaCorrectionModel], optional): Starting model for correcting red channel. Must be reversible for later steps to work. Defaults to Poly5CorrectionModel(). If None, no model will be fitted to the data.
        init_model_b (Optional[CaCorrectionModel], optional): Starting model for correcting blue channel. Must be reversible for later steps to work. Defaults to Poly5CorrectionModel(). If None, no model will be fitted to the data.
        max_distortion_additional_scale (float, optional): Maximum scale factor for CA reach. For example, a pixel 0.8x from the center to edge will sit in search region 0.8 * (1 ± scale). Defaults to 0.004.

    Returns:
        Tuple[Optional[CaCorrectionModel], Optional[CaCorrectionModel]]: (fitted init_model_r, fitted init_model_b). If either model was None, the corresponding output will also be None.
    """

    si = compute_structural_instability(raw)

    if init_model_r != None:
        init_model_r.compute_coefficients(get_scale_pairs_using_pooled_tiler(si[:,:,0], si[:,:,1], max_reach=max_distortion_additional_scale))
    
    if init_model_b != None:
        init_model_b.compute_coefficients(get_scale_pairs_using_pooled_tiler(si[:,:,2], si[:,:,1], max_reach=max_distortion_additional_scale))
    
    return (init_model_r, init_model_b)

def remove_ca_from_raw(raw : RawRgbgData_BaseType, lens_model_r : Optional[CaCorrectionModel], lens_model_b : Optional[CaCorrectionModel]):
    """Remove lateral chromatic abberation from a raw file by applying inverse lens distortions onto the red and blue channels to align it with green.

    This method overwrites the source Bayer data with corrected channels. Resampling is performed - repeated uses of this method may lead to
    quality loss and subtle color shifting. Try to get distortions well-approximated the first time.

    For a summary of how this method works:
    - Green is upsampled without cross-correlation (using channel alone, so lower quality but no fringing artifacts).
    - Upsampled green is distorted using the lens model to red/blue and the difference is taken.
    - The difference is upsampled using the upsampled green. Full-resolution red/blue are computed.
    - Upsampled red/blue are undistorted using the lens model to get their corrected images then used to overwrite the original pixel grid.

    For further information on method, this roughly follows 10.1109/ACCESS.2021.3096201.

    In typical cases, the produced raw will demosaic to an identical image minus CA artifacts. You'll even spot identical demosaicing issues. This
    method aims to correct CA without introducing error of its own - no additional sharpening or anything. 

    You should complete all cleaning corrections prior to this - hot pixel removal, shading removal, etc to get the image as pristine as possible
    before realignment to prevent contamination.

    Args:
        raw (RawRgbgData_BaseType): Input raw image.
        lens_model_r (Optional[CaCorrectionModel]): Distortion model for red onto green. Must be reversible. If None, no corrections will be applied to this channel.
        lens_model_b (Optional[CaCorrectionModel]): Distortion model for blue onto green. Must be reversible. If None, no corrections will be applied to this channel.

    Raises:
        ValueError: Raised if red lens distortion model is not reversible. This is a requirement due to how this method works.
        ValueError: Raised if blue lens distortion model is not reversible. This is a requirement due to how this method works.
    """

    # If no models defined, nothing to do
    if lens_model_r == None and lens_model_b == None:
        return
    
    # Check any models are okay
    if lens_model_r != None and not(isinstance(lens_model_r, ReversibleModelMixin)):
        raise ValueError("Red lens model is not reversible so green cannot be re-aligned to remove error. Use a reversible model and try again.")
    if lens_model_b != None and not(isinstance(lens_model_b, ReversibleModelMixin)):
        raise ValueError("Blue lens model is not reversible so green cannot be re-aligned to remove error. Use a reversible model and try again.")
    
    # Resample green using simple upsample approach. We want as high res as possible so we can resample G further without major quality loss
    r, g1, b, g2 = bayer_to_rgbg(raw.bayer_data_scaled)
    g_resampled = resample_g_to_full_resolution(g1, g2)

    # For both channels, this is the workflow:
    # - Distort G to align G with R/B at destination
    # - Use distorted G to upsample R/B to full res
    # - Undistort full res R/B to align with original G
    # - Extract sample at Bayer position and overwrite raw with corrected channel

    if lens_model_r != None:
        coords_g_at_r = lens_model_r.get_undistorted_coordinates(g_resampled)
        g_at_r = cv2.remap(g_resampled,
                           np.clip(coords_g_at_r[:,:,1] + (g_resampled.shape[1] - 1) / 2, 0, g_resampled.shape[1] - 1),
                           np.clip(coords_g_at_r[:,:,0] + (g_resampled.shape[0] - 1) / 2, 0, g_resampled.shape[0] - 1),
                           cv2.INTER_LINEAR)
        
        r_resampled = resample_r(r * raw.cam_wb.get_reciprocal_multipliers()[0], g_at_r)

        coords_r_at_g = lens_model_r.get_distorted_coordinates(r_resampled)
        r_at_g = cv2.remap(r_resampled,
                           np.clip(coords_r_at_g[:,:,1] + (r_resampled.shape[1] - 1) / 2, 0, r_resampled.shape[1] - 1),
                           np.clip(coords_r_at_g[:,:,0] + (r_resampled.shape[0] - 1) / 2, 0, r_resampled.shape[0] - 1),
                           cv2.INTER_LINEAR)
        
        r = bayer_to_rgbg(r_at_g)[0] / raw.cam_wb.get_reciprocal_multipliers()[0]
    
    if lens_model_b != None:
        coords_g_at_b = lens_model_b.get_undistorted_coordinates(g_resampled)
        g_at_b = cv2.remap(g_resampled,
                           np.clip(coords_g_at_b[:,:,1] + (g_resampled.shape[1] - 1) / 2, 0, g_resampled.shape[1] - 1),
                           np.clip(coords_g_at_b[:,:,0] + (g_resampled.shape[0] - 1) / 2, 0, g_resampled.shape[0] - 1),
                           cv2.INTER_LINEAR)
        
        b_resampled = resample_b(b * raw.cam_wb.get_reciprocal_multipliers()[2], g_at_b)

        coords_b_at_g = lens_model_b.get_distorted_coordinates(b_resampled)
        b_at_g = cv2.remap(b_resampled,
                           np.clip(coords_b_at_g[:,:,1] + (b_resampled.shape[1] - 1) / 2, 0, b_resampled.shape[1] - 1),
                           np.clip(coords_b_at_g[:,:,0] + (b_resampled.shape[0] - 1) / 2, 0, b_resampled.shape[0] - 1),
                           cv2.INTER_LINEAR)
        
        b = bayer_to_rgbg(b_at_g)[2] / raw.cam_wb.get_reciprocal_multipliers()[2]
    
    raw.bayer_data_scaled = rgbg_to_bayer(r, g1, b, g2)