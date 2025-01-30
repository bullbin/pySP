from pySP.bayer_chan_mixer import rgbg_to_bayer
from .image import RawRgbgData, RawDebayerData
from .colorize import cam_to_lin_srgb
import numpy as np
from typing import Tuple, List, Optional

def fuse_exposures_from_debayer(in_exposures : List[RawDebayerData], target_ev : Optional[float] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Fuse exposures to linearized sRGB HDR from a list of debayered images.

    This method operates in sensor-space so is unaffected by response curves or sensor saturation.
    It works by the following:
    - Using stored exposure values, images are shifted to the same target exposure.
    - HDR image is fused by weighted averaging on both sensor saturation and EV difference to reduce noise amplification.

    No noise reduction, bad pixel correction or alignment is performed. These must all be completed
    prior to merging. Noise is controlled well as long as a pixel has multiple images to sample from.

    Args:
        in_exposures (List[RawDebayerData]): List of input debayered images.
        target_ev (Optional[float], optional): Target exposure. Higher is darker. Defaults to None which will use the average of all inputs.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: (Linearized sRGB image, debug buffer tracking amount of contributions for each pixel); None if no valid images were provided.
    """

    valid_exposures : List[RawDebayerData] = []
    
    for exposure in in_exposures:
        if exposure.is_valid():
            valid_exposures.append(exposure)
    
    if len(valid_exposures) == 0:
        return None

    if target_ev == None:
        target_ev = 0
        for exposure in valid_exposures:
            target_ev += exposure.current_ev
        target_ev /= len(valid_exposures)
    else:
        assert target_ev > 0
    
    ev_offsets : List[float] = []
    for exposure in valid_exposures:
        ev_offsets.append(2 ** (exposure.current_ev - target_ev))
   
    sum_pixel = np.zeros(shape=in_exposures[0].image.shape, dtype=np.float32)
    sum_weight = np.zeros(shape=sum_pixel.shape, dtype=np.float32)
    debug_count_references = np.zeros(shape=sum_pixel.shape, dtype=np.int32)

    offset_max_exposure = np.max(ev_offsets)
    max_exposure = None

    for exposure, ev_offset in zip(valid_exposures, ev_offsets):
        
        exposure.wb_undo()

        weights = (0.5 - np.abs((exposure.image) - 0.5))     # Reweight according to white balance co-efficients to restore dynamic range of sensor

        bias = 1.6 ** (-0.1 * ev_offset)                                        # Bias stacking to favor closest EV - this is just a random curve that should weight
        weights *= bias                                                         #     target EVs (~0) at 1 and EVs up to 16x gaps still favorably (ISO 1600 is still good)

        sum_weight += weights

        exposure.wb_apply()

        if ev_offset == offset_max_exposure:
            max_exposure = exposure.image

        sum_pixel += exposure.image * weights * ev_offset

        debug_count_references[weights > 0] += 1
        
    max_exposure = np.multiply(max_exposure, offset_max_exposure)

    with np.errstate(divide='ignore', invalid='ignore'):    # Expected, we filter out bad results next
        sum_pixel = np.divide(sum_pixel, sum_weight)
    sum_pixel = np.where(sum_weight == 0, max_exposure, sum_pixel)
    sum_pixel = sum_pixel.astype(np.float32)

    sum_pixel = cam_to_lin_srgb(sum_pixel, in_exposures[0].mat_xyz, clip_highlights=False)

    return (sum_pixel, debug_count_references)

def fuse_exposures_to_raw(in_exposures : List[RawRgbgData], target_ev : Optional[float] = None) -> Optional[Tuple[RawRgbgData, np.ndarray]]:
    """Fuse exposures to a new HDR raw image from a list of raw images while preserving the Bayer pattern.

    This method operates in sensor-space so is unaffected by response curves or sensor saturation.
    It works by the following:
    - Using stored exposure values, images are shifted to the same target exposure.
    - HDR image is fused per sensor channel by weighted averaging on both sensor saturation and EV difference to reduce noise amplification.
    
    No noise reduction, bad pixel correction or alignment is performed. These must all be completed
    prior to merging. Noise is controlled well as long as a pixel has multiple images to sample from.
    
    The output will need to be debayered to be used which may cause problems with highlight clipping
    algorithms as HDR will extend channels beyond their natural clipping point. Make sure to use
    HDR-agnostic debayering algorithms to ensure retention of dynamic range.

    Args:
        in_exposures (List[RawRgbgData]): List of input Bayer images.
        target_ev (Optional[float], optional): Target exposure. Higher is darker. Defaults to None which will use the average of all inputs.

    Returns:
        Optional[Tuple[HdrRgbgData, np.ndarray]]: (HDR Bayer image, debug buffer tracking amount of contributions for each pixel); None if no valid images were provided.
    """

    valid_exposures : List[RawRgbgData] = []
    
    for exposure in in_exposures:
        if exposure.is_valid():
            valid_exposures.append(exposure)
    
    if len(valid_exposures) == 0:
        return None

    if target_ev == None:
        target_ev = 0
        for exposure in valid_exposures:
            target_ev += exposure.current_ev
        target_ev /= len(valid_exposures)
    else:
        assert target_ev > 0
    
    ev_offsets : List[float] = []
    for exposure in valid_exposures:
        ev_offsets.append(2 ** (exposure.current_ev - target_ev))
    
    debug_count_references = np.zeros(shape=valid_exposures[0].bayer_data_scaled.shape, dtype=np.int32)
    sum_pixel = np.zeros_like(valid_exposures[0].bayer_data_scaled)
    sum_weight = np.zeros_like(valid_exposures[0].bayer_data_scaled)

    # Since we'll correct WB later, add additional bias on WB to reduce noise gain after WB is scaled
    wb_coeff = valid_exposures[0].cam_wb.get_reciprocal_multipliers()
    bayer_noise_weight = np.ones((valid_exposures[0].bayer_data_scaled.shape[0] // 2, valid_exposures[0].bayer_data_scaled.shape[1] // 2), dtype=np.float32)
    bayer_noise_weight = rgbg_to_bayer(bayer_noise_weight * wb_coeff[0],
                                       wb_coeff[1],
                                       wb_coeff[2],
                                       wb_coeff[1])

    for exposure, ev_offset in zip(valid_exposures, ev_offsets):
        bias = 1.6 ** (-0.1 * np.abs(ev_offset * bayer_noise_weight))       # Bias stacking to favor closest EV - this is just a random curve that should weight
        weights = (0.5 - np.abs(exposure.bayer_data_scaled - 0.5)) * bias   #     target EVs (~0) at 1 and EVs up to 16x gaps still favorably (ISO 1600 is still good)
        sum_weight += weights
        sum_pixel += exposure.bayer_data_scaled * weights * ev_offset

        debug_count_references[weights > 0] += 1
        
    offset_max_exposure = np.argmax(ev_offsets)
    max_exposure = np.multiply(valid_exposures[offset_max_exposure].bayer_data_scaled, ev_offsets[offset_max_exposure])

    with np.errstate(divide='ignore', invalid='ignore'):    # Expected, we filter out bad results next
        sum_pixel = np.divide(sum_pixel, sum_weight)
    sum_pixel = np.where(sum_weight == 0, max_exposure, sum_pixel)

    hdr_image = RawRgbgData()
    hdr_image.bayer_data_scaled = sum_pixel
    hdr_image.current_ev = target_ev
    hdr_image.lim_sat = max(ev_offsets)
    hdr_image.cam_wb = valid_exposures[0].cam_wb.copy()
    hdr_image.set_hdr(True)

    return (hdr_image, debug_count_references)