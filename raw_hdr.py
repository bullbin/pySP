from .image import RawRgbgData, RawDebayerData
from .colorize import cam_to_lin_srgb
import numpy as np
from typing import Tuple, List, Optional

def fuse_exposures_from_debayer(in_exposures : List[RawDebayerData], target_ev : Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
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

def fuse_exposures_to_raw(in_exposures : List[RawRgbgData], target_ev : Optional[float] = None) -> Tuple[Optional[RawRgbgData], np.ndarray]:
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

    # TODO - Bad sample rejection
    # If EVs are ordered, weights should be ordered too, with a peak at the best EV.
    # If this isn't the case, favor the sample with lower boost required.
    # We can probably roughly compute the sample by performing non-saturated median on sample area to check for rejected areas.
    for exposure, ev_offset in zip(valid_exposures, ev_offsets):
        weights = (0.5 - np.abs(exposure.bayer_data_scaled - 0.5))
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
    hdr_image.mat_xyz = np.copy(valid_exposures[0].mat_xyz)
    hdr_image.wb_coeff = np.copy(valid_exposures[0].wb_coeff)

    return (hdr_image, debug_count_references)