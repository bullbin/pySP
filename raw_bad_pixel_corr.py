import numpy as np
import cv2
from typing import List, Optional
from bayer_chan_mixer import bayer_to_rgbg, rgbg_to_bayer
from image import RawRgbgData

# TODO - More aggressive checking for dimension sizes

def median2(chan : np.ndarray) -> np.ndarray:

    # Doing a median blur of smallest scale (3x3) is far too destructive at filter-level since we're already working at quarter resolution
    # Therefore, we try to do a 2x2 median blur (which CV2 will not accelerate) instead
    padded = np.pad(chan, (1,1), mode="reflect")

    chan_e_neighbour    = padded[1:-1, 2:]
    chan_s_neighbour    = padded[2:, 1:-1]

    # chan_nw_neighbour   = padded[:-2, :-2]
    # chan_ne_neighbour   = padded[:-2, 2:]
    chan_se_neighbour   = padded[2:, 2:]
    # chan_sw_neighbour   = padded[2:, :-2]
    
    flattened = np.array([chan, chan_e_neighbour, chan_s_neighbour, chan_se_neighbour])
    return np.median(flattened, axis=0)

def find_erroneous_pixels_median(bayer_scaled : np.ndarray, multiplier : float = 1.5, quantile : float = 0.9999) -> List[np.ndarray]:
    masks : List[np.ndarray] = []

    for chan in bayer_to_rgbg(bayer_scaled):
        chan_blur = median2(chan)
        delta = np.abs(chan - chan_blur)
        noise_floor = np.mean(delta)
        delta = abs(delta - noise_floor)

        strong_quantile = np.quantile(delta, quantile) * multiplier
        hot_pixels = np.zeros(shape=chan.shape, dtype=np.bool_)
        hot_pixels[delta > strong_quantile] = True
        masks.append(hot_pixels)

    return masks

def find_shared_pixels(erroneous_mask : List[List[np.ndarray]], min_ratio : float = 0.1) -> Optional[List[np.ndarray]]:
    if len(erroneous_mask) == 0:
        return None
    
    chan_size = len(erroneous_mask[0])
    for mask in erroneous_mask[1:]:
        if len(mask) != chan_size:
            return None

    min_acceptance = np.ceil(len(erroneous_mask) * min_ratio)

    chans = [[] for _ in range(chan_size)]

    for masks in erroneous_mask:
        for idx_chan, chan in enumerate(masks):
            chans[idx_chan].append(chan)
    
    chan_sums = [np.sum(np.array(i), axis=0, dtype=np.int16) for i in chans]
    
    masks = []
    for chan in chan_sums:
        hot_pixels = np.zeros(shape=chan.shape, dtype=np.bool_)
        hot_pixels[chan >= min_acceptance] = True
        masks.append(hot_pixels)
    
    return masks

def repair_bad_pixels(image : RawRgbgData, masks : List[np.ndarray]):
    if len(masks) != 4:
        return
    
    chans = bayer_to_rgbg(image.bayer_data_scaled)
    new_chans = []
    for chan, mask in zip(chans, masks):
        new_chans.append(cv2.inpaint(chan, mask.astype(np.uint8) * 255, 3, cv2.INPAINT_NS))
    
    image.bayer_data_scaled = rgbg_to_bayer(new_chans[0], new_chans[1], new_chans[2], new_chans[3])