import numpy as np
import cv2
from typing import List, Optional
from .bayer_chan_mixer import bayer_to_rgbg, rgbg_to_bayer
from .image import RawRgbgData

# TODO - More aggressive checking for dimension sizes

def median2(chan : np.ndarray) -> np.ndarray:
    """Performs a fast median blur with radius 2.

    Args:
        chan (np.ndarray): Input image.

    Returns:
        np.ndarray: Blurred image.
    """

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
    """Finds erroneous pixels on each color channel based on differences from
    a small median blur.

    Args:
        bayer_scaled (np.ndarray): Normalized Bayer image with RGBG pattern.
        multiplier (float, optional): Multiplier for hot pixel threshold. Higher requires more difference to trigger. Defaults to 1.5.
        quantile (float, optional): Quartile where hot pixels sit for thresholding. Should be between 0 and 1; higher is stricter. Defaults to 0.9999.

    Returns:
        List[np.ndarray]: List of bad pixel masks for each color channel.
    """

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
    """Finds shared erroneous pixels on each color channel based on differences from
    a small median blur. Returned masks will only include pixels that were included in more than
    ratio% amount of images (i.e., min_ratio of 0.1 means 10% of masks for that color must include
    that pixel or it is an outlier).

    Args:
        erroneous_mask (List[List[np.ndarray]]): List of erroneous pixel masks for multiple images.
        min_ratio (float, optional): Percentage of masks for pixels to be included in to be retained. Should be between 0 and 1. Defaults to 0.1 (10% of all masks).

    Returns:
        Optional[List[np.ndarray]]: List of bad pixel masks for each color channel; None if not enough masks were provided.
    """

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
    """Infill color channels in an image based on bad pixel masks.

    Args:
        image (RawRgbgData): Bayer image with RGBG pattern.
        masks (List[np.ndarray]): List of bad pixel masks for each color channel.
    """

    if len(masks) != 4:
        return
    
    chans = bayer_to_rgbg(image.bayer_data_scaled)
    new_chans = []
    for chan, mask in zip(chans, masks):
        new_chans.append(cv2.inpaint(chan, mask.astype(np.uint8) * 255, 3, cv2.INPAINT_NS))
    
    image.bayer_data_scaled = rgbg_to_bayer(new_chans[0], new_chans[1], new_chans[2], new_chans[3])