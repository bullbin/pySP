import numpy as np
import cv2

from pySP.bayer_chan_mixer import bayer_to_rgbg, rgbg_to_bayer
from .image import RawRgbgData

def dark_frame_subtraction(raw : RawRgbgData, dark_frame : RawRgbgData):
    """Remove dark current noise from an image.

    Args:
        raw (RawRgbgData): Raw RGBG image.
        dark_frame (RawRgbgData): Raw dark frame RGBG image.
    """
    return np.copy(raw)

def bias_frame_subtraction(raw : RawRgbgData, bias_frame : RawRgbgData):
    """Remove fixed-pattern noise from an image.

    Args:
        raw (RawRgbgData): Raw RGBG image.
        bias_frame (RawRgbgData): Raw bias frame RGBG image.
    """
    return np.copy(raw)

def flat_frame_correction(image : RawRgbgData, flat : RawRgbgData, clamp_high : bool = False):
    """Apply flat-frame correction in-place to an image.

    The output for this method will always be a valid image, but steps are taken in-case of problematic data:
    - Divide by zero is replaced by the maximal value of each channel after correction
    - Missing flat frames will leave the image untouched
    - Correction pushing below zero will clamp photosite values to zero.

    Args:
        image (RawRgbgData): Image to be corrected.
        flat (RawRgbgData): Flat field image.
        clamp_high (bool, optional): True to clamp corrected photosites at 1. Breaks HDR images but useful elsewhere. Defaults to False.
    """

    if not(image.is_valid() and flat.is_valid()):
        return

    r, g1, b, g2 = bayer_to_rgbg(image.bayer_data_scaled)
    flat_r, flat_g1, flat_b, flat_g2 = bayer_to_rgbg(flat.bayer_data_scaled)

    def correct_channel(chan : np.ndarray, chan_flat : np.ndarray) -> np.ndarray:
        # TODO - Add dark frame correction, currently we assume dark frame is zero
        mean_chan = np.mean(chan_flat)
        output = (chan * mean_chan) / chan_flat
        
        if np.isinf(output).all():
            # In the case where the flat frame was completely black, leave image alone
            return np.copy(chan)

        max_output = np.max(np.ma.masked_invalid(output))
        output[output == np.inf] = max_output   # Where we have infinity, replace with max channel
        output[output < 0] = 0

        if clamp_high:
            output[output > 1] = 1

        return output

    image.bayer_data_scaled = rgbg_to_bayer(correct_channel(r, flat_r),
                                            correct_channel(g1, flat_g1),
                                            correct_channel(b, flat_b),
                                            correct_channel(g2, flat_g2))