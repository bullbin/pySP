import numpy as np
from .bayer_chan_mixer import bayer_to_rgbg, rgbg_to_bayer

def bayer_normalize(rgbg : np.ndarray, chan_black : np.ndarray, chan_sat : np.ndarray):
    r, g1, b, g2 = bayer_to_rgbg(rgbg)

    # TODO - Check channel ordering, should be in Bayer pattern
    r   = np.clip(r - chan_black[0], 0, chan_sat[0]).astype(np.float32) / (chan_sat[0] - chan_black[0])
    g1  = np.clip(g1 - chan_black[1], 0, chan_sat[1]).astype(np.float32) / (chan_sat[1] - chan_black[1])
    b   = np.clip(b - chan_black[2], 0, chan_sat[2]).astype(np.float32) / (chan_sat[2] - chan_black[2])
    g2  = np.clip(g2 - chan_black[3], 0, chan_sat[3]).astype(np.float32) / (chan_sat[3] - chan_black[3])

    return rgbg_to_bayer(r, g1, b, g2)