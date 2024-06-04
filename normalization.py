import numpy as np

def bayer_normalize(rgbg : np.ndarray, chan_black : np.ndarray, chan_sat : np.ndarray):
    evens   = rgbg[0::2,:].astype(np.float32)
    odds    = rgbg[1::2,:].astype(np.float32)

    # TODO - Check channel ordering, should be in Bayer pattern
    # TODO - Shouldn't have to clip but white point exceeds stored data!
    r   = np.clip(evens[:,0::2] - chan_black[0], 0, chan_sat[0]) / chan_sat[0]
    g1  = np.clip(evens[:,1::2] - chan_black[1], 0, chan_sat[1]) / chan_sat[1]
    b   = np.clip(odds[:,1::2] - chan_black[2], 0, chan_sat[2]) / chan_sat[2]
    g2  = np.clip(odds[:,0::2] - chan_black[3], 0, chan_sat[3]) / chan_sat[3]

    output = np.zeros_like(rgbg, dtype=np.float32)
    output[0::2,:][:,0::2] = r
    output[0::2,:][:,1::2] = g1
    output[1::2,:][:,1::2] = b
    output[1::2,:][:,0::2] = g2

    return output