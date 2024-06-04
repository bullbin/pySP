import numpy as np
import cv2
from const import QualityDemosaic, PatternDemosaic

def debayer(raw_2d_scaled : np.ndarray, wb_rgb : np.ndarray, pattern : PatternDemosaic, quality : QualityDemosaic) -> np.ndarray:
    # TODO - Check channel ordering, should be in Bayer pattern
    evens   = raw_2d_scaled[0::2,:].astype(np.float32)
    odds    = raw_2d_scaled[1::2,:].astype(np.float32)

    r   = evens[:,0::2]
    g1  = evens[:,1::2]
    b   = odds[:,1::2]
    g2  = odds[:,0::2]

    def debayer_shift() -> np.ndarray:
        rgb = np.zeros((r.shape[0], r.shape[1], 3), dtype=np.float32)

        rgb[:,:,0] = r * wb_rgb[0]
        rgb[:,:,1] = ((g1 + g2) / 2) * wb_rgb[1]
        rgb[:,:,2] = b * wb_rgb[2]

        return cv2.resize(rgb, (raw_2d_scaled.shape[1], raw_2d_scaled.shape[0]))

    return debayer_shift()