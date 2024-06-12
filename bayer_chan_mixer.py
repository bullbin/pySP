import numpy as np
from typing import Tuple

def bayer_to_rgbg(rgbg : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    evens   = rgbg[0::2,:].astype(np.float32)
    odds    = rgbg[1::2,:].astype(np.float32)

    r   = evens[:,0::2]
    g1  = evens[:,1::2]
    b   = odds[:,1::2]
    g2  = odds[:,0::2]

    return (r,g1,b,g2)

def rgbg_to_bayer(r : np.ndarray, g1 : np.ndarray, b : np.ndarray, g2 : np.ndarray) -> np.ndarray:
    output = np.zeros(shape=(r.shape[0] * 2, r.shape[1] * 2), dtype=r.dtype)
    
    output[0::2,:][:,0::2] = r
    output[0::2,:][:,1::2] = g1
    output[1::2,:][:,1::2] = b
    output[1::2,:][:,0::2] = g2
    
    return output