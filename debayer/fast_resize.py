from typing import Optional

import cv2
import numpy as np

from pySP.bayer_chan_mixer import bayer_to_rgbg
from pySP.base_types.image_base import RawDebayerData, RawRgbgData_BaseType

def debayer(image : RawRgbgData_BaseType) -> Optional[RawDebayerData]:
    """Debayer by resizing channels to fit original resolution.

    This is fast but produces low-quality results. Divergence is left uncorrected.

    Args:
        image (RawRgbgData_BaseType): Bayer image with RGBG pattern.

    Returns:
        Optional[RawDebayerData]: Debayered image; None if image is not valid.
    """

    if not(image.is_valid()):
        return None

    def debayer_nearest() -> np.ndarray:
        r, g1, b, g2 = bayer_to_rgbg(image.bayer_data_scaled)

        rgb = np.zeros((r.shape[0], r.shape[1], 3), dtype=np.float32)

        rgb[:,:,1] = ((g1 + g2) / 2) * image.wb_coeff[1]

        r = cv2.copyMakeBorder(r, 0, 1, 0, 1, cv2.BORDER_REFLECT)
        b = cv2.copyMakeBorder(b, 1, 0, 1, 0, cv2.BORDER_REFLECT)

        r_diag = r[1:,1:]
        r = 0.75 * r[:-1,:-1] + 0.25 * r_diag
        b_diag = b[:-1, :-1]
        b = 0.75 * b[1:,1:] + 0.25 * b_diag

        rgb[:,:,0] = r * image.wb_coeff[0]
        rgb[:,:,2] = b * image.wb_coeff[2]

        return cv2.resize(rgb, (image.bayer_data_scaled.shape[1], image.bayer_data_scaled.shape[0]))

    output = RawDebayerData(debayer_nearest(), np.copy(image.wb_coeff), wb_norm=False)
    output.mat_xyz = np.copy(image.mat_xyz)
    output.current_ev = image.current_ev
    return output