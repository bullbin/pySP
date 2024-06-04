import rawpy, cv2
import numpy as np
from const import *
from demosaic import debayer
from colorize import cam_to_lin_srgb, lin_srgb_to_srgb
from normalization import bayer_normalize

PATH_IMAGE_DNG = "example_sat.dng"

with rawpy.imread(PATH_IMAGE_DNG) as in_dng:
    chan_sat = in_dng.camera_white_level_per_channel
    chan_black = in_dng.black_level_per_channel
    wb = in_dng.daylight_whitebalance
    mat = in_dng.rgb_xyz_matrix
    copy = np.copy(in_dng.raw_image)

debayered = bayer_normalize(copy, chan_black, chan_sat)
debayered = debayer(debayered, wb, PatternDemosaic.Rgbg, QualityDemosaic.Draft)
debayered = cv2.resize(debayered, (0,0), fx=0.1, fy=0.1)
debayered = cam_to_lin_srgb(debayered, mat, clip_highlights=False)
debayered = lin_srgb_to_srgb(debayered)

cv2.imshow("debug", cv2.cvtColor(debayered, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)