import cv2
import numpy as np
from const import *
from colorize import cam_to_lin_srgb
from raw_bad_pixel_corr import repair_bad_pixels
from colorize import cam_to_lin_srgb, lin_srgb_to_srgb
from image import RawRgbgDataFromRaw

PATH_IMAGE_DNG = "s1.dng"

image = RawRgbgDataFromRaw(PATH_IMAGE_DNG)
image = image.debayer(QualityDemosaic.Best)
debayered = cam_to_lin_srgb(image.image, image.mat_xyz, clip_highlights=True)
debayered = lin_srgb_to_srgb(debayered)

cv2.imwrite("debug.png", cv2.cvtColor(debayered * 255, cv2.COLOR_RGB2BGR))