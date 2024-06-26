from __future__ import annotations
import numpy as np
import exifread, rawpy
from typing import Optional
from .normalization import bayer_normalize
from .bayer_chan_mixer import bayer_to_rgbg
import cv2
from .const import QualityDemosaic
from math import log

class RawRgbgData():
    def __init__(self):
        self.bayer_data_scaled  : np.ndarray = None
        self.wb_coeff           : np.ndarray = None
        self.mat_xyz            : np.ndarray = None
        self.current_ev         : float      = np.inf
        self.lim_sat            : float      = 1.0
    
    def is_valid(self) -> bool:
        return type(self.bayer_data_scaled) != type(None) and type(self.wb_coeff) != type(None) and type(self.mat_xyz) != type(None) and self.current_ev != np.inf

    def debayer(self, quality : QualityDemosaic) -> Optional[RawDebayerData]:
        if not(self.is_valid()):
            return None

        r, g1, b, g2 = bayer_to_rgbg(self.bayer_data_scaled)

        def debayer_nearest() -> np.ndarray:
            rgb = np.zeros((r.shape[0], r.shape[1], 3), dtype=np.float32)

            rgb[:,:,0] = r * self.wb_coeff[0]
            rgb[:,:,1] = ((g1 + g2) / 2) * self.wb_coeff[1]
            rgb[:,:,2] = b * self.wb_coeff[2]

            return cv2.resize(rgb, (self.bayer_data_scaled.shape[1], self.bayer_data_scaled.shape[0]))

        output = RawDebayerData(debayer_nearest(), np.copy(self.wb_coeff), wb_norm=False)
        output.mat_xyz = np.copy(self.mat_xyz)
        output.current_ev = self.current_ev
        return output

class RawRgbgDataFromRaw(RawRgbgData):
    def __init__(self, filename : str):
        super().__init__()
        self.__is_valid         : bool       = False

        try:
            with rawpy.imread(filename) as in_dng:
                chan_sat = in_dng.camera_white_level_per_channel
                chan_black = in_dng.black_level_per_channel
                self.wb_coeff = in_dng.daylight_whitebalance
                self.mat_xyz = in_dng.rgb_xyz_matrix
                self.bayer_data_scaled = bayer_normalize(in_dng.raw_image, chan_black, chan_sat)
            
            with open(filename, 'rb') as raw:
                tags = exifread.process_file(raw)
            
            exp_time = 1
            f_stop = 1
            iso = 100

            if "EXIF ExposureTime" in tags:
                exp_time = str(tags["EXIF ExposureTime"]).split("/")
                exp_time = float(exp_time[0]) / float(exp_time[1])
            
            if "EXIF FNumber" in tags:
                if "/" in str(tags["EXIF FNumber"]):
                    f_stop = str(tags["EXIF FNumber"]).split("/")
                    f_stop = float(f_stop[0]) / float(f_stop[1])
                else:
                    f_stop = int(str(tags["EXIF FNumber"]))

            if "ISOSpeed" in tags:
                iso = int(str(tags["ISOSpeed"]))
            elif "Image Make" in tags and str(tags["Image Make"]) == "Panasonic" and "Image Tag 0x0017" in tags:
                iso = int(str(tags["Image Tag 0x0017"]))

            self.current_ev = log((100 * (f_stop * f_stop)) / (iso * exp_time), 2)
            self.__is_valid = True

        except (rawpy.LibRawError, FileNotFoundError, IOError) as e:
            self.__is_valid = False
    
    def is_valid(self) -> bool:
        return self.__is_valid

class RawDebayerData():
    def __init__(self, image : np.ndarray, wb_coeff : np.ndarray, wb_norm : bool = False):
        self.image              : np.ndarray = image
        self._wb_coeff         : np.ndarray = wb_coeff
        self._wb_applied       : bool = True
        self._wb_normalized    : bool = wb_norm
        
        self.mat_xyz            : np.ndarray = None
        self.current_ev         : float = np.inf
        
    def is_valid(self) -> bool:
        return type(self.image) != type(None) and type(self._wb_coeff) != type(None) and type(self.mat_xyz) != type(None) and self.current_ev != np.inf
    
    def wb_apply(self):
        if not(self._wb_applied):
            self.image = (self.image * self._wb_coeff[:3]).astype(np.float32)
            self._wb_applied = True
    
    def wb_undo(self):
        if self._wb_applied:
            if self._wb_normalized:
                self.image = self.image * max(self._wb_coeff)
            self.image = (self.image.astype(np.float64) / self._wb_coeff[:3]).astype(np.float32)
            self._wb_applied = False
            self._wb_normalized = False

class RawDebayerDataFromRaw(RawDebayerData):
    def __init__(self, filename : str):
        super().__init__(None, None)
        
        try:
            with rawpy.imread(filename) as in_dng:
                self.mat_xyz = in_dng.rgb_xyz_matrix
                self._wb_coeff = in_dng.daylight_whitebalance
                self.image = in_dng.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                                                fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Full,
                                                gamma=(1,1),
                                                use_camera_wb=True,
                                                use_auto_wb=False,
                                                output_color=rawpy.ColorSpace.raw,
                                                output_bps=16,
                                                no_auto_bright=True,
                                                highlight_mode=rawpy.HighlightMode.Ignore)
            
            self.image = self.image.astype(np.float32) / ((2 ** 16) - 1)
            
            with open(filename, 'rb') as raw:
                tags = exifread.process_file(raw)
            
            exp_time = 1
            f_stop = 1
            iso = 100

            if "EXIF ExposureTime" in tags:
                exp_time = str(tags["EXIF ExposureTime"]).split("/")
                exp_time = float(exp_time[0]) / float(exp_time[1])
            
            if "EXIF FNumber" in tags:
                if "/" in str(tags["EXIF FNumber"]):
                    f_stop = str(tags["EXIF FNumber"]).split("/")
                    f_stop = float(f_stop[0]) / float(f_stop[1])
                else:
                    f_stop = int(str(tags["EXIF FNumber"]))

            if "ISOSpeed" in tags:
                iso = int(str(tags["ISOSpeed"]))
            elif "Image Make" in tags and str(tags["Image Make"]) == "Panasonic" and "Image Tag 0x0017" in tags:
                iso = int(str(tags["Image Tag 0x0017"]))

            self.current_ev = log((100 * (f_stop * f_stop)) / (iso * exp_time), 2)

        except (rawpy.LibRawError, FileNotFoundError, IOError) as e:
            print(e)
    
        self._wb_applied = True
        self._wb_normalized = True