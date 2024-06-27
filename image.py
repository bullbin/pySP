from __future__ import annotations
import numpy as np
import exifread, rawpy
from typing import Optional, Union
from .normalization import bayer_normalize
from .bayer_chan_mixer import bayer_to_rgbg
import cv2
from .const import QualityDemosaic
from math import log
from io import BytesIO

class RawRgbgData():
    def __init__(self):
        """Base class for storing raw RGBG Bayer sensor data.
        """

        self.bayer_data_scaled  : np.ndarray = None
        self.wb_coeff           : np.ndarray = None
        self.mat_xyz            : np.ndarray = None
        self.current_ev         : float      = np.inf
        self.lim_sat            : float      = 1.0
    
    def is_valid(self) -> bool:
        """Check if contents of this image are valid, i.e., are expected.

        Returns:
            bool: True if image is valid; False otherwise.
        """
        return type(self.bayer_data_scaled) != type(None) and type(self.wb_coeff) != type(None) and type(self.mat_xyz) != type(None) and self.current_ev != np.inf

    def debayer(self, quality : QualityDemosaic) -> Optional[RawDebayerData]:
        """Debayers (demosaics) this image to a new RawDebayerData object.

        This does not modify the original data in any way. All image properties are copied to the new image.

        Args:
            quality (QualityDemosaic): Quality. Affects debayering algorithm; currently only shifting is supported which is fast but low quality.

        Returns:
            Optional[RawDebayerData]: Debayered image; None if this image is not valid.
        """

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
    def __init__(self, filename_or_data : Union[str, bytes]):
        """Class for storing RGBG Bayer sensor data from a raw file.

        For loading to produce a valid image, the raw file type must be supported by both rawpy and
        exifread so required metadata can be extracted. Most filetypes should work out of the box,
        e.g., DNG is well-supported.

        Args:
            filename_or_data (Union[str, bytes]): Either the filepath or bytes composing the raw file.
        """

        super().__init__()
        self.__is_valid         : bool       = False

        try:
            reader = filename_or_data
            if type(filename_or_data) != str:
                reader = BytesIO(filename_or_data)

            with rawpy.imread(reader) as in_dng:
                chan_sat = in_dng.camera_white_level_per_channel
                chan_black = in_dng.black_level_per_channel
                self.wb_coeff = in_dng.daylight_whitebalance
                self.mat_xyz = in_dng.rgb_xyz_matrix
                self.bayer_data_scaled = bayer_normalize(in_dng.raw_image, chan_black, chan_sat)
            
            if type(filename_or_data) == str:
                with open(filename_or_data, 'rb') as raw:
                    tags = exifread.process_file(raw)
            else:
                tags = exifread.process_file(reader)
            
            exp_time = 1
            f_stop = 1
            iso = 100

            if "EXIF ExposureTime" in tags:
                if "/" in str(tags["EXIF ExposureTime"]):
                    exp_time = str(tags["EXIF ExposureTime"]).split("/")
                    exp_time = float(exp_time[0]) / float(exp_time[1])
                else:
                    exp_time = float(str(tags["EXIF ExposureTime"]))
            
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
        """Class for storing RGB pixel data after debayering.

        Args:
            image (np.ndarray): RGB pixel data in shape (height, width, 3). Should be normalized and with colors in camera-space after applying white balance correction.
            wb_coeff (np.ndarray): White balance co-efficients. Should have minimum length of 3.
            wb_norm (bool, optional): Optional flag to set whether white-balanced was applied in normalizing or naive way. Defaults to False, meaning naive application (multiplied through).
        """

        self.image              : np.ndarray = image
        self._wb_coeff         : np.ndarray = wb_coeff
        self._wb_applied       : bool = True
        self._wb_normalized    : bool = wb_norm
        
        self.mat_xyz            : np.ndarray = None
        self.current_ev         : float = np.inf
        
    def is_valid(self) -> bool:
        """Check if contents of this image are valid, i.e., are expected.

        Returns:
            bool: True if image is valid; False otherwise.
        """
        return type(self.image) != type(None) and type(self._wb_coeff) != type(None) and type(self.mat_xyz) != type(None) and self.current_ev != np.inf
    
    def wb_apply(self):
        """Apply white balance co-efficients if not already applied.
        """
        if not(self._wb_applied):
            self.image = (self.image * self._wb_coeff[:3]).astype(np.float32)
            self._wb_applied = True
    
    def wb_undo(self):
        """Undo white balance to return to pure camera-space if white balance was applied. Removes normalization in process.
        """
        if self._wb_applied:
            if self._wb_normalized:
                self.image = self.image * max(self._wb_coeff)
            self.image = (self.image.astype(np.float64) / self._wb_coeff[:3]).astype(np.float32)
            self._wb_applied = False
            self._wb_normalized = False

class RawDebayerDataFromRaw(RawDebayerData):
    def __init__(self, filename_or_data : Union[str, bytes]):
        """Class for storing RGB demosaiced data from a raw file.

        This class exists as a means to use higher-quality demosaic algorithms in external
        libraries; by default, AHD in libraw is used with FBDD noise reduction.

        Raws processed using this class are largely identical to manual processing with
        provided debayering algorithms; colors and contrast are very slightly different.

        Args:
            filename_or_data (Union[str, bytes]): Either the filepath or bytes composing the raw file.
        """
        super().__init__(None, None)
        
        try:
            reader = filename_or_data
            if type(filename_or_data) != str:
                reader = BytesIO(filename_or_data)

            with rawpy.imread(reader) as in_dng:
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
            
            if type(filename_or_data) == str:
                with open(filename_or_data, 'rb') as raw:
                    tags = exifread.process_file(raw)
            else:
                tags = exifread.process_file(reader)
            
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