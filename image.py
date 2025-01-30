from __future__ import annotations
import numpy as np
import exifread, rawpy
from typing import Optional, Union

from pySP.wb_cct.cam_wb import get_optimal_camera_mat_from_coords
from .normalization import bayer_normalize
from .debayer import debayer_ahd, debayer_fast
from .base_types.image_base import RawRgbgData_BaseType, RawDebayerData

from .const import QualityDemosaic
from math import log
from io import BytesIO

def compute_ev(iso : int, exp_time : float, f_stop : float) -> float:
    """Compute exposure value.

    Args:
        iso (int): Sensor gain.
        exp_time (float): Exposure time, seconds.
        f_stop (float): F-Stop, e.g., 1:3.5 corresponds to 3.5.

    Returns:
        float: Exposure value.
    """

    return log((100 * (f_stop * f_stop)) / (iso * exp_time), 2)

def compute_ev_from_exif(filename_or_data : Union[str, bytes]) -> float:
    """Compute exposure value from bundled EXIF data inside file.

    Args:
        filename_or_data (Union[str, bytes]): Either the filepath or bytes composing the raw file.

    Returns:
        float: Exposure value; np.inf if invalid.
    """

    exp_time = 1
    f_stop = 1
    iso = 100

    try:
        if type(filename_or_data) == str:
            with open(filename_or_data, 'rb') as raw:
                tags = exifread.process_file(raw)
        else:
            tags = exifread.process_file(BytesIO(filename_or_data))
    except:
        return np.inf
    
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

    return compute_ev(iso, exp_time, f_stop)

class RawRgbgData(RawRgbgData_BaseType):
    def __init__(self):
        """Base class for storing raw RGBG Bayer sensor data.
        """
        super().__init__()

    def debayer(self, quality : QualityDemosaic, postprocess_steps : int = 1) -> Optional[RawDebayerData]:
        """Debayers (demosaics) this image to a new RawDebayerData object.

        This does not modify the original data in any way. All image properties are copied to the new image.

        Args:
            quality (QualityDemosaic): Quality. Affects debayering algorithm; currently only shifting is supported which is fast but low quality.
            postprocess_steps (int, optional): Amount of divergence correction steps. Lower values retain detail but leave artifacts. Ignored unless using Best quality. Defaults to 1.

        Returns:
            Optional[RawDebayerData]: Debayered image; None if this image is not valid.
        """

        if quality == QualityDemosaic.Best:
            return debayer_ahd(self, postprocess_stages=postprocess_steps)
        else:
            return debayer_fast(self)

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
                self.wb_coeff = np.array(in_dng.camera_whitebalance[:3])
                self.bayer_data_scaled = bayer_normalize(in_dng.raw_image, chan_black, chan_sat)
            
            if type(filename_or_data) == str:
                with open(filename_or_data, 'rb') as raw:
                    tags = exifread.process_file(raw)
            else:
                tags = exifread.process_file(BytesIO(filename_or_data))
            
            self.mat_xyz = get_optimal_camera_mat_from_coords(tags, self.wb_coeff[1] / self.wb_coeff)
            
            self.current_ev = compute_ev_from_exif(filename_or_data)
            if self.current_ev != np.inf:
                self.__is_valid = True

        except (rawpy.LibRawError, FileNotFoundError, IOError) as e:
            self.__is_valid = False
    
    def is_valid(self) -> bool:
        return self.__is_valid

# TODO - This is due to be removed
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
                self._wb_coeff = in_dng.daylight_whitebalance
                self.image = in_dng.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                                                fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Full,
                                                gamma=(1,1),
                                                use_camera_wb=True,
                                                use_auto_wb=False,
                                                output_color=rawpy.ColorSpace.raw,
                                                output_bps=16,
                                                no_auto_bright=True,
                                                highlight_mode=rawpy.HighlightMode.Clip)
            
            if type(filename_or_data) == str:
                with open(filename_or_data, 'rb') as raw:
                    tags = exifread.process_file(raw)
            else:
                tags = exifread.process_file(BytesIO(filename_or_data))
            
            self.mat_xyz = get_optimal_camera_mat_from_coords(tags, self._wb_coeff)
            self.image = self.image.astype(np.float32) / ((2 ** 16) - 1)
            self.current_ev = compute_ev_from_exif(filename_or_data)

        except (rawpy.LibRawError, FileNotFoundError, IOError, OSError) as e:
            print(e)
    
        self._wb_applied = True
        self._wb_normalized = True