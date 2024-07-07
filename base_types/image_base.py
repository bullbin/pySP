from typing import Optional
import numpy as np

from pySP.const import QualityDemosaic

class RawDebayerData():
    def __init__(self, image : np.ndarray, wb_coeff : np.ndarray, wb_norm : bool = False):
        """Class for storing RGB pixel data after debayering.

        Args:
            image (np.ndarray): RGB pixel data in shape (height, width, 3). Should be normalized and with colors in camera-space after applying white balance correction.
            wb_coeff (np.ndarray): White balance co-efficients. Should have minimum length of 3.
            wb_norm (bool, optional): Optional flag to set whether white-balanced was applied in normalizing or naive way. Defaults to False, meaning naive application (multiplied through).
        """

        self.image              : np.ndarray = image
        self._wb_coeff          : np.ndarray = wb_coeff
        self._wb_applied        : bool = True
        self._wb_normalized     : bool = wb_norm
        
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

class RawRgbgData_BaseType():
    def __init__(self):
        """Base class for storing raw RGBG Bayer sensor data. This is primarily for subclasses and typing. Do not instantiate this class manually.
        """

        self.bayer_data_scaled  : np.ndarray = None
        self.wb_coeff           : np.ndarray = None
        self.mat_xyz            : np.ndarray = None
        self.current_ev         : float      = np.inf
        self.lim_sat            : float      = 1.0
        self.__is_hdr           : bool       = False

    def set_hdr(self, is_hdr : bool):
        """Set whether the image should be treated as HDR or not.

        Args:
            is_hdr (bool): HDR flag.
        """
        self.__is_hdr = is_hdr
    
    def get_hdr(self) -> bool:
        """Get whether the image is HDR or not.

        Returns:
            bool: True if HDR.
        """
        # TODO - Return True if any part of bayer_data_scaled is greater than 1. 
        return self.__is_hdr
    
    def is_valid(self) -> bool:
        """Check if contents of this image are valid, i.e., are expected.

        Returns:
            bool: True if image is valid; False otherwise.
        """
        return type(self.bayer_data_scaled) != type(None) and type(self.wb_coeff) != type(None) and type(self.mat_xyz) != type(None) and self.current_ev != np.inf

    def debayer(self, quality : QualityDemosaic, postprocess_steps : int = 1) -> Optional[RawDebayerData]:
        """Debayer this image. Override this method. This is intentionally unimplemented for the base class.

        Args:
            quality (QualityDemosaic): Quality. Affects debayering algorithm; currently only shifting is supported which is fast but low quality.
            postprocess_steps (int, optional): Amount of divergence correction steps. Lower values retain detail but leave artifacts. Ignored unless using Best quality. Defaults to 1.

        Returns:
            Optional[RawDebayerData]: Base class always returns None.
        """

        return None