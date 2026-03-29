from __future__ import annotations
import numpy as np
import exifread, rawpy
from typing import Dict, List, Optional, Tuple, Union

from tifftools import read_tiff as tt_read_tiff, Datatype as tt_Datatype, Tag as tt_Tag

from pySP.wb_cct.cam_wb import CameraWhiteBalanceControllerFromExif
from .normalization import bayer_normalize
from .debayer import debayer_ahd, debayer_fast, debayer_eag
from .base_types.image_base import BayerPattern, RawBayerData_BaseType, RawRggbBayerData_BaseType, RawDemosaicData

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

def get_image_area_from_tiff(filename_or_data : Union[str, bytes]) -> Tuple[Optional[List[int,int,int,int]], Optional[Tuple[List[int,int], List[int,int]]]]:
    # Note - this extracts the crop zone as the absolute area to crop, so uses a few tags to do that
    #        A small margin should be provided so that demosaic algorithms can access pixels beyond the viewable
    #            area for best results near the edge
    # Not all valid DNG files will contain required data for this tag to function. Software often just keeps
    #     a database of cameras and their actual imaging area. You really want this to be perfect for lens operations
    #     that assume the center of image to be the imaging center - active area being wrong ruins this
    # In my Lumix camera the area beyond this area isn't good data, it's chunks of the image being repeated. This is
    #     probably just extra readout instead of sensor data

    # TODO - I don't like using tifftools, too much can go wrong with IFD not applying to same IFD as raw... assumptions assumptions
    # TODO - tifftools also needed for some opcode3 stuff. Should probably just keep it always loaded in raw class
    # TODO - as an aside, raw levels stored in this can also differ quite radically from libraw defaults. Some old Lumix files
    #            convert with white level 2800 which is wrong. Libraw corrects this to 4095 which is more correct

    def decode_tiff_data(datatag : Dict[str, Union[List[int], bytes, List[float]]]) -> Optional[List[Union[int,float]]]:
        try:
            pack = tt_Datatype.get(datatag["datatype"]).pack
        except AttributeError:
            return None
        
        target = datatag["data"]

        # This is disgusting but it works, to try to be stable against changes we're testing against binary unpacking codes for dtypes
        if pack in ['B', 'H', 'L', 'b', 'h', 'l', 'f', 'd', 'Q', 'q']:
            return target
        else:
            if len(target) % 2 != 0:
                return None
            
            evens = [x for i,x in enumerate(target) if i % 2 == 0]
            odds = [x for i,x in enumerate(target) if i % 2 == 1]

            if set(odds) == {1}:
                return evens

            output = [x/y for x,y in zip(evens,odds)]
            for i, x in enumerate(output):
                if x.is_integer():
                    output[i] = int(output[i])

            return output

    if type(filename_or_data) == bytes:
        filename_or_data = BytesIO(filename_or_data)
    
    try:
        info = tt_read_tiff(filename_or_data)
    except:
        return None

    raw_ifd_tags = info['ifds'][0]['tags'][tt_Tag.SubIFD.value]['ifds'][0][0]['tags']

    tag_active_area = decode_tiff_data(raw_ifd_tags[50829])
    tag_crop_start = decode_tiff_data(raw_ifd_tags[50719])
    tag_crop_length = decode_tiff_data(raw_ifd_tags[50720])

    if tag_crop_start == None or tag_crop_length == None:
        return (tag_active_area, None)
    return (tag_active_area, (tag_crop_start, tag_crop_length))

def reversible_transform_rggb(sensor_data : np.ndarray, bayer_pattern : BayerPattern):
    if bayer_pattern == BayerPattern.Rggb:
        return sensor_data
    elif bayer_pattern == BayerPattern.Bggr:
        return np.rot90(sensor_data, k=2)
    elif bayer_pattern == BayerPattern.Gbrg:
        return np.flip(sensor_data, axis=1)
    elif bayer_pattern == BayerPattern.Grbg:
        return np.flip(sensor_data, axis=0)
    raise NotImplementedError(str(bayer_pattern) + " not implemented!")

class RawRggbBayerData(RawRggbBayerData_BaseType):

    def demosaic(self, quality : QualityDemosaic, postprocess_steps : int = 1) -> RawDemosaicData:
        """Debayers (demosaics) this image to a new RawDebayerData object.

        This does not modify the original data in any way. All image properties are copied to the new image.

        Args:
            quality (QualityDemosaic): Quality. Affects debayering algorithm; currently only shifting is supported which is fast but low quality.
            postprocess_steps (int, optional): Amount of divergence correction steps. Lower values retain detail but leave artifacts. Ignored unless using Best quality. Defaults to 1.

        Returns:
            Optional[RawDebayerData]: Debayered image; None if this image is not valid.
        """

        if quality == QualityDemosaic.Best:
            debayered = debayer_ahd(self, postprocess_stages=postprocess_steps)
        elif quality == QualityDemosaic.Fast:
            debayered = debayer_eag(self)
        elif quality == QualityDemosaic.Draft:
            debayered = debayer_fast(self)
        else:
            raise NotImplementedError("Quality mode not implemented: %s" % str(quality))
        
        # TODO - Change this to remove WB from demosaic methods

        # Revert any transforms needed to make data RGGB
        debayered.image = reversible_transform_rggb(debayered.image, self.source_pattern)

        return debayered

class RawBayerData(RawBayerData_BaseType):
    def __init__(self):
        """Base class for storing raw RGBG Bayer sensor data.
        """
        super().__init__()

    def to_rggb(self):
        rggb = reversible_transform_rggb(self.sensor_scaled, self.sensor_pattern)
        return RawRggbBayerData(rggb, self.cam_wb.copy(), self.current_ev, self.lim_sat, self.sensor_pattern)
    
    def demosaic(self, quality : QualityDemosaic, postprocess_steps : int = 1) -> RawDemosaicData:
        rggb = self.to_rggb()
        return rggb.demosaic(quality, postprocess_steps)

class RawBayerDataFromRaw(RawBayerData):
    def __init__(self, filename_or_data : Union[str, bytes]):
        """Class for storing RGBG Bayer sensor data from a raw file.

        For loading to produce a valid image, the raw file type must be supported by both rawpy and
        exifread so required metadata can be extracted. Most filetypes should work out of the box,
        e.g., DNG is well-supported.

        Args:
            filename_or_data (Union[str, bytes]): Either the filepath or bytes composing the raw file.
        """

        super().__init__()
        self._region_crop : Optional[Tuple[Tuple[int,int], Tuple[int,int]]] = None

        try:
            reader = filename_or_data
            if type(filename_or_data) != str:
                reader = BytesIO(filename_or_data)
            
            image_area_param = get_image_area_from_tiff(filename_or_data)
            if image_area_param != None:
                region_active_area, region_crop_data = image_area_param
                try:
                    self._region_crop = ((region_crop_data[0][0], region_crop_data[0][1]), (region_crop_data[1][0], region_crop_data[1][1]))
                except IndexError:
                    pass

            with rawpy.imread(reader) as in_dng:
                # TODO - This might change depending on Bayer configuration. So far every file I've seen has had equal
                #        saturation and black values on every channel.
                chan_sat = in_dng.camera_white_level_per_channel
                chan_black = in_dng.black_level_per_channel
                self.sensor_scaled = bayer_normalize(in_dng.raw_image, chan_black, chan_sat)

                if in_dng.raw_pattern.shape != (2,2):
                    raise ValueError("Raw has unsupported Bayer pattern, cannot continue!")
                
                try:
                    raw_cfa_planes = in_dng.color_desc.decode('ascii')
                except UnicodeDecodeError:
                    raise ValueError("Raw has unknown color array, %s" % str(in_dng.color_desc))
                
                if ''.join(sorted(list(set(raw_cfa_planes.upper())))) != "BGR":
                    raise ValueError("Raw has unsupported colors, %s" % raw_cfa_planes)
                
                try:
                    raw_cfa_decoded_pattern = ''.join(raw_cfa_planes[i] for i in in_dng.raw_pattern.flatten())
                except IndexError:
                    raise ValueError("Raw tried to index out-of-bounds color filter, malformed input!")
                
                if raw_cfa_decoded_pattern == "BGGR":
                    self.sensor_pattern = BayerPattern.Bggr
                elif raw_cfa_decoded_pattern == "RGGB":
                    self.sensor_pattern = BayerPattern.Rggb
                elif raw_cfa_decoded_pattern == "GBRG":
                    self.sensor_pattern = BayerPattern.Gbrg
                elif raw_cfa_decoded_pattern == "GRBG":
                    self.sensor_pattern = BayerPattern.Grbg
                else:
                    raise NotImplementedError(f"Bayer pattern {raw_cfa_decoded_pattern} is not supported!")
                
                # If active masking is enabled, remove inactive areas from the sensor
                if region_active_area != None:
                    x_start, x_end = region_active_area[1], region_active_area[3] + 1
                    y_start, y_end = region_active_area[0], region_active_area[2] + 1

                    x_start = np.clip(x_start, 0, self.sensor_scaled.shape[1])
                    x_end   = np.clip(x_end  , 0, self.sensor_scaled.shape[1])
                    y_start = np.clip(y_start, 0, self.sensor_scaled.shape[0])
                    y_end   = np.clip(y_end  , 0, self.sensor_scaled.shape[0])

                    self.sensor_scaled = self.sensor_scaled[y_start:y_end, x_start:x_end]
                
                if self._region_crop != None:
                    # For safety, crop the sensor region to only use the export area on the sensor
                    # This will worsen demosaic quality on the very edges.
                    # TODO - Make edge param more available so we can keep most of data and crop only at end
                    #        Some arrangements mean that the image center is not the sensor center so we have to crop before
                    #        lens operations
                    region_start, region_len = self._region_crop

                    # If these are bigger than the filter array size (2), this changes the filter order.
                    if region_start[0] % 2 != 0 or region_start[1] % 2 != 0:
                        raise NotImplementedError("Sensor crop start would modify CFA pattern order. Not implemented!")
                    if region_len[0] % 2 != 0 or region_len[1] % 2 != 0:
                        raise NotImplementedError("Sensor crop length would cut the CFA array. Not implemented!")
                    
                    # start is horizontal (x), vertical (y)
                    r_s_x = np.clip(region_start[0], 0, self.sensor_scaled.shape[1] - 1)
                    r_s_y = np.clip(region_start[1], 0, self.sensor_scaled.shape[0] - 1)

                    # len is width, height
                    r_e_x = np.clip(r_s_x + region_len[0], r_s_x + 1, self.sensor_scaled.shape[1])
                    r_e_y = np.clip(r_s_y + region_len[1], r_s_y + 1, self.sensor_scaled.shape[0])

                    self.sensor_scaled = self.sensor_scaled[r_s_y:r_e_y, r_s_x:r_e_x]
            
            if type(filename_or_data) == str:
                with open(filename_or_data, 'rb') as raw:
                    tags = exifread.process_file(raw)
            else:
                tags = exifread.process_file(BytesIO(filename_or_data))
            
            self.cam_wb = CameraWhiteBalanceControllerFromExif(tags)
            
            self.current_ev = compute_ev_from_exif(filename_or_data)
            if self.current_ev == np.inf:
                raise ValueError("Error reading exposure value from raw!")

        except (rawpy.LibRawError, FileNotFoundError, IOError) as e:
            raise ValueError("Raw couldn't be read! " + str(e))

class RawDebayerDataFromRaw(RawDemosaicData):
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
            
            cont = CameraWhiteBalanceControllerFromExif(tags)
            cont.update_by_reference(self._wb_coeff)    # TODO - Check this isn't normalized
            self.mat_xyz = cont.get_matrix()
            self.image = self.image.astype(np.float32) / ((2 ** 16) - 1)
            self.current_ev = compute_ev_from_exif(filename_or_data)

        except (rawpy.LibRawError, FileNotFoundError, IOError, OSError) as e:
            raise ValueError("Input raw couldn't be read! " + str(e))
    
        self._wb_applied = True
        self._wb_normalized = True