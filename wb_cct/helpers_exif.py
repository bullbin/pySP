from pySP.wb_cct.helpers_cam_mat import MatXyzToCamera
from pySP.wb_cct.standard_ill import get_chromacity_from_illuminant, get_illuminant_from_lightsource, get_series_from_illuminant
from typing import Optional, List, Dict, Any
from colour import xy_to_XYZ
import numpy as np

# TODO - !=3 color plane support

def exif_get_color_mat_sources(tags : Dict[str, Any]) -> List[MatXyzToCamera]:
    """Extract XYZ to reference matrices from EXIF data.

    Args:
        tags (Dict[str, Any]): Tags dictionary as held by exifread.

    Returns:
        List[MatXyzToCamera]: List of camera matrices. If length is zero, none could be extracted.
    """

    # TODO - Support custom illuminant (exifread might not be helpful here)

    def get_mat(idx : int) -> Optional[MatXyzToCamera]:
        if idx < 0 or idx > 3:
            return None
        
        tag_mat = 0xC621 + idx
        tag_light = 0xC65A + idx
        
        tag_mat = "Image Tag 0x%s" % hex(tag_mat)[2:].upper()
        tag_light = "Image Tag 0x%s" % hex(tag_light)[2:].upper()

        if not(tag_mat in tags and tag_light in tags):
            return None
        
        try:
            ill = get_illuminant_from_lightsource(tags[tag_light].values[0])
            xy = get_chromacity_from_illuminant(ill)
            series = get_series_from_illuminant(ill)
        except KeyError:
            return None

        mat = np.zeros((3,3), dtype=np.float32)

        idx = 0
        for y in range(mat.shape[0]):
            for x in range(mat.shape[1]):
                mat[y,x] = tags[tag_mat].values[idx].decimal()
                idx += 1

        # DNG stores XYZ to camera, we want the other way around
        return MatXyzToCamera(mat, xy_to_XYZ(xy), series)
    
    output = []
    id = 0
    while id < 3:
        mat = get_mat(id)
        id += 1
        
        if mat == None:
            break
        output.append(mat)
    
    return output

def exif_get_as_shot_neutral(tags : Dict[str, Any]) -> np.ndarray:
    """Extract neutral multipliers from EXIF tag. These multipliers are for rough white balancing directly in-camera
    and have been pre-adjusted to minimize tint. Ideally, multiplying neutral with an appropriate interpolation
    of the ColorMatrix should result in a value somewhat close to the Planckian locus.

    Args:
        tags (Dict[str, Any]): Tags dictionary as held by exifread.

    Raises:
        KeyError: Raised if AsShotNeutral was not in the tag dictionary.

    Returns:
        np.ndarray: AsShotNeutral channel multipliers.
    """

    as_shot_neutral = np.zeros(3, dtype=np.float32)
    idx = 0
    for x in range(as_shot_neutral.shape[0]):
        try:
            as_shot_neutral[x] = tags["Image Tag 0xC628"].values[idx].decimal()
        except:
            raise KeyError("AsShotNeutral missing inside tags!")
        idx += 1
    return as_shot_neutral