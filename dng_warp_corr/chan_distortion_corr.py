from io import BytesIO
from struct import unpack
from typing import Optional, Union

import cv2
import numpy as np
import tifftools

from .dng_warp_rectilinear_coords import compute_remapping_table, compute_offset_remapping_table

def invert_remapping(mapping : np.ndarray):
    """Invert an image mapping table in-place.

    Mapping table needs shape [y_in,x_in,x_out,y_out]. x,y should have same dimensions as image.
    
    The output map is clamped so it can be used immediately.

    Args:
        mapping (np.ndarray): Input mapping.
    """

    x_map = mapping[:,:,0]
    y_map = mapping[:,:,1]

    x_map_empty = np.ones_like(x_map)
    y_map_empty = np.ones_like(y_map)
    for x in range(mapping.shape[1]):
        x_map_empty[:, x] = x
    for y in range(mapping.shape[0]):
        y_map_empty[y, :] = y
    
    o_x =  x_map - x_map_empty      # a -> b = b - a
    x_map = x_map_empty - o_x

    o_y =  y_map - y_map_empty
    y_map = y_map_empty - o_y

    x_map = np.clip(x_map, 0, mapping.shape[1])
    y_map = np.clip(y_map, 0, mapping.shape[0])
    
    mapping[:, :, 0] = x_map
    mapping[:, :, 1] = y_map

def apply_opcode_3_warp(demosaiced_image : np.ndarray, ifd_opcode_3_data : bytes, invert_warp : bool, prior : Optional[np.ndarray] = None):
    """Apply the WarpRectilinear distortion correction operator from DNG data. Other opcodes will be skipped.
    Operations are applied in-place and in-order. While our WarpRectilinear implementation roughly follows
    spec, it may not match other raw processors that are closer to spec (i.e., use clamping) or have support
    for other warping operators.

    Typically distortion is corrected in the inverse direction to remove it from the image. For the purposes
    of slide correction, however, applying it forwards tends to make scans flatter. Warp direction can be set
    using the invert_warp argument.

    Args:
        demosaiced_image (np.ndarray): Raw image after demosaicing.
        ifd_opcode_3_data (bytes): Full data section for the OpcodeList3 block.
        invert_warp (bool): True to invert warp, False to use as-is.
        prior (Optional[np.ndarray]): Prior mapping before warping. Should have shape [height,width,channels,2]. Defaults to None, so returns new transform.
    """

    def opcode_warp_rectilinear(image : np.ndarray, data : bytes, invert_warp : bool) -> bool:
        """Apply an inverted WarpRectilinear operator in-place on an image.

        This will fail if the provided data is invalid or doesn't match the amount of color channels on the input.

        Args:
            image (np.ndarray): Input image. Should be raw with shape (height, width, channels). Channels must match stored DNG channel count.
            data (bytes): Operator data (variable length block after opcode preamble).
            invert_warp (bool): True to invert warp, False to use as-is.

        Returns:
            bool: True if warp completed, False otherwise.
        """
        # Credit - Adobe, DNG Specification 1.4.0.0

        # Exit if no plane definition
        if len(data) < 4:
            return False

        # Exit if length is invalid or plane mismatch
        count_planes = int.from_bytes(data[:4], byteorder='big')
        if len(data) != 4 + (6 * 8 * count_planes) + 16 or count_planes != image.shape[2]:
            return False
        
        # Decode operator data
        coefficients = []
        for idx_plane in range(count_planes):
            coefficients.append(unpack(">6d", data[4 + (6 * 8 * idx_plane) : 4 + (6 * 8 * (idx_plane + 1))]))
        
        cam_center_normalized = unpack(">2d", data[4 + (6 * 8 * count_planes) : 4 + (6 * 8 * count_planes) + 16])

        # Apply per-channel correction
        for idx_coeff, coefficient in enumerate(coefficients):
            kr0, kr1, kr2, kr3, kt0, kt1 = coefficient
            if prior is None:
                map_dist_corr = compute_remapping_table(kr0, kr1, kr2, kr3, kt0, kt1, image.shape[1], image.shape[0], cam_center_normalized[0], cam_center_normalized[1])
            else:
                map_dist_corr = compute_offset_remapping_table(prior[...,idx_coeff,:], kr0, kr1, kr2, kr3, kt0, kt1, image.shape[1], image.shape[0], cam_center_normalized[0], cam_center_normalized[1])

            if invert_warp and prior is None:
                # Invert the mapping. This is probably not intended but it makes the image flatter
                #     This held true with another camera too (older version of digitizer based on a compact camera)
                invert_remapping(map_dist_corr)

            # Apply warp
            image[:, :, idx_coeff] = cv2.remap(image[:, :, idx_coeff], map_dist_corr[:,:,0], map_dist_corr[:,:,1], cv2.INTER_LANCZOS4)
        return True

    assert prior is None or prior.shape == (demosaiced_image.shape[0], demosaiced_image.shape[1], demosaiced_image.shape[2], 2)

    count_opcodes = int.from_bytes(ifd_opcode_3_data[:4], byteorder = 'big')
    offset = 4

    for _idx_opcode in range(count_opcodes):
        opcode_id = int.from_bytes(ifd_opcode_3_data[offset:offset + 4], byteorder='big')
        _opcode_ver = int.from_bytes(ifd_opcode_3_data[offset + 4:offset + 8], byteorder='big')
        opcode_flags = int.from_bytes(ifd_opcode_3_data[offset + 8:offset + 12], byteorder='big')
        opcode_var_len = int.from_bytes(ifd_opcode_3_data[offset + 12:offset + 16], byteorder='big')
        
        _opcode_optional = opcode_flags & 0x01 > 0
        _opcode_preview = opcode_flags & 0x02 > 0
        
        offset += 16

        if opcode_id == 1:
            opcode_warp_rectilinear(demosaiced_image, ifd_opcode_3_data[offset:offset + opcode_var_len], invert_warp)
        else:
            print("Unimplemented opcode %d" % opcode_id)

        offset += opcode_var_len

def get_opcode_3_block(filename_or_data : Union[str, bytes]) -> Optional[bytes]:
    """Extract the OpcodeList3 data block from a DNG image.

    OpcodeList3 is responsible for post-demosaic corrections like vignetting, chromatic abberation
    or distortion correction.

    Args:
        filename_or_data (Union[str, bytes]): Either the filepath or bytes composing the raw file.

    Returns:
        Optional[bytes]: Data block after preamble if valid, None otherwise.
    """
    if type(filename_or_data) == bytes:
        filename_or_data = BytesIO(filename_or_data)
    
    try:
        info = tifftools.read_tiff(filename_or_data)
    except:
        return None

    try:
        return info['ifds'][0]['tags'][tifftools.Tag.SubIFD.value]['ifds'][0][0]['tags'][51022]['data']
    except KeyError:
        return None