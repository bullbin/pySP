from io import BytesIO
from struct import unpack
from typing import Optional, Union

import cv2
import numpy as np
import tifftools

from .dng_warp_rectilinear_coords import compute_remapping_table, compute_offset_remapping_table

def stack_warp_prior(demosaiced_image : np.ndarray, remap_r : Optional[np.ndarray], remap_g : Optional[np.ndarray],
                     remap_b : Optional[np.ndarray]) -> np.ndarray:
    """Combine per-channel warps into one array for use as a prior for apply_opcode_3_warp.

    Mappings are given as a cv2.remap style matrix where map[y,x] = new_x,new_y.

    Args:
        demosaiced_image (np.ndarray): Input image. Used to compute bypass mapping where channel is skipped.
        remap_r (Optional[np.ndarray]): Channel mapping for red. None if no custom mapping is used.
        remap_g (Optional[np.ndarray]): Channel mapping for green. None if no custom mapping is used.
        remap_b (Optional[np.ndarray]): Channel mapping for blue. None if no custom mapping is used.

    Returns:
        np.ndarray: Prior matrix.
    """

    if remap_r is None or remap_g is None or remap_b is None:
        arr_empty = np.zeros(shape=(demosaiced_image.shape[0], demosaiced_image.shape[1], 2), dtype=np.float32)
        for x in range(demosaiced_image.shape[1]):
            arr_empty[:,x,0] = x
        for y in range(demosaiced_image.shape[0]):
            arr_empty[y,:,1] = y

        if remap_r is None:
            remap_r = arr_empty
        if remap_g is None:
            remap_g = arr_empty
        if remap_b is None:
            remap_b = arr_empty

    return np.stack((remap_r, remap_g, remap_b), axis=2)

def apply_opcode_3_warp(demosaiced_image : np.ndarray, ifd_opcode_3_data : bytes, scale : float = 1.0, prior : Optional[np.ndarray] = None):
    """Apply the WarpRectilinear distortion correction operator from DNG data. Other opcodes will be skipped.
    Operations are applied in-place and in-order. While our WarpRectilinear implementation roughly follows
    spec, it may not match other raw processors that are closer to spec (i.e., use clamping) or have support
    for other warping operators.

    Args:
        demosaiced_image (np.ndarray): Raw image after demosaicing.
        ifd_opcode_3_data (bytes): Full data section for the OpcodeList3 block.
        prior (Optional[np.ndarray]): Prior mapping before warping. Should have shape [height,width,channels,2]. Defaults to None, so returns new transform.
    """

    def opcode_warp_rectilinear(image : np.ndarray, data : bytes) -> bool:
        """Apply an inverted WarpRectilinear operator in-place on an image.

        This will fail if the provided data is invalid or doesn't match the amount of color channels on the input.

        Args:
            image (np.ndarray): Input image. Should be raw with shape (height, width, channels). Channels must match stored DNG channel count.
            data (bytes): Operator data (variable length block after opcode preamble).

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
                map_dist_corr = compute_remapping_table(kr0, kr1, kr2, kr3, kt0, kt1, image.shape[1], image.shape[0], cam_center_normalized[0], cam_center_normalized[1], scale)
            else:
                map_dist_corr = compute_offset_remapping_table(prior[...,idx_coeff,:], kr0, kr1, kr2, kr3, kt0, kt1, image.shape[1], image.shape[0], cam_center_normalized[0], cam_center_normalized[1], scale)

            # Apply warp
            image[:, :, idx_coeff] = cv2.remap(image[:, :, idx_coeff],
                                               np.clip(map_dist_corr[:,:,0], 0, map_dist_corr.shape[1] - 1),
                                               np.clip(map_dist_corr[:,:,1], 0, map_dist_corr.shape[0] - 1),
                                               cv2.INTER_LANCZOS4)
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
            opcode_warp_rectilinear(demosaiced_image, ifd_opcode_3_data[offset:offset + opcode_var_len])
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