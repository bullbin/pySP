import numpy as np

from pySP.wb_cct.helpers_cam_mat import MatXyzToCamera, bradford_adapt_matrix
from pySP.wb_cct.standard_ill import StandardIlluminant, get_chromacity_from_illuminant
from colour import xy_to_XYZ

def clip_rgb(rgb : np.ndarray) -> np.ndarray:
    """Clip an RGB image to [0,1].

    Args:
        rgb (np.ndarray): RGB image.

    Returns:
        np.ndarray: Clipped RGB image.
    """
    out = np.zeros_like(rgb)
    out[:,:,0] = np.clip(rgb[:,:,0], 0, 1)
    out[:,:,1] = np.clip(rgb[:,:,1], 0, 1)
    out[:,:,2] = np.clip(rgb[:,:,2], 0, 1)
    return out

def cam_to_lin_srgb(rgb : np.ndarray, cam_xyz_matrix : MatXyzToCamera, clip_highlights : bool = True) -> np.ndarray:
    """Convert an input image from camera-space to linearized sRGB colors.

    Args:
        rgb (np.ndarray): Debayered normalized camera-space RGB image.
        cam_xyz_matrix (np.ndarray): Camera color matrix; this is supplied by the image object.
        clip_highlights (bool, optional): Whether to clip highlights to reduce false highlight colors. Reduces color artifacts at cost of detail. Defaults to True.

    Returns:
        np.ndarray: Linearized sRGB image.
    """
    # Transform variable cable white to fixed white point using Bradform adaptation (proposed in DNG spec)
    ca_mat = bradford_adapt_matrix(xy_to_XYZ(get_chromacity_from_illuminant(StandardIlluminant.D65)), cam_xyz_matrix.xyz)
    
    # Mat to transform XYZd65 to linearized sRGB, http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    mat_srgb_d65 = np.array([[0.412453, 0.357580, 0.180423],
                             [0.212671, 0.715160, 0.072169],
                             [0.019334, 0.119193, 0.950227]])
    
    # TODO - Highlight recovery, allow for changing highlight peak to allow for subpixel HDR
    if clip_highlights:
        rgb = clip_rgb(rgb)

    # Credit - dcraw, https://ninedegreesbelow.com/files/dcraw-c-code-annotated-code.html#E3
    # Credit - https://stackoverflow.com/questions/8904694/how-to-normalize-a-2-dimensional-numpy-array-in-python-less-verbose

    # Form color output matrix
    # Color matrix produces RGB (linear, D65) -> R'G'B' (reference space)
    # This is composed of the following transforms in sequence:
    #     RGB -> XYZd65         mat_srgb_d65
    #     XYZd65 -> XYZdCam     ca_mat
    #     XYZdCam -> R'G'B'     cam_xyz_matrix
    color_mat = np.matmul(ca_mat, cam_xyz_matrix.mat)
    color_mat = np.matmul(mat_srgb_d65, color_mat)

    # Normalize to remove tint (refer to dcraw for why)
    color_sum = color_mat.sum(axis=1)
    color_mat = color_mat / color_sum[:, np.newaxis]

    # Invert to produce R'G'B' (reference space) -> RGB (linear, D65)
    color_mat = np.linalg.inv(color_mat)

    # Transform camera RGB to linearized sRGB
    rgb_pre = np.dot(rgb, color_mat.T)
    return rgb_pre.astype(np.float32)

def lin_srgb_to_srgb(rgb : np.ndarray) -> np.ndarray:
    """Convert an input linearized sRGB image to a true sRGB image by applying the gamma curve.

    Args:
        rgb (np.ndarray): Linearized sRGB image.

    Returns:
        np.ndarray: sRGB image.
    """
    rgb_working = clip_rgb(rgb)
    return np.where(rgb_working <= 0.0031308, rgb_working * 12.92, (1.055 * (rgb_working ** (1 / 2.4))) - 0.055)

def srgb_to_lin_srgb(srgb : np.ndarray) -> np.ndarray:
    """Convert an sRGB image to a linearized sRGB image by removing the gamma curve.

    Args:
        srgb (np.ndarray): sRGB image.

    Returns:
        np.ndarray: Linearized sRGB image.
    """
    srgb_working = clip_rgb(srgb)
    return np.where(srgb_working <= 0.04045, srgb_working / 12.92, ((srgb_working + 0.055) / 1.055) ** 2.4)