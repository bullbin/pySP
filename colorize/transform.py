import numpy as np

from pySP.colorize.rgb_space import ArbitraryRgbColorspace, LinRgbColorspace
from pySP.wb_cct.helpers_cam_mat import MatXyzToCamera

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

def cam_to_rgb_norm(rgb : np.ndarray, cam_xyz_matrix : MatXyzToCamera, destination_colorspace : ArbitraryRgbColorspace, clip_highlights : bool = True) -> np.ndarray:
    """Convert an input image from camera-space to another linear RGB space.

    This uses the camera XYZ matrix which can introduce tint. Tint is cancelled out by forcing camera white to meet output white which should be good enough
    if white balance was close.

    Args:
        rgb (np.ndarray): Debayered normalized camera-space RGB image.
        cam_xyz_matrix (MatXyzToCamera): Camera color matrix; this is supplied by the image object.
        destination_colorspace (ArbitraryRgbColorspace): Output colorspace. White will be adapted using camera matrix stored white.
        clip_highlights (bool, optional): Whether to clip highlights to reduce false highlight colors. Defaults to True.

    Returns:
        np.ndarray: Detinted linear RGB output.
    """

    if clip_highlights:
        rgb = clip_rgb(rgb)
    
    mat_rgb_to_xyz_d_cam = destination_colorspace.mat_to_xyz(cam_xyz_matrix.xyz.tolist())
    color_mat = np.matmul(mat_rgb_to_xyz_d_cam, cam_xyz_matrix.mat)

    # Normalize to remove tint
    # Cam -> XYZ is imperfect with WB so you end up with tint on RGB channels
    # To fix this, balance out the tint such that cam r=g=b -> rgb r'=g'=b'
    color_sum = color_mat.sum(axis=1)
    color_mat = color_mat / color_sum[:, np.newaxis]        # Normalize per-channel to keep R=G=B during transform

    color_mat = np.linalg.inv(color_mat)
    
    # Transform camera RGB to linearized sRGB
    rgb_pre = np.dot(rgb, color_mat.T)
    return rgb_pre.astype(np.float32)

def cam_to_clean_xyz(rgb : np.ndarray, cam_xyz_matrix : MatXyzToCamera, pcs_colorspace : ArbitraryRgbColorspace = LinRgbColorspace.REC2020, clip_highlights : bool = True) -> np.ndarray:
    """Convert an input image from camera-space to XYZ by passing through RGB and using detinting. This produces a purer XYZ spectrum.

    The XYZ reference white will be connected to the PCS colorspace so won't necessarily be D50. Use Bradford if you want to make reference white consistent.

    Args:
        rgb (np.ndarray): Debayered normalized camera-space RGB image.
        cam_xyz_matrix (MatXyzToCamera): Camera color matrix; this is supplied by the image object.
        pcs_colorspace (ArbitraryRgbColorspace, optional): Working colorspace for rebalancing color. Wide gamut recommended. Defaults to LinRgbColorspace.REC2020.
        clip_highlights (bool, optional): Whether to clip highlights to reduce false highlight colors. Defaults to True.

    Returns:
        np.ndarray: Cleaned XYZ image.
    """

    rgb_norm = cam_to_rgb_norm(rgb, cam_xyz_matrix, pcs_colorspace, clip_highlights)
    rgb_to_xyz = pcs_colorspace.mat_to_xyz()    # TODO - Can add any whitepoint here

    # Transform working RGB to XYZ
    return np.dot(rgb_norm, rgb_to_xyz.T).astype(np.float32)

def cam_to_lin_srgb(rgb : np.ndarray, cam_xyz_matrix : MatXyzToCamera, clip_highlights : bool = True) -> np.ndarray:
    """Convert an input image from camera-space to linearized sRGB colors.

    Args:
        rgb (np.ndarray): Debayered normalized camera-space RGB image.
        cam_xyz_matrix (np.ndarray): Camera color matrix; this is supplied by the image object.
        clip_highlights (bool, optional): Whether to clip highlights to reduce false highlight colors. Reduces color artifacts at cost of detail. Defaults to True.

    Returns:
        np.ndarray: Linearized sRGB image.
    """
    return cam_to_rgb_norm(rgb, cam_xyz_matrix, LinRgbColorspace.REC709, clip_highlights)

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