import numpy as np

def cam_to_lin_srgb(rgb : np.ndarray, cam_xyz_matrix : np.ndarray, clip_highlights : bool = True) -> np.ndarray:
    """Convert an input image from camera-space to linearized sRGB colors.

    Args:
        rgb (np.ndarray): Debayered normalized camera-space RGB image.
        cam_xyz_matrix (np.ndarray): Camera color matrix; this is supplied by the image object.
        clip_highlights (bool, optional): Whether to clip highlights to reduce false highlight colors. Reduces color artifacts at cost of detail. Defaults to True.

    Returns:
        np.ndarray: Linearized sRGB image.
    """

    # TODO - Assert CMYK
    mat = cam_xyz_matrix[:3]
    
    # TODO - Adapted color spaces, http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    mat_srgb_d65 = np.array([[0.412453, 0.357580, 0.180423],
                             [0.212671, 0.715160, 0.072169],
                             [0.019334, 0.119193, 0.950227]])
    
    # TODO - Highlight recovery, allow for changing highlight peak to allow for subpixel HDR
    if clip_highlights:
        rgb[:,:,0] = np.clip(rgb[:,:,0], 0, 1)
        rgb[:,:,1] = np.clip(rgb[:,:,1], 0, 1)
        rgb[:,:,2] = np.clip(rgb[:,:,2], 0, 1)

    # Credit - dcraw, https://ninedegreesbelow.com/files/dcraw-c-code-annotated-code.html#E3
    # Credit - https://stackoverflow.com/questions/8904694/how-to-normalize-a-2-dimensional-numpy-array-in-python-less-verbose
    color_mat = np.matmul(mat, mat_srgb_d65)
    color_sum = color_mat.sum(axis=1)
    color_mat = color_mat / color_sum[:, np.newaxis]
    color_mat = np.linalg.inv(color_mat)

    rgb_pre = np.dot(rgb, color_mat.T)
    return rgb_pre.astype(np.float32)

def lin_srgb_to_srgb(rgb : np.ndarray) -> np.ndarray:
    """Convert an input linearized sRGB image to a true sRGB image by applying the gamma curve.

    Args:
        rgb (np.ndarray): Linearized sRGB image.

    Returns:
        np.ndarray: sRGB image.
    """
    rgb[:,:,0] = np.clip(rgb[:,:,0], 0, 1)
    rgb[:,:,1] = np.clip(rgb[:,:,1], 0, 1)
    rgb[:,:,2] = np.clip(rgb[:,:,2], 0, 1)
    return np.where(rgb <= 0.0031308, rgb * 12.92, (1.055 * (rgb ** (1 / 2.4))) - 0.055)