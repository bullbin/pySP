import numpy as np

def cam_to_lin_srgb(rgb : np.ndarray, cam_xyz_matrix : np.ndarray, clip_highlights : bool = True) -> np.ndarray:
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
    rgb[:,:,0] = np.clip(rgb[:,:,0], 0, 1)
    rgb[:,:,1] = np.clip(rgb[:,:,1], 0, 1)
    rgb[:,:,2] = np.clip(rgb[:,:,2], 0, 1)
    return np.where(rgb <= 0.0031308, rgb * 12.92, (1.055 * (rgb ** (1 / 2.4))) - 0.055)