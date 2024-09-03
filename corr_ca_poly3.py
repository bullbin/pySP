import numpy as np

# TODO - Test using full set, currently only tested with v (but is correct)

def get_remap_coords(chan : np.ndarray, poly3_b : float, poly3_c : float, poly3_v : float, max_iterations : int = 16, stop_epsilon : float = 0.00001) -> np.ndarray:
    """Compute remapping coordinates for a channel using the Lensfun Poly3 model. This can be used to effectively remove chromatic abberation after
    demosaicing.

    Model follows implementation as according to https://lensfun.github.io/calibration-tutorial/lens-distortion.html. The lens polynomial is solved
    using Newton's method. The output uses the same pixel centering and alignment as OpenCV so is suitable for usage with cv2.remap.

    Coefficients can be computed using hugin but a faster way to estimate them is to use darktable's TCA override. The provided coefficients are for
    v and can be converted as v = 1 + (1 - <darktable_coeff>) with b = 0, c = 0.

    Args:
        chan (np.ndarray): Input channel. Maximal radius and bounds will be computed using this shape.
        poly3_b (float): Poly3 B coefficient
        poly3_c (float): Poly3 C coefficient
        poly3_v (float): Poly3 V coefficient
        max_iterations (int, optional): Maximum iterations for polynomial solver. Higher increases potential accuracy. Defaults to 16.
        stop_epsilon (float, optional): Stop solver early if improvement is below this threshold. Lower improves precision. Defaults to 0.00001.

    Returns:
        np.ndarray: Array of (x,y) with each component of shape (height,width) containing destination coords for each incoming pixel.
    """
    
    def distort_poly3(radius : np.ndarray, b : float, c : float, v : float) -> np.ndarray:
        r_square = radius ** 2
        r_cubed = r_square * radius
        return b * r_cubed + c * r_square + v * radius

    def distort_poly3_sub(rad_distorted : np.ndarray, radius : np.ndarray, b : float, c : float, v : float) -> np.ndarray:
        return distort_poly3(radius, b, c, v) - rad_distorted

    def distort_prior_poly3(radius : np.ndarray, b : float, c : float, v : float) -> np.ndarray:
        r_square = radius ** 2
        return 3 * b * r_square + 2 * c * radius + v

    p_height, p_width = chan.shape[0] - 1, chan.shape[1] - 1
    c_x, c_y = p_width / 2, p_height / 2
    
    arr_x = np.zeros(shape=(chan.shape[0], chan.shape[1]), dtype=np.float32)
    arr_y = np.zeros(shape=(chan.shape[0], chan.shape[1]), dtype=np.float32)
    for x in range(chan.shape[1]):
        arr_x[:,x] = x - c_x
    for y in range(chan.shape[0]):
        arr_y[y,:] = y - c_y
    
    arr_rad_dist = np.sqrt(arr_x ** 2 + arr_y ** 2)
    arr_ang = np.arctan2(arr_y, arr_x)
    arr_rad_undist = np.zeros_like(arr_rad_dist)

    err = np.inf
    last_err = err

    i = 0
    while i < max_iterations:
        arr_rad_prior = np.copy(arr_rad_undist)
        arr_rad_undist = arr_rad_undist - (distort_poly3_sub(arr_rad_dist, arr_rad_undist, poly3_b, poly3_c, poly3_v) /
                                           distort_prior_poly3(arr_rad_undist, poly3_b, poly3_c, poly3_v))
        err = np.max(np.abs(arr_rad_prior - arr_rad_undist))
        
        if err < stop_epsilon or err == last_err:
            break
        last_err = err
        i += 1
    
    arr_new_x = arr_rad_undist * np.cos(arr_ang) + c_x
    arr_new_y = arr_rad_undist * np.sin(arr_ang) + c_y
    return np.dstack((arr_new_x, arr_new_y))