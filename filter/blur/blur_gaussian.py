import numpy as np
import cv2

def get_gaussian_filter_window_size(sigma : float, cutoff : int = 3) -> int:
    """Get the width of the window required for a given Gaussian blur kernel.

    This method doesn't compute the kernel, just returns the size of the window needed to store the kernel while avoiding obvious cutoff at filter edges.

    Args:
        sigma (float): Reach of the bell part of the curve. Comparable to radius where higher sigma (standard deviation) means more blur reach.
        cutoff (int, optional): How many standard deviations to preserve. More deviations means more terms which worsens performance. 3 deviations captures basically the whole bell. Defaults to 3.

    Raises:
        ValueError: Raised if sigma is invalid, i.e., negative.

    Returns:
        int: Maximum window size for 1D gaussian that contains the amount of deviations needed for sigma.
    """
    if sigma < 0:
        raise ValueError("Filter cannot be computed with negative sigma!")
    
    # Typically past 3 sd the filter approaches 0
    # https://en.wikipedia.org/wiki/Gaussian_blur#Mechanics
    radius = sigma * cutoff
    diameter = np.ceil(radius * 2)

    if diameter % 2 == 0:
        diameter += 1

    return max(3, diameter)

def get_1d_gaussian_filter(sigma : float) -> np.ndarray:
    """Return a 1D Gaussian filter for a given sigma.

    The size of the kernel is computed according to sigma. It is always odd and is close to sigma * 6 so that the Gaussian bell is approximated completely.

    Args:
        sigma (float): Reach of the bell part of the curve. Comparable to radius where higher sigma (standard deviation) means more blur reach.

    Returns:
        np.ndarray: 1D Gaussian bell. Size is automatically computed.
    """

    try:
        radius = get_gaussian_filter_window_size(sigma) // 2
    except ValueError:
        return np.array([[1]])
    
    filter = np.arange(-radius, radius + 1, 1)

    denom = 1 / (np.sqrt(2 * np.pi) * sigma)
    filter = np.exp(-filter ** 2 / (2 * sigma ** 2))

    filter = denom * filter
    return filter

def blur_gaussian(image : np.ndarray, sigma : float) -> np.ndarray:
    """Perform a Gaussian blur.

    This uses a 2-pass approach to reduce computation. Edges are reflected which may produce artifacts.

    Args:
        image (np.ndarray): Base image. Must be of shape (a,b,<channels>) or (a,b).
        sigma (float): Reach of the bell part of the curve. Comparable to radius where higher sigma (standard deviation) means more blur reach.

    Returns:
        np.ndarray: Blurred image. Same shape as original.
    """

    # TODO - Cython this, a bit slow for larger images.

    # This is a simple Gaussian blur in 2-pass mode (takes advantage of separability)
    # This reduces overall computation:
    #     A separated filter has length N so needs N products to solve
    #     The full 2D filter is size NxN so needs N*N products to solve
    #     The full filter can be solved by doing 2 separated filters so 2N instead of N*2
    #         operations.

    filter = get_1d_gaussian_filter(sigma)
    border = filter.shape[0] // 2

    image_replicated = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_REFLECT)

    # Duplicate the shape, don't hardcode channels so we can support anything of shape (a,b,<channels>) or just (a,b)
    padded_shape = list(image.shape)
    padded_shape[0] += border + border

    h_pass = np.zeros(shape=padded_shape, dtype=np.float32)

    # Blur in X
    for idx_coeff, coeff in enumerate(filter):
        h_pass[:,:] += (image_replicated[:,idx_coeff:idx_coeff + image.shape[1]] * coeff)
    
    # Free padded image, start V blur
    del image_replicated
    v_pass = np.zeros_like(image, dtype=np.float32)

    # Blur in Y
    for idx_coeff, coeff in enumerate(filter):
            v_pass[:,:] += (h_pass[idx_coeff:idx_coeff + image.shape[0]] * coeff)

    return v_pass