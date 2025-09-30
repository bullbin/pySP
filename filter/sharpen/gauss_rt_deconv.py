import numpy as np

from pySP.colorize import lin_srgb_to_oklab, oklab_to_lin_srgb
from pySP.filter.blur import blur_gaussian

def gaussian_rt_deconvolution(image : np.ndarray, sigma : float, iterations : int = 20) -> np.ndarray:
    """Richardson-Lucy deconvolution operating on a per-channel basis. This is a reconstruction method
    so is effective at boosting general sharpness and detail.

    This is a semi-blind operation and assumes the convolution was Gaussian-like. Adjust sigma until
    real details become clearer. If the image continually degrades it is likely Gaussian kernels do
    not fit well. Use a different sharpening method instead.

    Args:
        image (np.ndarray): Image, any range, must be of shape (a,b,<channels>) or (a,b).
        sigma (float): Radius for blur kernel.
        iterations (int, optional): Iterations. Lower values are faster and less artifact prone but reduce strength. Defaults to 20.

    Returns:
        np.ndarray: Unclipped sharpened image. Values may exceed ranges of original.
    """

    # Credits
    # Theory - https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    # Breakdown - https://stargazerslounge.com/topic/228147-lucy-richardson-deconvolution-so-what-is-it/

    # General case is simplified by Gaussian kernel being symmetrical - we can avoid inverse PSF

    estimate = np.copy(image)

    for i in range(iterations):
        blurred_estimate = blur_gaussian(estimate, sigma)
        estimate_factor = blur_gaussian(image / (blurred_estimate + 1e-25), sigma)  # Very small float to avoid divide by zero

        estimate = estimate * estimate_factor
    
    return estimate

def gaussian_rt_deconvolution_lab(lin_srgb : np.ndarray, radius : float, iterations : int = 20) -> np.ndarray:
    """Richardson-Lucy deconvolution operating in a LAB space. This is a reconstruction method
    so is effective at boosting general sharpness and detail.

    This is a semi-blind operation and assumes the convolution was Gaussian-like. Adjust sigma until
    real details become clearer. If the image continually degrades it is likely Gaussian kernels do
    not fit well. Use a different sharpening method instead.

    This applies deconvolution to just the L channel, avoiding color artifacts. Keep the amount sensible to
    prevent crunchiness between the color and lightness information.

    Args:
        lin_srgb (np.ndarray): Linearized sRGB image, must be of shape (a,b,3) with order RGB. Should be in [0,1] per channel for transforms to be valid.
        sigma (float): Radius for blur kernel.
        iterations (int, optional): Iterations. Lower values are faster and less artifact prone but reduce strength. Defaults to 20.

    Returns:
        np.ndarray: Unclipped sharpened image. Values may exceed ranges of original.
    """

    lab = lin_srgb_to_oklab(lin_srgb)

    lab[:,:,0] = gaussian_rt_deconvolution(lab[:,:,0], radius, iterations)
    
    return oklab_to_lin_srgb(lab)

def gaussian_rt_deconvolution_yuv(lin_srgb : np.ndarray, radius : float, iterations : int = 20) -> np.ndarray:
    """Richardson-Lucy deconvolution operating in YUV space. This is a reconstruction method
    so is effective at boosting general sharpness and detail.

    This is a semi-blind operation and assumes the convolution was Gaussian-like. Adjust sigma until
    real details become clearer. If the image continually degrades it is likely Gaussian kernels do
    not fit well. Use a different sharpening method instead.

    This applies only to the Y channel in a YUV color space in linear space. Because YUV isn't
    perceptual, it preserves linearity which makes it better for sensor-level or HDR transforms.
    Although this is more 'correct' for those applications, LAB might still result in a more natural
    result. Keep the amount sensible to prevent crunchiness between the color and lightness
    information.

    Args:
        lin_srgb (np.ndarray): Linearized sRGB image, must be of shape (a,b,3) with order RGB. Should be in [0,1] per channel for transforms to be valid.
        sigma (float): Radius for blur kernel.
        iterations (int, optional): Iterations. Lower values are faster and less artifact prone but reduce strength. Defaults to 20.

    Returns:
        np.ndarray: Unclipped sharpened image. Values may exceed ranges of original.
    """

    y = 0.299 * lin_srgb[:,:,0] + 0.587 * lin_srgb[:,:,1] + 0.114 * lin_srgb[:,:,2]
    
    y_modified = gaussian_rt_deconvolution(y, radius, iterations)

    # In theory, perceptual UV could modify the scale factor per channel. This is acceptable for now
    scale_factor = y_modified / y

    rgb_shifted = np.zeros_like(lin_srgb)
    rgb_shifted[:,:,0] = lin_srgb[:,:,0] * scale_factor
    rgb_shifted[:,:,1] = lin_srgb[:,:,1] * scale_factor
    rgb_shifted[:,:,2] = lin_srgb[:,:,2] * scale_factor
    return rgb_shifted

