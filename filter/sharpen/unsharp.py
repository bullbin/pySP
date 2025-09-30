import numpy as np

from pySP.colorize import lin_srgb_to_oklab, oklab_to_lin_srgb
from pySP.filter.blur import blur_gaussian

def unsharp_mask_per_channel(image : np.ndarray, radius : float, amount : float) -> np.ndarray:
    """Unsharp mask operating on a per-channel basis. This is a microcontrast boosting method so is
    effective improving edge sharpness and overall clarity.

    This uses naive unsharp on every channel. On RGB images, for example, expect overshoot and fringing.

    Args:
        image (np.ndarray): Image, any range, must be of shape (a,b,<channels>) or (a,b).
        radius (float): Radius for high pass blur kernel.
        amount (float): Amount of sharpening. Sensible values are in [0,1] but there is no restriction.

    Returns:
        np.ndarray: Unclipped sharpened image. Values may exceed ranges of original.
    """

    high_pass = image - blur_gaussian(image, radius)
    return image + high_pass * amount

def unsharp_mask_lab(lin_srgb : np.ndarray, radius : float, amount : float) -> np.ndarray:
    """Unsharp mask operating in a LAB space. This is a microcontrast boosting method so is
    effective improving edge sharpness and overall clarity.

    This applies unsharp to just the L channel, avoiding color artifacts. Keep the amount sensible to
    prevent crunchiness between the color and lightness information.

    Args:
        lin_srgb (np.ndarray): Linearized sRGB image, must be of shape (a,b,3) with order RGB. Should be in [0,1] per channel for transforms to be valid.
        radius (float): Radius for high pass blur kernel.
        amount (float): Amount of sharpening. Sensible values are in [0,1] but there is no restriction.

    Returns:
        np.ndarray: Unclipped sharpened image in linearized sRGB space. Values may exceed ranges of original.
    """

    lab = lin_srgb_to_oklab(lin_srgb)

    lab[:,:,0] = unsharp_mask_per_channel(lab[:,:,0], radius, amount)
    
    return oklab_to_lin_srgb(lab)