from enum import IntEnum, auto
from scipy.interpolate import BSpline
import numpy as np

from pySP.colorize.rgb_space import ArbitraryRgbColorspace
from pySP.colorize.transform import oklab_to_xyzd65, xyzd65_to_oklab
from pySP.wb_cct.standard_ill import StandardIlluminant

class METHOD_HUE_PRESERVATION(IntEnum):
    LINEAR_LAB = auto()
    DISABLED = auto()

def apply_tone_curve(channel : np.ndarray, curve : BSpline) -> np.ndarray:
    """Apply a base tone curve to a channel.

    Args:
        channel (np.ndarray): Channel. Must be shape (y,x).
        curve (BSpline): Toning curve. Should be defined over channel interval.

    Returns:
        np.ndarray: Toned channel.
    """
    assert len(channel.shape) == 2
    in_shape = channel.shape
    flat = channel.flatten()
    flat = curve(flat)
    return flat.reshape(in_shape)

def apply_tone_curve_rgb(linear_image : np.ndarray, curve : BSpline, colorspace : ArbitraryRgbColorspace, hue_preservation : METHOD_HUE_PRESERVATION) -> np.ndarray:
    """Apply a base tone curve to an image.

    This method supports hue preservation so toning will only effect the brightness of the color to prevent unnatural outputs.

    Args:
        linear_image (np.ndarray): Linear image. Should be in shape compatible with colorspace operations (y,x,3).
        curve (BSpline): Toning curve. Should be defined over channel interval.
        colorspace (ArbitraryRgbColorspace): Input colorspace. Used for transformation for certain hue preservation methods.
        hue_preservation (METHOD_HUE_PRESERVATION): Hue preservation mode.

    Raises:
        NotImplementedError: Raised if unimplemented hue preservation is requested.

    Returns:
        np.ndarray: Toned RGB image. Colorspace will match input as defined by colorspace parameter.
    """
    # TODO - Some inoptimal stuff happening with Bradford - our Oklab-XYZ implementation wants D65 input always which means we have to compute
    #        full Bradford adaptation prior to input. If we support premult linear matrices, we can skip double matrix op.
    # TODO - Remove and clip as needed to prevent invalid data hitting Oklab input. Currently this is mostly pure black causing issues but sanitization is good here.
    # TODO - Add something to perform hue preserving color conversion that clamps to closest in-gamut color. Desaturate until sane?

    if hue_preservation == METHOD_HUE_PRESERVATION.LINEAR_LAB:
        # Oklab works great for hue preservation, i.e., applying tone curve to lightness only to keep colors similar.
        # A cheaper alternative would be YUV but the desaturation was horrid
        # The issue is that Oklab uses D65 white so we need to be careful to adapt as needed back and forth
        #     so we get D65 during toning and source white on return.

        adaptation = StandardIlluminant.D65
        mat_to_xyz = colorspace.mat_to_xyz(adaptation)
        oklab = xyzd65_to_oklab(np.dot(linear_image, mat_to_xyz.T))

        l = oklab[:,:,0]

        # 'Linearize' l again (undo cube root)
        # Source https://discuss.pixls.us/t/oklab-cielab-linear-cielab-tonemapping/28767
        # Without this linearization there will be big shift in lightness
        l_linear = l ** 3

        # Apply curve, undo linearization
        l_linear = (apply_tone_curve(l_linear, curve) ** (1/3))

        xyz = oklab_to_xyzd65(np.dstack((l_linear, oklab[...,1], oklab[...,2])))

        # Return to original colorspace by adapting back to source white
        return np.dot(xyz, colorspace.mat_to_rgb(adaptation).T)
    
    elif hue_preservation == METHOD_HUE_PRESERVATION.DISABLED:
        count_channels = linear_image.shape[2]

        toned = [apply_tone_curve(linear_image[...,x], curve) for x in range(count_channels)]
        return np.dstack(toned)

    raise NotImplementedError("Hue preservation mode " + str(hue_preservation) + " not implemented!")