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

# Oklab operators taken from fezzypixels (i wrote it, its okay)
def lin_srgb_to_oklab(lin_srgb : np.ndarray) -> np.ndarray:
	"""Convert a linearized sRGB image to Oklab.

	Args:
		lin_srgb (np.ndarray): Linearized sRGB image.

	Returns:
		np.ndarray: Oklab image.
	"""
	# Credit - https://bottosson.github.io/posts/oklab/
	r,g,b = lin_srgb[...,0], lin_srgb[...,1], lin_srgb[...,2]

	l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
	m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
	s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

	l_prime = np.cbrt(l)
	m_prime = np.cbrt(m)
	s_prime = np.cbrt(s)

	ok_l = 0.2104542553 * l_prime + 0.7936177850 * m_prime - 0.0040720468 * s_prime
	ok_a = 1.9779984951 * l_prime - 2.4285922050 * m_prime + 0.4505937099 * s_prime
	ok_b = 0.0259040371 * l_prime + 0.7827717662 * m_prime - 0.8086757660 * s_prime
	return np.dstack((ok_l,ok_a,ok_b))

def oklab_to_lin_srgb(oklab : np.ndarray) -> np.ndarray:
	"""Convert an Oklab image to linear sRGB.

	Args:
		oklab (np.ndarray): Oklab image. No clamping is applied.

	Returns:
		np.ndarray: Linearized sRGB image.
	"""
	# Credit - https://bottosson.github.io/posts/oklab/
	ok_l,ok_a,ok_b = oklab[...,0], oklab[...,1], oklab[...,2]
    
	l_prime = ok_l + 0.3963377774 * ok_a + 0.2158037573 * ok_b
	m_prime = ok_l - 0.1055613458 * ok_a - 0.0638541728 * ok_b
	s_prime = ok_l - 0.0894841775 * ok_a - 1.2914855480 * ok_b

	l_prime = l_prime ** 3
	m_prime = m_prime ** 3
	s_prime = s_prime ** 3

	r =  4.0767416621 * l_prime - 3.3077115913 * m_prime + 0.2309699292 * s_prime
	g = -1.2684380046 * l_prime + 2.6097574011 * m_prime - 0.3413193965 * s_prime
	b = -0.0041960863 * l_prime - 0.7034186147 * m_prime + 1.7076147010 * s_prime
	return np.dstack((r,g,b))