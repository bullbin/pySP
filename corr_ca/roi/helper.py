from typing import Tuple
import cv2
import numpy as np

def quarter_res_pool(image : np.ndarray) -> np.ndarray:
    """Pool (additive) the input image to quarter res.

    If the input shape is not even y,x, the result will be cropped. For example, giving this image a 5x3 image will result
    in a 2x1 output where the odd pixels from the input were removed.

    Args:
        image (np.ndarray): Input image, minimum shape (y,x).

    Returns:
        np.ndarray: Pooled version of the input image.
    """

    max_x = image.shape[1] // 2
    max_y = image.shape[0] // 2
    return (image[::2,::2][:max_y,:max_x] + image[1::2,::2][:max_y,:max_x] + image[::2,1::2][:max_y,:max_x] + image[1::2,1::2][:max_y,:max_x])

def remove_radial_content(channel : np.ndarray, fill_val : float = 0, radial_percent : float = 0.3):
    """Remove (fill) radial content from a channel, center outwards.

    <b>This modifies <u>in-place</b></u>, be careful.

    Args:
        channel (np.ndarray): Channel. Must be shape (y,x).
        fill_val (float, optional): Infill value. Should be compatible with np datatype. Defaults to 0.
        radial_percent (float, optional): Percentage towards diagonal, such that 1.0 fills the whole channel. Defaults to 0.3.
    """

    center_x, center_y = channel.shape[1] // 2, channel.shape[0] // 2
    max_radius = np.sqrt((center_x ** 2 + center_y ** 2))
    cv2.circle(channel, (center_x, center_y), int(round(max_radius * radial_percent)), fill_val, thickness=-1)

def bilinear_sample(image : np.ndarray, offset : Tuple[float,float], width : int, height : int) -> np.ndarray:
    """Sample a section from an input using bilinear interpolation.

    Args:
        image (np.ndarray): Source image, shape (y,x) or (y,x,c)
        offset (Tuple[int,int]): Offset (y,x) for corner of 2D sample. Samples begin at the center of the pixel so where the offset is whole, no interpolation is used.
        width (int): Width of output.
        height (int): Height of output.

    Returns:
        np.ndarray: Sample, same amount of dimensions as source.
    """
    
    offset_y, offset_x = offset
    img_h, img_w = image.shape[:2]
    
    # Create coordinate grids for the output
    y_coords = np.arange(height, dtype=np.float32) + offset_y
    x_coords = np.arange(width, dtype=np.float32) + offset_x
    
    # Create meshgrid
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    
    # Get integer parts (floor)
    x0 = np.floor(x_grid).astype(np.int32)
    y0 = np.floor(y_grid).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Get fractional parts
    fx = x_grid - x0
    fy = y_grid - y0
    
    # Clip coordinates to valid range
    x0_clip = np.clip(x0, 0, img_w - 1)
    x1_clip = np.clip(x1, 0, img_w - 1)
    y0_clip = np.clip(y0, 0, img_h - 1)
    y1_clip = np.clip(y1, 0, img_h - 1)
    
    # Sample the four nearest pixels
    I00 = image[y0_clip, x0_clip]
    I01 = image[y0_clip, x1_clip]
    I10 = image[y1_clip, x0_clip]
    I11 = image[y1_clip, x1_clip]
    
    # Bilinear interpolation
    # For multi-channel images, we need to expand fx and fy
    if image.ndim == 3:
        fx = fx[..., np.newaxis]
        fy = fy[..., np.newaxis]
    
    w00 = (1 - fx) * (1 - fy)
    w01 = fx * (1 - fy)
    w10 = (1 - fx) * fy
    w11 = fx * fy
    
    output = w00 * I00 + w01 * I01 + w10 * I10 + w11 * I11
    
    return output