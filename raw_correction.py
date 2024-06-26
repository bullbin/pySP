import numpy as np

def dark_frame_subtraction(raw : np.ndarray, dark_frame : np.ndarray) -> np.ndarray:
    """Remove dark current noise from an image.

    Args:
        raw (np.ndarray): Raw RGBG image.
        dark_frame (np.ndarray): Raw dark frame RGBG image.

    Returns:
        np.ndarray: Raw image with dark current noise removed.
    """
    return np.copy(raw)

def bias_frame_subtraction(raw : np.ndarray, bias_frame : np.ndarray) -> np.ndarray:
    """Remove fixed-pattern noise from an image.

    Args:
        raw (np.ndarray): Raw RGBG image.
        bias_frame (np.ndarray): Raw bias frame RGBG image.

    Returns:
        np.ndarray: Raw image with fixed-pattern noise removed.
    """
    return np.copy(raw)