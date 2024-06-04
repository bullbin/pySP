import numpy as np

def dark_frame_subtraction(raw : np.ndarray, dark_frame : np.ndarray) -> np.ndarray:
    return np.copy(raw)

def bias_frame_subtraction(raw : np.ndarray, dark_frame : np.ndarray) -> np.ndarray:
    return np.copy(raw)