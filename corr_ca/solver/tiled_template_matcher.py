import numpy as np
from pySP.corr_ca.roi.helper import bilinear_sample

def template_match(target : np.ndarray, tile_blurred : np.ndarray, start : np.ndarray, end : np.ndarray, integer_only : bool = False, resample : bool = True, resample_max_steps : int = 8) -> np.ndarray:
    """Find the optimal position of a tile along an axis in the reference image which minimises the numerical difference.

    This method includes two solving methods, both of which operate along single-pixel steps:
    - Integer only, which uses approximate locations while sampling the reference image. This is fast but imprecise.
    - Bilinear, which resamples the input along estimates using bilinear sampling. This improves quality at some speed cost.

    While using bilinear sampling, additional resampling is permitted which lets the solver converge in sub-pixel steps around
    the initial solving location. This may be useful if optimal precision is required.

    For best performance on subpixel scale, blur the input tile image.

    Args:
        target (np.ndarray): Reference image.
        tile_blurred (np.ndarray): Tile image. A slight (<1px) blur is recommended for best quality with some noise reduction.
        start (np.ndarray): Start point of the sliding axis. The top-left corner of the tile (through the corner pixel center) is slid along this axis.
        end (np.ndarray): End point of the sliding axis.
        integer_only (bool, optional): Round sampling co-ordinates so indexing can be used for fast reference lookups. Defaults to False.
        resample (bool, optional): Use bilinear sampling for exact lookups of the reference image. Defaults to True. This will be disabled if integer-only sampling is enabled.
        resample_max_steps (int, optional): Maximum steps of the resampling algorithm. More steps increase cost. Steps converge quickly and early-stopping occurs when change gets small. Defaults to 8, which is optimal to ~4 decimal places.

    Returns:
        np.ndarray: Optimal sampling location. Sits on defined axis.
    """

    # Course match with slight pixel shift
    # This is for the most part fine, both images are blurry

    # TODO - Can remove gamma weighting, optimal sum should be unchanged
    if integer_only:
        def compute_err(offset : np.ndarray) -> float:
            offset = np.floor(offset).astype(np.int32)
            section = target[offset[0]:offset[0] + tile_blurred.shape[0],
                                    offset[1]:offset[1] + tile_blurred.shape[1]]
            return np.sum(np.abs(section - tile_blurred) ** 1/2.2)
    else:
        def compute_err_fractional(offset : np.ndarray) -> float:
            section = bilinear_sample(target, offset, tile_blurred.shape[1], tile_blurred.shape[0])
            return np.sum(np.abs(section - tile_blurred) ** 1/2.2)
        
        def compute_err(offset : np.ndarray) -> float:
            return compute_err_fractional(offset)

    delta = end - start
    mag = np.sqrt(np.sum(delta ** 2))
    
    vec = delta / mag

    size_step = 4

    vec = vec / size_step

    partial_steps_remaining = int(np.floor(mag * size_step))
    sample_location_indexing = np.copy(start)
    best_err = np.inf
    best_step = None
    for step in range(0, partial_steps_remaining):
        err = compute_err(sample_location_indexing)
        if err < best_err:
            best_err = err
            best_step = step

        sample_location_indexing += vec

    # TODO - Compute on end too
    # Do not attempt narrow band solve if using integer, it'll all be rounded away
    if not(resample) or integer_only:
        return start + (best_step * vec)
    
    # subpixel refine strategy
    # end cap (err) -> center -> end cap (err)
    # subdivide both caps and update err until delta between caps is low, return output!
    solver_start = start + ((best_step - 1) * vec)
    solver_end = start + ((best_step + 1) * vec)
    solver_center = (solver_start + solver_end) / 2

    last_center = np.copy(solver_end)

    for step in range(resample_max_steps):
        err_start = compute_err_fractional(solver_start)
        err_middle = compute_err_fractional(solver_center)
        err_end = compute_err_fractional(solver_end)

        if np.abs(err_middle - err_start) > np.abs(err_middle - err_end):
            solver_start = solver_center
        else:
            solver_end = solver_center

        solver_center = (solver_start + solver_end) / 2

        if np.all(solver_center == last_center):
            break

        last_center = solver_center
    
    return solver_center