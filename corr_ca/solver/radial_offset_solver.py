from typing import List, Tuple

import cv2
import numpy as np

from pySP.corr_ca.roi.tiled.tile_pooler import PooledChannel
from pySP.corr_ca.roi.tiled.tile_roi_finder import RoiDetector, TileResult
from pySP.corr_ca.solver.tiled_template_matcher import template_match

def get_start_end_points_from_centers(center_feature : np.ndarray, offset_actual_feature : np.ndarray, center_image : np.ndarray, radius_percent : float) -> Tuple[np.ndarray, np.ndarray]:
    delta = center_feature + offset_actual_feature - center_image
    return (center_image + (delta * (1 + radius_percent)) - offset_actual_feature, center_image + (delta * (1 - radius_percent)) - offset_actual_feature)

def get_radius_scale_factors_from_bins(detector : RoiDetector, pool : PooledChannel, reference_channel : np.ndarray, top_n : int = 16, max_reach : float = 0.004):

    if pool.source.shape != reference_channel.shape:
        raise ValueError("Reference and pooled channel shapes are not identical. No mapping can be formed.")

    tiles : List[TileResult] = []

    for bin in detector.bins:
        amount_to_select = min(top_n, len(bin))

        if amount_to_select == 0:
            continue

        tiles.extend(bin[:amount_to_select])
    
    if len(tiles) <= 4:
        raise ValueError("Not enough tiles to compute max quality model (PTLens).")
    
    radius_undistorted = []
    radius_distorted = []

    idx_center = (np.array(pool.source.shape[:2]) - 1) / 2
    max_r = np.sqrt(np.sum(idx_center ** 2))

    source_blurred = cv2.GaussianBlur(pool.source, (3,3), 0.33)

    for tile in tiles:
        tile_graphic = source_blurred[tile.offset_real_tl[0]:tile.offset_real_tl[0] + pool.get_tile_width(),
                                      tile.offset_real_tl[1]:tile.offset_real_tl[1] + pool.get_tile_width()]
        start, end = get_start_end_points_from_centers(tile.offset_real_tl, tile.offset_average_n, idx_center, max_reach)

        corrected = template_match(reference_channel,
                                   tile_graphic,
                                   start, end)
        
        tile_coords_feature             = tile.offset_real_tl + tile.offset_average_n
        tiles_coords_feature_corrected  = corrected + tile.offset_average_n

        delta_distorted     = tile_coords_feature - idx_center
        delta_undistorted   = tiles_coords_feature_corrected - idx_center

        r_d = np.sqrt(np.sum(delta_distorted ** 2))
        r_ud = np.sqrt(np.sum(delta_undistorted ** 2))

        radius_undistorted.append(r_ud / max_r)
        radius_distorted.append(r_d / max_r)

    return np.dstack((radius_distorted, radius_undistorted))[0]

def get_scale_pairs_using_pooled_tiler(channel_distorted : np.ndarray, channel_undistorted : np.ndarray, threshold : int = 16, max_reach : float = 0.004):
    # TODO - Expose tile power, threshold should be auto to 2 ** tile_pow
    pool = PooledChannel(channel_distorted)
    detector = RoiDetector(pool, default_threshold = threshold)
    return get_radius_scale_factors_from_bins(detector, pool, channel_undistorted, max_reach=max_reach)