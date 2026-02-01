from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np
from pipeline.border_control.linework.line import Line2DXeY, Line2DYeX
from pySP.corr_ca.roi.helper import remove_radial_content
from pySP.corr_ca.roi.tiled.tile_pooler import PooledChannel

def linear_regression_fit(data_x : np.ndarray, data_y : np.ndarray) -> Tuple[float, np.polynomial.Polynomial]:
    fit, data = np.polynomial.polynomial.Polynomial.fit(x=data_x, y=data_y, deg=1, full=True)
    if data[0].shape[0] == 0:
        return (np.inf, fit)
    return (data[0][0], fit)

@dataclass
class TileResult:
    offset_real_tl : np.ndarray
    average_n : float
    offset_average_n : np.ndarray

class RoiDetector():
    def __init__(self, pooled_resource : PooledChannel, remove_percent : float = 0.3, bins : int = 16, highest_n : int = 6, acceptable_error : float = 5, acceptable_edge_proximity : float = 0.8, acceptable_cos_angle : float = 0.5, default_threshold : int = 0):
        self._resource = pooled_resource
        remove_radial_content(self._resource.pooled, 0, remove_percent)

        self._max_bin_count = bins
        self._threshold = -1
        self._threshold_map = np.ones(self._resource.pooled.shape, dtype=np.bool)
        self._map_tile_idx = np.ones(self._resource.pooled.shape, dtype=np.int32) * -1

        self._detector_n_sample = highest_n
        self._detector_max_error = acceptable_error
        self._detector_edge_prox = acceptable_edge_proximity
        self._detector_max_angle = acceptable_cos_angle

        self.__central_point_idx = (np.array(self._resource.source.shape[:2]) - 1) / 2

        self._tiles : List[TileResult]          = []
        self._bins  : List[List[TileResult]]    = []

        # During init, compute fast radial lookup map
        # This isn't exact (its missing y,x crop) but good enough for our purposes
        y_shape, x_shape = tuple(self._threshold_map.shape[:2])
        if x_shape % 2 != 1 or y_shape % 2 != 1:
            raise ValueError("Incorrect shape for packing!")

        # Compute radius lookup map
        radius = np.zeros(((y_shape // 2) + 1, (x_shape // 2) + 1), dtype=np.float32)

        # Add squared axial deltas
        radius[:,] = np.arange((x_shape // 2) + 1)[::-1] ** 2
        radius += (np.arange((y_shape // 2) + 1)[::-1] ** 2)[:,np.newaxis]

        # Compute radius, normalize by diagonal to make radius 1 for image circle touching diagonal
        radius = np.sqrt(radius)
        radius = radius / (radius[0,0] + np.spacing(radius[0,0]))   # Add a slight spacing to force diagonals when flooring to be under divisons

        self._radial_lookup = (radius * self._max_bin_count).astype(np.uint16)  # With epsilon, range should be 0 - divisions-1

        # Mirror the array for other quadrants (just representing TL right now). Leave outer bound, that corresponds to center
        self._radial_lookup = np.concatenate((self._radial_lookup[:,:-1], self._radial_lookup[:,::-1]), axis=1)
        self._radial_lookup = np.concatenate((self._radial_lookup[:-1], self._radial_lookup[::-1]), axis=0)

        self.apply_threshold(default_threshold)

    def _update_bins(self):
        # Invalidate current bins
        self._bins = []

        # Copy out our pre-indexed array for radius
        lookup = np.copy(self._radial_lookup)

        # Invalidate anything below the threshold by setting it beyond our indexing range
        lookup[self._threshold_map == False] = self._max_bin_count

        # Extract co-ordinates of all points at each radial bin
        candidate_groups = [np.argwhere(lookup == x) for x in range(self._max_bin_count)]

        for group in candidate_groups:
            bin = []
            
            for point in group:
                bin.append(self._tiles[self._map_tile_idx[point[0], point[1]]])
            
            bin_sorted = sorted(bin, key=lambda result: result.average_n, reverse=True)
            self._bins.append(bin_sorted)
    
    def __extract_feature_map_from_tile(self, tile_index : np.ndarray):
        width = self._resource.get_tile_width()
        offset = self._resource.tile_offset_to_real_coords(tile_index).astype(np.uint32)
        tile = self._resource.source_cropped[offset[0]:offset[0] + width,
                                             offset[1]:offset[1] + width]

        # Get the location of the top n values in the tile
        flattened = tile.flatten()
        samples = np.argpartition(flattened, -self._detector_n_sample)[-self._detector_n_sample:]
        unflatten = np.unravel_index(samples, tile.shape)

        # Tested - skipping y fit doesn't lead to that much performance saving. Like 300ms. Probably
        #              faster if we just Cythonized everything
        y_err, y_fit = linear_regression_fit(unflatten[1], unflatten[0])
        x_err, x_fit = linear_regression_fit(unflatten[0], unflatten[1])

        is_y = y_err < x_err

        if is_y:
            fit = y_fit
            err = y_err
        else:
            fit = x_fit
            err = x_err

        if err > self._detector_max_error:
            return None

        # Avoid points too close to the edge, it will make matching hard since you'd need context
        #     from other tiles to get the best match
        # This is mostly okay for natural edges. For synthetic patterns (i.e., the test card used in
        #     kodachrom1e), this may lead to excessive rejections. These could be mitigated by applying
        #     a half tile offset and re-selecting ROI.

        # Compute midpoint
        midpoint = np.average(unflatten, axis=1)

        # Weight how close the midpoint is to the bounds of each axis
        ratio = np.abs(0.5 - (midpoint / tile.shape)) / 0.5

        # If the midpoint is close to the bounds, reject it because its
        #      probably an incomplete feature
        if ratio[0] >= self._detector_edge_prox or ratio[1] >= self._detector_edge_prox:
            return None
        
        params = fit.convert().coef         # Convert fit from np scale back to actual scale
        params = np.append(params, 0)       # Gradient not returned if m=0, workaround
        
        if is_y:
            line = Line2DYeX(params[1], params[0])
            
            # We need a direction vector later for cross product, compute for base = 0, base = 1
            point_a = (0, params[0])
            point_b = (1, params[0] + params[1])
        else:
            line = Line2DXeY(params[1], params[0])

            # We need a direction vector later for cross product, compute for base = 0, base = 1
            point_a = (params[0], 0)
            point_b = (params[0] + params[1], 1)

        # Convert to our line encoding, recompute midpoint as closest point on line.
        # Flip y,x to change from indexing to points
        midpoint = line.get_perpendicular_intersection(tuple(midpoint[::-1]))
        midpoint = (midpoint[0] + offset[0], midpoint[1] + offset[1])   # Convert to absolute co-ords

        # To compute angle, use dot product. Re-express line
        vec_center_to_mid = np.array((midpoint[0] - self.__central_point_idx[0], midpoint[1] - self.__central_point_idx[1]))
        vec_ab = np.array((point_b[0] - point_a[0], point_b[1] - point_a[1]))

        # Normalize so we don't have do it later
        vec_center_to_mid = vec_center_to_mid / np.sqrt(np.sum(vec_center_to_mid ** 2))
        vec_ab = vec_ab / np.sqrt(np.sum(vec_ab ** 2))

        dot = (vec_center_to_mid[0] * vec_ab[0]) + (vec_center_to_mid[1] * vec_ab[1])

        # If our line isn't that perpendicular from the radius, it will be hard to find a good scale
        #     factor because we will be sliding the feature along itself - all scales will seem good.
        # Reject matches where sliding along radius won't give a good result.
        if abs(dot) >= self._detector_max_angle:
            return None
        
        # At this point, feature should be strong with good directionality. The best features should have
        #     the strongest edge, so keep the average n values
        # During matching we will end up aligning this feature, so also keep the average feature location
        return TileResult(offset, np.average(tile[unflatten]), midpoint)

    def apply_threshold(self, threshold : float):
        if threshold == self._threshold:
            return
        
        self._threshold = threshold
        self._threshold_map = self._resource.pooled >= self._threshold
        
        candidate_groups = np.argwhere(self._threshold_map != False)
        for point in candidate_groups:

            # If tile is already cached, skip
            if self._map_tile_idx[point[0], point[1]] != -1:
                continue
            
            result = self.__extract_feature_map_from_tile(point)

            # If tile has no good features, prevent it from being used in future
            #     computation (invalidate its pooled source and threshold map)
            if result == None:
                # Invalidate this space from being used in future thresholds
                # Tile feature extraction is not dependent on threshold, this is okay
                self._resource.pooled[point[0], point[1]] = -1
                self._threshold_map[point[0], point[1]] = False
                continue

            self._map_tile_idx[point[0], point[1]] = len(self._tiles)
            self._tiles.append(result)

        self._update_bins()
