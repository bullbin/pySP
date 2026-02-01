import numpy as np

from pySP.corr_ca.roi.helper import quarter_res_pool

class PooledChannel():
    def __init__(self, channel : np.ndarray, tile_pow : int = 4):
        self._tile_width = 2 ** tile_pow
        self._extra_yx = np.array(channel.shape[:2]) % self._tile_width

        shape = np.copy(channel.shape)
        shape[:2] = shape[:2] - self._extra_yx

        # Crop to fit only tile grid
        pooled = channel[self._extra_yx[0] // 2:(shape[0] + self._extra_yx[0] // 2),
                        self._extra_yx[1] // 2:(shape[1] + self._extra_yx[1] // 2)]

        self.source_cropped = np.copy(pooled)

        # Pool down to tile resolution to find useful areas
        for _i in range(tile_pow):
            pooled = quarter_res_pool(pooled)

        self.source = channel
        self.pooled = pooled
    
    def get_tile_width(self) -> int:
        return self._tile_width

    def tile_offset_to_real_coords(self, point : np.ndarray) -> np.ndarray:
        return np.array(point) * self._tile_width + (self._extra_yx // 2)