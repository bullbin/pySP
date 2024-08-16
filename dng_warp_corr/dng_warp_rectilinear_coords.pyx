import numpy as np
cimport numpy as np

import cython
cimport cython

from cython.parallel import prange, parallel
from libc.math cimport sqrt
np.import_array()

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

# Credit - Transcription of provided algorithm from Adobe, DNG Specification 1.4.0.0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_table(DTYPE_t[:,:,:] table, float kr0, float kr1, float kr2, float kr3, float kt0, float kt1,
                        float m, float cam_center_x, float cam_center_y, unsigned int width, unsigned int height) noexcept nogil:
    cdef float dx, dy, r, f, dxr, dyr, dxt, dyt, xp, yp
    cdef unsigned int x, y

    with parallel():
        for x in prange(width):
            dx = (x - cam_center_x) / m

            for y in prange(height):    
                dy = (y - cam_center_y) / m
                r = sqrt((dx ** 2) + (dy ** 2))
                f = kr0 + (kr1 * r ** 2) + (kr2 * r ** 4) + (kr3 * r ** 6)
                
                dxr = f * dx
                dyr = f * dy
                dxt = kt0 * (2 * dx * dy) + kt1 * (r ** 2 + 2 * dx ** 2)
                dyt = kt1 * (2 * dx * dy) + kt0 * (r ** 2 + 2 * dy ** 2)

                xp = cam_center_x + m * (dxr + dxt)
                yp = cam_center_y + m * (dyr + dyt)
                table[y,x,0] = xp
                table[y,x,1] = yp

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void offset_table(DTYPE_t[:,:,:] seed, DTYPE_t[:,:,:] table, float kr0, float kr1, float kr2, float kr3, float kt0, float kt1,
                        float m, float cam_center_x, float cam_center_y, unsigned int width, unsigned int height) noexcept nogil:
    cdef float dx, dy, r, f, dxr, dyr, dxt, dyt, xp, yp
    cdef unsigned int x, y

    with parallel():
        for x in prange(width):
            for y in prange(height):
                dx = (seed[y,x,0] - cam_center_x) / m
                dy = (seed[y,x,1] - cam_center_y) / m
                r = sqrt((dx ** 2) + (dy ** 2))
                f = kr0 + (kr1 * r ** 2) + (kr2 * r ** 4) + (kr3 * r ** 6)
                
                dxr = f * dx
                dyr = f * dy
                dxt = kt0 * (2 * dx * dy) + kt1 * (r ** 2 + 2 * dx ** 2)
                dyt = kt1 * (2 * dx * dy) + kt0 * (r ** 2 + 2 * dy ** 2)

                xp = cam_center_x + m * (dxr + dxt)
                yp = cam_center_y + m * (dyr + dyt)
                table[y,x,0] = xp
                table[y,x,1] = yp

cpdef np.ndarray[DTYPE_t, ndim=3] compute_remapping_table(float kr0, float kr1, float kr2, float kr3, float kt0, float kt1,
                 unsigned int width, unsigned int height, float cam_center_norm_x, float cam_center_norm_y):
    
    cdef np.ndarray[DTYPE_t, ndim=3] table
    table = np.zeros([height, width, 2], dtype=DTYPE)

    cdef float cam_center_x = (width - 1) * cam_center_norm_x
    cdef float cam_center_y = (height - 1) * cam_center_norm_y
    cdef float max_dist_x = max(abs(-cam_center_x), abs(width - 1 - cam_center_x))
    cdef float max_dist_y = max(abs(-cam_center_y), abs(height - 1 - cam_center_y))
    cdef float m = np.sqrt(max_dist_x ** 2 + max_dist_y ** 2)

    compute_table(table, kr0, kr1, kr2, kr3, kt0, kt1, m, cam_center_x, cam_center_y, width, height)
    return table

cpdef np.ndarray[DTYPE_t, ndim=3] compute_offset_remapping_table(np.ndarray[DTYPE_t, ndim=3] seed, float kr0,
                 float kr1, float kr2, float kr3, float kt0, float kt1, unsigned int width, unsigned int height,
                 float cam_center_norm_x, float cam_center_norm_y):
    
    cdef np.ndarray[DTYPE_t, ndim=3] table
    table = np.zeros([height, width, 2], dtype=DTYPE)

    cdef float cam_center_x = (width - 1) * cam_center_norm_x
    cdef float cam_center_y = (height - 1) * cam_center_norm_y
    cdef float max_dist_x = max(abs(-cam_center_x), abs(width - 1 - cam_center_x))
    cdef float max_dist_y = max(abs(-cam_center_y), abs(height - 1 - cam_center_y))
    cdef float m = np.sqrt(max_dist_x ** 2 + max_dist_y ** 2)

    offset_table(seed, table, kr0, kr1, kr2, kr3, kt0, kt1, m, cam_center_x, cam_center_y, width, height)
    return table