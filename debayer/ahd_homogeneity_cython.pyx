import numpy as np
cimport numpy as np

import cython
cimport cython

from cpython cimport bool
from cython.parallel import prange, parallel
from libcpp cimport bool as bool_c
np.import_array()

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float get_diff_square(float a_x, float a_y, float b_x, float b_y) noexcept nogil:
    return (a_x - b_x) ** 2 + (a_y - b_y) ** 2

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_map(DTYPE_t[:,:,:] lab, DTYPE_t[:,:] output, int r_x, int r_y, int k_pad, bint is_vertical) noexcept nogil:
    
    cdef int s_y, s_x, x, y, w_x, w_y
    cdef float ref_l, ref_a, ref_b, epsi_l, epsi_c_2

    cdef int domain_k = k_pad * 2 + 1
    
    with parallel():
        for y in prange(r_y):
            s_y = y + k_pad

            for x in range(r_x):
                s_x = x + k_pad
            
                # Get target pixel in LAB space
                ref_l = lab[s_y,s_x,0]
                ref_a = lab[s_y,s_x,1]
                ref_b = lab[s_y,s_x,2]

                # Compute bounds by assuming edge and looking at relevant neighbour pixels
                if is_vertical:
                    epsi_l = max(abs(ref_l - lab[s_y - 1,s_x,0]),
                                    abs(ref_l - lab[s_y + 1,s_x,0]))
                    epsi_c_2 = max(get_diff_square(ref_a, ref_b, lab[s_y - 1,s_x][1], lab[s_y - 1,s_x][2]),
                                    get_diff_square(ref_a, ref_b, lab[s_y + 1,s_x][1], lab[s_y + 1,s_x][2]))
                else:
                    epsi_l = max(abs(ref_l - lab[s_y,s_x - 1,0]),
                                    abs(ref_l - lab[s_y,s_x + 1,0]))
                    epsi_c_2 = max(get_diff_square(ref_a, ref_b, lab[s_y,s_x - 1][1], lab[s_y,s_x - 1][2]),
                                    get_diff_square(ref_a, ref_b, lab[s_y,s_x + 1][1], lab[s_y,s_x + 1][2]))

                # Consider pixels in window homogenenous if within bounds of brightness and color
                for w_y in range(y, y + domain_k):
                    for w_x in range(x, x + domain_k):
                        if lab[w_y, w_x, 0] - ref_l <= epsi_l:
                            if ((lab[w_y, w_x, 1] - ref_a) ** 2 + (lab[w_y, w_x, 2] - ref_b) ** 2) <= epsi_c_2:
                                output[y,x] = output[y,x] + 1


cpdef np.ndarray[DTYPE_t, ndim=2] build_map(np.ndarray[DTYPE_t, ndim=3] lab, unsigned int k_pad, unsigned int domain_k, bool is_vertical):
    
    cdef int r_x = lab.shape[1] - k_pad - k_pad
    cdef int r_y = lab.shape[0] - k_pad - k_pad
    cdef np.ndarray[DTYPE_t, ndim=2] homogeneity
    homogeneity = np.zeros([r_y, r_x], dtype=DTYPE)

    compute_map(lab, homogeneity, r_x, r_y, k_pad, is_vertical)
    return homogeneity