from __future__ import annotations
import numpy as np

def bradford_adapt_matrix(current_xyz : np.ndarray, target_xyz : np.ndarray) -> np.ndarray:
    mat_xyz_to_lms = np.array([[0.8951000, 0.2664000, -0.1614000],
                               [-0.7502000, 1.7135000, 0.0367000],
                               [0.0389000, -0.0685000, 1.0296000]])

    lms_curr = np.matmul(mat_xyz_to_lms, current_xyz)
    lms_targ = np.matmul(mat_xyz_to_lms, target_xyz)
    lms_scale = lms_targ / lms_curr

    mat_scale = np.array([[lms_scale[0], 0, 0],
                          [0, lms_scale[1], 0],
                          [0, 0, lms_scale[2]]])
    
    return np.matmul(np.linalg.inv(mat_xyz_to_lms), np.matmul(mat_scale, mat_xyz_to_lms))

class ChromacityMat():
    def __init__(self, mat : np.ndarray, xyz : np.ndarray):
        self.mat = np.copy(mat)
        self.mat.setflags(write=False)
        self.xyz = np.copy(xyz)
        self.xyz.setflags(write=False)
        
class MatXyzToCamera(ChromacityMat):
    def interpolate(self, next : MatXyzToCamera, blend : float) -> np.ndarray:
        blend = np.clip(blend, 0.0, 1.0)
        mat = self.mat * (1 - blend) + (next.mat * blend)
        return mat