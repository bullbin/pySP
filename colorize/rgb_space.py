from typing import Optional, Tuple, Union

from colour import xy_to_XYZ
import numpy as np

from pySP.wb_cct.helpers_cam_mat import bradford_adapt_matrix
from pySP.wb_cct.standard_ill import StandardIlluminant, get_chromacity_from_illuminant

class ArbitraryRgbColorspace():
    def __init__(self, primary_xy_r : Tuple[float, float], primary_xy_g : Tuple[float, float], primary_xy_b : Tuple[float, float], whitepoint : StandardIlluminant):
        self.__primary_r = primary_xy_r
        self.__primary_g = primary_xy_g
        self.__primary_b = primary_xy_b
        self.__whitepoint = xy_to_XYZ(get_chromacity_from_illuminant(whitepoint))
    
    def mat_to_rgb(self, source_whitepoint : Optional[Union[Tuple[float,float,float], StandardIlluminant]] = None) -> np.ndarray:
        return np.linalg.inv(self.mat_to_xyz(source_whitepoint))

    def mat_to_xyz(self, destination_whitepoint : Optional[Union[Tuple[float,float,float], StandardIlluminant]] = None) -> np.ndarray:

        def get_coeff_0(primary : Tuple[float,float]):
            return primary[0] / primary[1]

        def get_coeff_1(primary : Tuple[float,float]):
            return (1 - primary[0] - primary[1]) / primary[1]
        
        matrix = np.array([[get_coeff_0(self.__primary_r), get_coeff_0(self.__primary_g), get_coeff_0(self.__primary_b)],
                           [                            1,                             1,                             1],
                           [get_coeff_1(self.__primary_r), get_coeff_1(self.__primary_g), get_coeff_1(self.__primary_b)]])
        
        s = np.linalg.inv(matrix) @ self.__whitepoint

        matrix[:,0] *= s[0]
        matrix[:,1] *= s[1]
        matrix[:,2] *= s[2]

        if destination_whitepoint != None:

            if type(destination_whitepoint) == StandardIlluminant:
                destination_white = xy_to_XYZ(get_chromacity_from_illuminant(destination_whitepoint))
            else:
                destination_white = np.array(destination_whitepoint)
            
            assert destination_white.shape[0] == 3
            assert len(destination_white.shape) == 1

            adapt_mat = bradford_adapt_matrix(self.__whitepoint, destination_white)

            # Feels a bit weird this is backwards but it matches with Lindbloom mats so...
            return adapt_mat @ matrix

        return matrix

class LinRgbColorspace():
    REC709 = ArbitraryRgbColorspace((0.64, 0.33), (0.3, 0.6), (0.15, 0.06), StandardIlluminant.D65)
    REC2020 = ArbitraryRgbColorspace((0.708, 0.292), (0.170, 0.797), (0.131, 0.046), StandardIlluminant.D65)