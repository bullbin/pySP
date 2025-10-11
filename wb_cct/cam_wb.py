from __future__ import annotations

import colour
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from pySP.wb_cct.helpers_cam_mat import MatXyzToCamera
from pySP.wb_cct.helpers_exif import exif_get_as_shot_neutral, exif_get_color_mat_sources
from pySP.wb_cct.standard_ill import StandardIlluminantSeries

###############################################################################
#       NOTE - White balancing (at least how I've learnt it works)
###############################################################################
# Whites are typically simplified as chromacities that sit close to the
# Planckian locus. You don't need to fully understand what that is, just that
# typical reference whites usually sit close to it. I don't get it either!
#
#    Cameras use a variety of techniques to figure out what reference value
# corresponds to white. In this case, we use AsShotNeutral, but this can also
# be chromacity - the matrices provided (mostly) convert between the two. DNGs
# provide up to 3 matrices which do this, each calibrated to get optimal color
# under different illuminants. The shift from using the wrong one isn't too
# substantial. XYZ is absolute after all so really you are only doing this to
# minimize tint under the estimated illuminant.
#
#    To convert AsShotNeutral to a temperature and get the optimal color
# matrix, you have to solve for the blending factor between the XYZtoCam
# matrices that achieves the minimum tint (dUV between output XYZ and closest
# locus XYZ). Note that the Planckian locus is shifted a bit while solving -
# temperature alone doesn't map to chromacity. D-series illuminants which
# are most similar to daylight have a slight tint from the locus which needs
# to be considered since these tend to look more natural. The output from
# this is the neutral chromacity and optimal matrix, both of which are
# needed for adaptation to a fixed reference white later.
#
#     Converting temp to camera white balance multipliers is easier.
# Temp (D-series adjusted) can be transformed to XYZ illuminant, then passed
# into the camera matrix to get a corresponding AsShotNeutral. The D-series
# adjustment is required to get values which match other software.
###############################################################################

def get_ideal_duv(temperature : float) -> float:
    """Get a desirable dUV for a given CCT based on the Planckian locus with adjustments after 4000K to D-series illuminants.

    This has a discontinuity at 4000K as D-series ends at D40. Below that is a sharp fall to 0.

    Args:
        temperature (float): Target color temperature.

    Returns:
        float: Desirable dUV.
    """
    if temperature < 4000:  # D-series illuminants are undefined under 4000K. Switch to blackbody under this
        return 0    # TODO - Doesn't match other software, still slight tint. This leads to a discontinuity,
                    #        searching git issues will find reports about this in other software too
    return colour.temperature.uv_to_CCT_Ohno2013(colour.xy_to_UCS_uv(colour.temperature.CCT_to_xy_CIE_D(temperature)))[1]

class CameraWhiteBalanceController():
    def __init__(self, mats : List[MatXyzToCamera], initial_ref_white : np.ndarray):
        """Create a white balance controller for a camera profile.

        Args:
            mats (List[MatXyzToCamera]): Camera XYZ calibration profiles. Must have at least 1.
            initial_ref_white (np.ndarray): Initial reference white for pre-optimization.
        """

        assert len(mats) > 1

        self.__mats = mats
        self.__optimal_multipliers = np.copy(initial_ref_white)
        self.__optimal_mat : MatXyzToCamera = None

        self.update_by_reference(initial_ref_white)

    def __set_optimal_mat_and_multipliers(self, mat : np.ndarray, xyz : np.ndarray):
        self.__optimal_mat = MatXyzToCamera(mat, xyz)
        self.__optimal_multipliers = np.matmul(self.__optimal_mat.mat, xyz)
        self.__optimal_multipliers = self.__optimal_multipliers / self.__optimal_multipliers[1]
        print(self.__optimal_multipliers)

    def update_by_temperature(self, cct : float, duv : Optional[int] = None, allow_cross_blend : bool = False):
        """Interpolate ColorMatrix calibrations to update the optimal matrix and neutral point assuming a new scene illuminant.

        Args:
            cct (float): Target color temperature.
            duv (Optional[int], optional): Target delta from Planckian locus. Defaults to an idealized curve which leans close to D-series above 4000K and blackbody below. Defaults to None.
            allow_cross_blend (bool, optional): Allow blending between matrices between different calibration series. May produce strange colors. Defaults to False.
        """

        if len(self.__mats) == 0:
            raise ValueError("No calibration matrices provided! Cannot interpolate matrix.")
        
        if len(self.__mats) == 1:
            # If there is only one calibration matrix, use that
            self.__set_optimal_mat_and_multipliers(self.__mats[0].mat, targ_xyz)
            return
        
        mat_k = [colour.temperature.XYZ_to_CCT_Ohno2013(mat.xyz)[0] for mat in self.__mats]
        mats_by_k = [x for _, x in sorted(zip(mat_k, self.__mats))]                             # Sort mats by temperature
        mat_k.sort()

        if duv == None:
            # Tint is usually computed for a given temperature. This is because we typically imagine temperature as relating
            #     to D-series illuminants which simulate daylight.
            # D series is tinted away from the Planckian locus, so compute a tint for the D-series illuminant.
            # If no illuminant is available, match to Planckian instead.

            duv = get_ideal_duv(cct)

        targ_xyz = colour.temperature.CCT_to_XYZ_Ohno2013(np.array([cct,duv]))

        if cct <= mat_k[0] or cct >= mat_k[-1]:         # If CCT is outside calibration range, return edge calibration matrices (may be tinted)
            if cct <= mat_k[0]:
                self.__set_optimal_mat_and_multipliers(mats_by_k[0].mat, targ_xyz)
            else:
                self.__set_optimal_mat_and_multipliers(mats_by_k[-1].mat, targ_xyz)
            return
        
        # Find closest calibration reference
        ref_list_k = mat_k
        ref_list_mats = mats_by_k

        if not(allow_cross_blend):
            # White balancing is typically under D-series but we sort matrices by k, meaning it is possible
            #     that two matrices from different series are picked for blending to meet a target k
            # This is likely to produce a strange result (although the different in my gear seems insignificant)
            # Prevent this by picking out only D-series for blending attempt

            # TODO - Swap to D-series or other illuminant, this will fail out and release only D-series if K is too low

            ref_list_k = []
            ref_list_mats = []

            for k, mat in zip(mat_k, mats_by_k):
                if mat.series == StandardIlluminantSeries.SERIES_DAYLIGHT:
                    ref_list_k.append(k)
                    ref_list_mats.append(mat)
            
            if len(ref_list_mats) == 0:
                raise ValueError("Could not find any daylight series matrices inside DNG!")

            if len(ref_list_mats) == 1:
                # If there is only one calibration matrix, use that
                self.__set_optimal_mat_and_multipliers(ref_list_mats[0].mat, targ_xyz)
                return
            
        ref_list_k.append(cct)
        ref_list_k.sort()

        idx_0 = ref_list_k.index(cct) - 1
        idx_1 = idx_0 + 1

        ref_list_k.pop(idx_1)

        mat_0 = ref_list_mats[idx_0]
        mat_1 = ref_list_mats[idx_1]

        mired_0 = colour.temperature.CCT_to_mired(mat_k[idx_0])
        mired_1 = colour.temperature.CCT_to_mired(mat_k[idx_1])
        mired_target = colour.temperature.CCT_to_mired(cct)

        override_blend = (mired_1 - mired_target) / (mired_1 - mired_0)
        blended_mat = mat_0.interpolate(mat_1, 1 - override_blend)

        self.__set_optimal_mat_and_multipliers(blended_mat, targ_xyz)

    def update_by_reference(self, ref_white : np.ndarray, max_iters : int = 30, stop_epsilon : float = 0.000001):
        """Interpolate ColorMatrix calibrations to update the optimal matrix under a provided camera neutral point.

        This algorithm is iterative but for most values will return one of the preset calibrations. It works by finding the matrix that minimizes
        the tint from an ideal curve assuming neutral is an approximate illuminant. Because daylight illuminants are typically preferred and have
        a different tinting profile than other illuminants (like A), usually the returned matrix is close to (or exactly) one of the D-series
        transformation presets.

        Args:
            ref_white (np.ndarray): Camera neutral point in reference (sensor) space [0,1]. Do not normalize to G' = 1.0.
            max_iters (int, optional): Maximum iterations before stopping. Defaults to 30.
            stop_epsilon (float, optional): Minimum step size before assumed converged. Defaults to 0.000001.
        """

        # TODO - Work within series or figure something else out

        self.__optimal_multipliers = np.copy(ref_white)

        if len(self.__mats) == 1:
            self.__optimal_mat = MatXyzToCamera(np.copy(self.__mats[0].mat), np.matmul(np.linalg.inv(self.__mats[0].mat), self.__optimal_multipliers))
            return 
        
        mat_k = []
        for mat in self.__mats:
            cct_and_tint = colour.temperature.XYZ_to_CCT_Ohno2013(mat.xyz)
            mat_k.append(cct_and_tint[0])
        
        mats = [x for _, x in sorted(zip(mat_k, self.__mats))] # Sort mats by CCT
        
        mat_t = [colour.temperature.XYZ_to_CCT_Ohno2013(np.matmul(np.linalg.inv(mat.mat), self.__optimal_multipliers))[1] for mat in mats]
        mat_t = [abs(get_ideal_duv(k) - x) for x,k in zip(mat_t, mat_k)]

        idx_lowest = [x for _, x in sorted(zip(mat_t, [y for y in range(len(mats))]))]

        if abs(idx_lowest[0] - idx_lowest[1]) == 1:
            mat_0 = mats[idx_lowest[0]]
            mat_1 = mats[idx_lowest[1]]
        else:
            mat_0 = mats[idx_lowest[0]]
            return MatXyzToCamera(np.copy(mat_0.mat), np.matmul(np.linalg.inv(mat_0.mat), self.__optimal_multipliers))

        # Solve blend factor for minimum error alongside two matrices
        best_xyz = np.matmul(np.linalg.inv(mat_0.mat), self.__optimal_multipliers)

        best = min(mat_t)
        best_bf = 0.0
        worst_bf = 1.0

        current = float("inf")

        i = 0
        while i < max_iters and abs(best_bf - worst_bf) > stop_epsilon:
            current = (worst_bf + best_bf) / 2
            current_xyz = np.matmul(np.linalg.inv(mat_0.interpolate(mat_1, current)), self.__optimal_multipliers)
            cct_and_tint = colour.temperature.XYZ_to_CCT_Ohno2013(current_xyz)
            tint = abs(get_ideal_duv(cct_and_tint[0]) - cct_and_tint[1])

            if tint <= best:
                best = tint
                best_xyz = current_xyz
                best_bf = current
            else:
                worst_bf = current

            i += 1
        
        self.__optimal_mat = MatXyzToCamera(mat_0.interpolate(mat_1, best_bf), best_xyz)
        return

    def get_reciprocal_multipliers(self) -> np.ndarray:
        """Get reciprocal neutral channel multipliers. Reciprocal is more useful because it can be immediately
        multiplied with color channels to achieve initial white balance pass.

        Returns:
            np.ndarray: Reciprocal channel multipliers.
        """
        return np.copy(1.0 / self.__optimal_multipliers)

    def get_matrix(self) -> MatXyzToCamera:
        """Get optimized color matrix under current parameters.

        Returns:
            MatXyzToCamera: Optimized color matrix.
        """
        return self.__optimal_mat

    def copy(self) -> CameraWhiteBalanceController:
        """Return a copy of this controller.

        Returns:
            CameraWhiteBalanceController: Deep copy of controller.
        """
        mats = []
        for mat in self.__mats:
            mats.append(MatXyzToCamera(mat.mat, mat.xyz))
        output = CameraWhiteBalanceController(mats, self.__optimal_multipliers)
        output.__optimal_mat = MatXyzToCamera(self.__optimal_mat.mat, self.__optimal_mat.xyz)
        return output

class CameraWhiteBalanceControllerFromExif(CameraWhiteBalanceController):
    def __init__(self, tags : Dict[str, Any]):
        """Create a white balance controller using DNG ColorMatrix calibrations.

        The returned controller will be pre-optimized for quality under camera AsShotNeutral.

        Args:
            tags (Dict[str, Any]): Tags dictionary as held by exifread.

        Raises:
            KeyError: Raised if the tag dictionary did not contained required tags for color management.
        """

        mats = exif_get_color_mat_sources(tags)
        if len(mats) == 0:
            raise KeyError("EXIF ColorMatrix tags or illuminant tags missing, could not create white balance controller!")

        try:
            multi_cam_wb = exif_get_as_shot_neutral(tags)
        except:
            raise KeyError("EXIF ColorMatrix tags or illuminant tags missing, could not create white balance controller!")

        super().__init__(mats, multi_cam_wb)