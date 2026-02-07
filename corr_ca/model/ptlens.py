from pySP.corr_ca.model.generic import NewtonRaphsonModel

import numpy as np

# TODO - Check PTLens for issues. When training on kodachrom1e configuration samples,
#        missing data at r < 0.3 tricks the solver into producing a bump below 0.3,
#        distorting near r=0. Fit is otherwise very good and error is very minimal
#        in training region.
#            There's some notes online about a scale factor d which is used to fix
#        how PTLens works, namely that it can end up changing the focal length. But
#        from what I can tell, our formulation of (1 - a - b - c) effectively forces
#        the same constraint (it's what the formulation given online...) and the fit
#        is good with zero zoom on real data (synthetic untested).
#            If I stretch the fitting graph, it can basically match Poly5 fit. Which
#        suggests maybe this is a scale thing and the fit would be better with auto d.

class PtLensCorrectionModel(NewtonRaphsonModel):
    """PTLens is a high-order distortion model capable of modelling
    more complex radial distortions.

    Because of its complexity some scale error may be introduced,
    especially on unseen data. Make sure to fit the polynomial to
    a full range of lens distortions radiuses to avoid curling
    near the extremes - it may predict a scaling bump where
    none is present.

    For a model which handles missing data more gracefully, try the
    Poly5 model.
    """

    def __init__(self, a : float = 0, b : float = 0, c : float = 0):
        super().__init__()
        self._a = a
        self._b = b
        self._c = c
    
    def _undistorted_to_distorted(self, undistorted):
        # Rd = a * Ru^4 + b * Ru^3 + c * Ru^2 + (1 - a - b - c) * Ru
        r2 = undistorted ** 2
        r3 = undistorted * r2
        r4 = undistorted * r3

        return (self._a * r4 +
                self._b * r3 +
                self._c * r2 +
                (1 - self._a - self._b - self._c) * undistorted)
    
    def _undistorted_to_distorted_prior(self, undistorted):
        r2 = undistorted ** 2
        r3 = undistorted * r2

        return (4 * self._a * r3 +
                3 * self._b * r2 +
                2 * self._c * undistorted +
                (1 - self._a - self._b - self._c))

    def get_coefficients(self):
        return np.array((self._a, self._b, self._c))
    
    def compute_coefficients(self, r_distorted_undistorted):
        r_d = r_distorted_undistorted[:,0]
        r_ud = r_distorted_undistorted[:,1]

        # Rd = a * Ru^4 + b * Ru^3 + c * Ru^2 + (1 - a - b - c) * Ru

        # Rearrange to matrix system
        # Rd / Ru = a * Ru^3 + b * Ru^2 + c * Ru + (1 - a - b - c)
        # Rd / Ru = a * (Ru^3 - 1) + b * (Ru^2 - 1) + c * (Ru - 1) + 1
        # (Rd / Ru) - 1 = a * (Ru^3 - 1) + b * (Ru^2 - 1) + c * (Ru - 1)

        # Make more readable
        # g = (Rd / Ru) - 1
        # x = (Ru^3 - 1)
        # y = (Ru^2 - 1)
        # z = (Ru - 1)

        # Now we have something solvable using least squares
        # g = ax + by + cz

        g = (r_d / r_ud) - 1
        x = r_ud ** 3 - 1
        y = r_ud ** 2 - 1
        z = r_ud      - 1

        m = np.dstack((x,y,z))[0]

        try:
            solution, residuals, rank, singular_values = np.linalg.lstsq(m, g)
            self._a, self._b, self._c = solution
            return True
        except np.linalg.LinAlgError:
            return False