from pySP.corr_ca.model.generic import NewtonRaphsonModel

import numpy as np

# TODO - Check Poly3 for issues. See PTLens model.

class Poly3CorrectionModel(NewtonRaphsonModel):
    """Poly3 is a simplified variant of the PTLens distortion model.
    
       It is fast and most suitable for correcting light distortions.
       For bigger distortions towards the edges of the frame, use the full
       PTLens model.
    """

    def __init__(self, initial_k1 : float = 0):
        """Apply corrections using the Poly3 correction model.

        Args:
            initial_k1 (float, optional): Initial k-value (PTLens b). Defaults to 0, producing a model with zero adjustment.
        """

        self._k1 = min(1.0, max(initial_k1, 0.0))
        super().__init__()

    def _undistorted_to_distorted(self, undistorted):
        return self._k1 * undistorted ** 3 + (1 - self._k1) * undistorted
    
    def _undistorted_to_distorted_prior(self, undistorted):
        return 3 * self._k1 * undistorted ** 2 + (1 - self._k1)

    def get_coefficients(self):
        return np.array((self._k1))

    def compute_coefficients(self, r_distorted_undistorted : np.ndarray):
        r_d = r_distorted_undistorted[:,0]
        r_ud = r_distorted_undistorted[:,1]
    
        # Rd = k1 * Ru^3 + (1 - k1) * Ru
        # Rd/Ru = k1 * Ru^2 + 1 - k1
        # Rd/Ru - 1 = k1 * Ru^2 - k1
        # Rd/Ru - 1 = k1 * (Ru^2 - 1)
        # (Rd/Ru - 1) / (Ru^2 - 1) = k1

        k1 = ((r_d / r_ud) - 1) / (r_ud ** 2 - 1)
        self._k1 = np.median(k1)
        return True