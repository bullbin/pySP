import numpy as np
from pySP.corr_ca.model.generic import NewtonRaphsonModel

class Poly5CorrectionModel(NewtonRaphsonModel):
    """Poly5 is a medium-complexity model capable of modelling
    a wide range of lens distortions.

    With less parameters and a simpler formulation than PTLens,
    Poly5 doesn't suffer from scaling issues or curly fitting.

    Poly5 is best for lens distortions with smooth falloff where
    it will produce predictable behaviour with flatness at r=0.
    """
    
    def __init__(self, h1 : float = 0, h2 : float = 0):
        super().__init__()
        self._h1 = h1
        self._h2 = h2
    
    def _undistorted_to_distorted(self, undistorted):
        # https://lensfun.github.io/manual/v0.3.2/group__Lens.html
        # https://www.imatest.com/support/docs/23-2/distortion_instructions/

        # Rd = Ru * (1 + h1 * Ru^2 + h2 * Ru^4)
        # Rd = Ru + h1 * Ru^3 + h2 * Ru^5

        r2 = undistorted ** 2
        r3 = undistorted * r2
        r5 = r3          * r2

        return (undistorted +
                self._h1 * r3 +
                self._h2 * r5)
    
    def _undistorted_to_distorted_prior(self, undistorted):
        # Rd = Ru * (1 + h1 * Ru^2 + h2 * Ru^4)
        # Rd/Ru = 1 + 3h1 * Ru^2 + 5h2 * Ru^4
        r2 = undistorted ** 2
        r4 = r2          * r2

        return (5 * self._h2 * r4 +
                3 * self._h1 * r2 +
                1)

    def get_coefficients(self):
        return np.array((self._h1, self._h2))
    
    def compute_coefficients(self, r_distorted_undistorted):
        r_d = r_distorted_undistorted[:,0]
        r_ud = r_distorted_undistorted[:,1]

        # Rd = Ru + h1 * Ru^3 + h2 * Ru^5

        # Rearrange to matrix system
        # Rd - Ru = h1 * Ru^3 + h2 * Ru^5

        # Make more readable
        # g = Rd - Ru
        # x = Ru^3
        # y = Ru^5

        # g = ax + by + cz
        # g = h1x + h2y

        # Now we have something solvable using least squares
        # g = ax + by + cz

        g = r_d - r_ud
        x = r_ud ** 3
        y = r_ud ** 5

        m = np.dstack((x,y))[0]

        try:
            solution, residuals, rank, singular_values = np.linalg.lstsq(m, g)
            self._h1, self._h2 = solution
            return True
        except np.linalg.LinAlgError:
            return False