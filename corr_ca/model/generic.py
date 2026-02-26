from abc import abstractmethod
from typing import List

import numpy as np

def get_empty_coord_field(image : np.ndarray) -> np.ndarray:
    y_shape, x_shape = tuple(image.shape[:2])
    if x_shape % 2 == 1 or y_shape % 2 == 1:
        # This shouldn't be hit - an image using Bayer array should always be divisible by
        #     2 on both axis since the filter array is 2x2
        raise ValueError("Incorrect shape for packing!")
    
    coords = np.zeros(shape=(image.shape[0] // 2, image.shape[1] // 2, 2), dtype=np.int32)
    coords[:,:,1] = np.arange(coords.shape[1])
    coords[:,:,0] = np.arange(coords.shape[0])[:,np.newaxis]

    return coords

def get_empty_radius_field(image : np.ndarray) -> np.ndarray:
    y_shape, x_shape = tuple(image.shape[:2])
    if x_shape % 2 == 1 or y_shape % 2 == 1:
        # This shouldn't be hit - an image using Bayer array should always be divisible by
        #     2 on both axis since the filter array is 2x2
        raise ValueError("Incorrect shape for packing!")

    # Compute radius lookup map
    radius = np.zeros(((y_shape // 2), (x_shape // 2)), dtype=np.float32)

    # Add squared axial deltas
    radius[:,] = (np.arange(radius.shape[1])[::-1] + 0.5) ** 2
    radius   += ((np.arange(radius.shape[0])[::-1] + 0.5) ** 2)[:,np.newaxis]

    # Square root for radius
    radius = np.sqrt(radius)

    # Normalize
    radius = radius / radius[0,0]

    return radius

class CaCorrectionModel():
    @abstractmethod
    def compute_coefficients(self, r_distorted_undistorted : np.ndarray) -> bool:
        pass

    @abstractmethod
    def get_coefficients(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_distorted(self, undistorted : np.ndarray) -> np.ndarray:
        pass

    def compute_error_statistics(self, r_distorted_undistorted : np.ndarray):
        raise NotImplementedError("")
    
    def get_distorted_coordinates(self, image : np.ndarray) -> np.ndarray:
        """For the entire image array, get the co-ordinates for sampling the distorted sample
        at the undistorted location. These can be used with cv2.remap to get the undistorted
        original.

        Args:
            image (np.ndarray): Input image. Corners will be used for maximum radius.

        Returns:
            np.ndarray: Output co-ordinates, such that undistorted input co-ordinates would get distorted to these.
        """

        # Get radial field and coord offset fields (top-left quarter)
        radius = get_empty_radius_field(image)
        coords = get_empty_coord_field(image)
        center = (np.array(image.shape[:2]) - 1) / 2
        
        deltas = np.copy(coords).astype(np.float32)
        deltas[:,:,0] -= center[0]
        deltas[:,:,1] -= center[1]

        # From our undistorted sampling points, get the distorted destination under the model
        distorted_r = self.get_distorted(radius.flatten()).reshape(-1, radius.shape[1])

        # Get scale factor between radius for rescaling coords
        scale_r = distorted_r / radius

        deltas[:,:,0] *= scale_r
        deltas[:,:,1] *= scale_r

        # Reflect everything
        full_coords = np.zeros(shape=(image.shape[0], image.shape[1], 2), dtype=np.float32)
        full_coords[:deltas.shape[0],:deltas.shape[1]] = deltas     # Top-left corner

        # Top-right corner
        working = np.copy(deltas)
        working[..., 1] = -working[..., 1]  # invert x
        full_coords[:deltas.shape[0],deltas.shape[1]:] = np.flip(working, axis=1)   # Flip across x, write top-right

        # Bottom
        working = np.copy(full_coords[:deltas.shape[0]])
        working[..., 0] = -working[..., 0]  # invert y
        full_coords[deltas.shape[0]:] = np.flip(working, axis=0)   # Flip across y, write bottom

        return full_coords

class ReversibleModelMixin():
    """Abstract class for any correction that can be reversed.
    """
    @abstractmethod
    def estimate_undistorted(self, distorted : np.ndarray, max_iterations : int = 8, max_epsilon : float = 0.00001) -> np.ndarray:
        pass

    def get_undistorted_coordinates(self, image : np.ndarray) -> np.ndarray:
        """For the entire image array, get the co-ordinates for sampling the undistorted sample
        at the distorted location. These can be used with cv2.remap to get the distorted
        original.

        One example usecase is aligning CA channels. Using the lens distortion model to get a
        forward mapping of red (distorted) to green (undistorted), we can reverse this to get a
        mapping of green onto red.

        Args:
            image (np.ndarray): Input image. Corners will be used for maximum radius.

        Returns:
            np.ndarray: Output co-ordinates, such that distorted input co-ordinates would get undistorted to these.
        """
        # TODO - Remove duplication, we can get output shape from delta array alone

        # Get radial field and coord offset fields (top-left quarter)
        radius = get_empty_radius_field(image)
        coords = get_empty_coord_field(image)
        center = (np.array(image.shape[:2]) - 1) / 2
        
        deltas = np.copy(coords).astype(np.float32)
        deltas[:,:,0] -= center[0]
        deltas[:,:,1] -= center[1]

        # From our undistorted sampling points, get the distorted destination under the model
        distorted_r = self.estimate_undistorted(radius.flatten()).reshape(-1, radius.shape[1])

        # Get scale factor between radius for rescaling coords
        scale_r = distorted_r / radius

        deltas[:,:,0] *= scale_r
        deltas[:,:,1] *= scale_r

        # Reflect everything
        full_coords = np.zeros(shape=(image.shape[0], image.shape[1], 2), dtype=np.float32)
        full_coords[:deltas.shape[0],:deltas.shape[1]] = deltas     # Top-left corner

        # Top-right corner
        working = np.copy(deltas)
        working[..., 1] = -working[..., 1]  # invert x
        full_coords[:deltas.shape[0],deltas.shape[1]:] = np.flip(working, axis=1)   # Flip across x, write top-right

        # Bottom
        working = np.copy(full_coords[:deltas.shape[0]])
        working[..., 0] = -working[..., 0]  # invert y
        full_coords[deltas.shape[0]:] = np.flip(working, axis=0)   # Flip across y, write bottom

        return full_coords

class NewtonRaphsonModel(CaCorrectionModel, ReversibleModelMixin):
    """Abstract class for polynomial-based lens correction models that can be reversed using Newton-Rhapson.
    """

    @abstractmethod
    def _undistorted_to_distorted(self, undistorted : np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _undistorted_to_distorted_prior(self, undistorted : np.ndarray) -> np.ndarray:
        pass

    def get_distorted(self, undistorted):
        return self._undistorted_to_distorted(undistorted)
    
    def estimate_undistorted(self, distorted : np.ndarray, max_iterations : int = 8, max_epsilon : float = 0.00001) -> np.ndarray:
        # Use Newton-Rhapson iterations to estimate root
        # TODO - Remove corr_ca_poly3, this uses the same method and can do the same.
        undistorted = np.zeros_like(distorted)

        def lens_prior(distorted : np.ndarray, undistorted : np.ndarray) -> np.ndarray:
            # Newton's method finds x such that f(x) = 0
            # We want to find undistorted co-ordinates.
            # Rd = f(Ru)
            # Create g(x) such that g(Ru) = f(Ru) - Rd
            # At g(Ru) = 0, f(Ru) = Rd.
            # Therefore, we can solve g to get undistorted using Newton's method.
            return self._undistorted_to_distorted(undistorted) - distorted

        err = np.inf
        last_err = err

        i = 0
        while i < max_iterations:
            prior = np.copy(undistorted)
            undistorted = undistorted - (lens_prior(distorted, undistorted) / self._undistorted_to_distorted_prior(undistorted))

            err = np.max(np.abs(prior - undistorted))
            if err < max_epsilon or err == last_err:
                break
            last_err = err
            i += 1

        return undistorted