import jax.random as jrn
import jax.numpy as jnp

from neural_fofin.generators.bezier import BezierSurfaceAsymmetric
from neural_fofin.generators.bezier import BezierSurfaceSymmetric
from neural_fofin.generators.bezier import BezierSurfaceSymmetricDouble


# ===============================================================================
# Generators
# ===============================================================================

class BezierSurfacePointGenerator:
    """
    A generator that outputs point evaluated on a wiggled bezier surface.
    """
    def __init__(self, surface, u, v, minval, maxval):
        self._check_array_shapes(surface, minval, maxval)

        self.surface = surface
        self.u = u
        self.v = v
        self.minval = minval
        self.maxval = maxval

    def _check_array_shapes(self, surface, minval, maxval):
        """
        Verify that input shapes are consistent.
        """
        tile_shape = surface.grid.tile.shape
        minval_shape = minval.shape
        maxval_shape = maxval.shape

        assert minval_shape == tile_shape, f"{minval_shape} vs. {tile_shape}"
        assert maxval_shape == tile_shape, f"{maxval_shape} vs. {tile_shape}"

    def wiggle(self, key):
        """
        Sample a translation vector from a uniform distribution.
        """
        shape = self.surface.grid.tile.shape
        return jrn.uniform(key, shape=shape, minval=self.minval, maxval=self.maxval)

    def __call__(self, key, wiggle=True):
        """
        """
        if wiggle:
            transform = self.wiggle(key)

        points = self.surface.evaluate_points(self.u, self.v, transform)

        return jnp.ravel(points)


class BezierSurfaceSymmetricDoublePointGenerator(BezierSurfacePointGenerator):
    """
    A generator that outputs point evaluated on a wiggled, doubly-symmetric bezier surface.
    """
    def __init__(self, size, num_pts, u, v, minval, maxval):
        surface = BezierSurfaceSymmetricDouble(size, num_pts)
        super().__init__(surface, u, v, minval, maxval)


class BezierSurfaceSymmetricPointGenerator(BezierSurfacePointGenerator):
    """
    A generator that outputs point evaluated on a wiggled, symmetric bezier surface.
    """
    def __init__(self, size, num_pts, u, v, minval, maxval):
        surface = BezierSurfaceSymmetric(size, num_pts)
        super().__init__(surface, u, v, minval, maxval)


class BezierSurfaceAsymmetricPointGenerator(BezierSurfacePointGenerator):
    """
    A generator that outputs point evaluated on a wiggled bezier surface.
    """
    def __init__(self, size, num_pts, u, v, minval, maxval):
        surface = BezierSurfaceAsymmetric(size, num_pts)
        super().__init__(surface, u, v, minval, maxval)
