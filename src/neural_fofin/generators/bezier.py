from jax import lax
from jax import vmap

import jax.numpy as jnp

import matplotlib.pyplot as plt

from neural_fofin.generators.grids import PointGridAsymmetric
from neural_fofin.generators.grids import PointGridSymmetric
from neural_fofin.generators.grids import PointGridSymmetricDouble


# ===============================================================================
# Functions
# ===============================================================================

def factorial(n):
    """
    Calculate the factorial of a number.
    """
    return jnp.where(n < 0, 0, lax.exp(lax.lgamma(n + 1)))


def binomial_coefficient(n, i):
    """
    Compute the binomial coefficient.
    """
    return factorial(n) / (factorial(i) * factorial(n - i))


def bernstein_poly(n, t):
    """
    Compute all Bernstein polynomials of degree n at t using vectorized operations.
    """
    i = jnp.arange(n + 1)
    binomial_coeff = binomial_coefficient(n, i)

    return binomial_coeff * (t ** i) * ((1 - t) ** (n - i))


def degree_u(control_points):
    """
    Get the degree along the U direction of the Bezier surface.
    """
    n, m, _ = control_points.shape
    n -= 1.0

    return n


def degree_v(control_points):
    """
    Get the degree along the V direction of the Bezier surface.
    """
    n, m, _ = control_points.shape
    m -= 1.0

    return m


def bezier_surface_point(control_points, u, v):
    """
    Compute a point on a Bezier surface using a series of dot products.
    control_points is a 3D numpy array of shape (n+1, m+1, 3)
    u and v are the parameters (between 0 and 1)
    """
    n = degree_u(control_points)
    m = degree_v(control_points)

    # Compute the Bernstein polynomial values at u and v
    bernstein_u = bernstein_poly(n, u)
    bernstein_v = bernstein_poly(m, v)

    # Calculate the weighted sum of control points along the u direction
    weighted_u = jnp.dot(bernstein_u, control_points)  # shape becomes (m+1, 3)

    # Calculate the final point on the surface by combining the results across v
    point = jnp.dot(weighted_u.T, bernstein_v)

    return point


# Evaluate points
def evaluate_bezier_surface(control_points, u, v):
    """
    Sample a series of 3D points on a Bezier surface using vmap.
    control_points is a 3D numpy array of shape (n+1, m+1, 3)
    u and v are parameter arrays where to sample (between 0 and 1)
    """
    fn = vmap(vmap(bezier_surface_point,
                   in_axes=(None, 0, None)),
              in_axes=(None, None, 0))

    return fn(control_points, u, v)


def evaluate_bezier_surface_einsum(control_points, u, v):
    """
    Vectorized computation of a point on a Bezier surface.
    control_points is a 3D numpy array of shape (n+1, m+1, 3).
    u and v are arrays of parameters (between 0 and 1).
    """
    n = degree_u(control_points)
    m = degree_v(control_points)

    # Compute the Bernstein polynomial values at u and v
    bernstein_u = bernstein_poly(n, u[:, :, None])
    bernstein_v = bernstein_poly(m, v[:, :, None])

    # Calculate the weighted sum of control points along the u and v directions
    surface_points = jnp.einsum('ijk,lmi,lmj->lmk',
                                control_points,
                                bernstein_u,
                                bernstein_v)

    return surface_points


# ===============================================================================
# Surfaces
# ===============================================================================


class BezierSurface:
    """
    """
    def __init__(self, grid):
        self.grid = grid

    def control_points(self, transform=None):
        """
        """
        return self.grid.points(transform)

    def evaluate_points(self, u, v, transform=None):
        """
        """
        control_points = self.control_points(transform)
        return evaluate_bezier_surface(control_points, u, v)


class BezierSurfaceSymmetric(BezierSurface):
    """
    """
    def __init__(self, size, num_pts):
        grid = PointGridSymmetric(size, num_pts)
        super().__init__(grid)


class BezierSurfaceSymmetricDouble(BezierSurface):
    """
    """
    def __init__(self, size, num_pts):
        grid = PointGridSymmetricDouble(size, num_pts)
        super().__init__(grid)


class BezierSurfaceAsymmetric(BezierSurface):
    """
    """
    def __init__(self, size, num_pts):
        grid = PointGridAsymmetric(size, num_pts)
        super().__init__(grid)


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    grid_size = 4
    num_u = 11  # 12 for youtube
    num_v = 11

    # Control points rhino
    points = [
        [-5, -5, 0],
        [-5, -1.666667, 0],
        [-5, 1.666667, 0],
        [-5, 5, 0],
        [-1.666667, -5, 0],
        [-1.666667, -1.666667, 10],
        [-1.666667, 1.666667, 10],
        [-1.666667, 5, 0],
        [1.666667, -5, 0],
        [1.666667, -1.666667, 10],
        [1.666667, 1.666667, 10],
        [1.666667, 5, 0],
        [5, -5, 0],
        [5, -1.666667, 0],
        [5, 1.666667, 0],
        [5, 5, 0]
    ]
    # Control points Youtube
    # cx = [[-0.5, -2.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    # cy = [[2.0, 1.0, 0.0], [2.0, 0.0, -1.0], [2.0, 1.0, 1.0]]
    # cz = [[1.0, -1.0, 2.0], [0.0, -0.5, 2.0], [0.5, 1.0, 2.0]]
    # cx = jnp.array(cx)
    # cy = jnp.array(cy)
    # cz = jnp.array(cz)

    assert len(points) % grid_size == 0
    points = jnp.array(points)
    control_points = jnp.reshape(points, (grid_size, grid_size, 3))

    # U, V
    u = jnp.linspace(0.0, 1.0, num_u)
    v = jnp.linspace(0.0, 1.0, num_v)
    u_grid, v_grid = jnp.meshgrid(u, v)

    # surface_points = bezier_surface(control_points, u_grid, v_grid)
    surface_points = evaluate_bezier_surface(control_points, u, v)
    print("Surface Points Shape:", surface_points.shape)  # Should be (10, 10, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(surface_points[:, :, 0],
                    surface_points[:, :, 1],
                    surface_points[:, :, 2])

    ax.scatter(control_points[:, :, 0],
               control_points[:, :, 1],
               control_points[:, :, 2],
               edgecolors="face")

    plt.show()
