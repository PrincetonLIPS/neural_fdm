import numpy as np

from jax import vmap

import jax.random as jrn

import jax.numpy as jnp


# ===============================================================================
# Generators
# ===============================================================================

class TubePointGenerator:
    """
    A generator that outputs point evaluated on a wiggled tube.
    """
    pass


class EllipticalTubePointGenerator(TubePointGenerator):
    """
    A generator that outputs point evaluated on a wiggled elliptical tube.
    """
    def __init__(
            self,
            height,
            radius,
            num_sides,
            num_levels,
            num_rings,
            minval,
            maxval):

        # sanity checks
        assert num_rings >= 3, "Must include at least 1 ring in the middle!"
        self._check_array_shapes(num_rings, minval, maxval)

        self.height = height
        self.radius = radius

        self.num_sides = num_sides
        self.num_levels = num_levels
        self.num_rings = num_rings

        self.minval = minval
        self.maxval = maxval

        self.indices_rings = self._calculate_ring_indices()
        self.indices_rings_ravel = self._calculate_ring_indices_ravel()
        self.indices_rings_free_ravel = self._calculate_ring_indices_free_ravel()

        self.shape_tube = (num_levels, num_sides, 3)
        self.shape_rings = (num_rings, num_sides, 3)

    def __call__(self, key, wiggle=True):
        """
        Generate points.
        """
        # points = self.points_on_ellipses(key)
        points = self.points_on_tube(key, wiggle)

        return jnp.ravel(points)

    def _calculate_ring_indices(self):
        """
        Compute the indices of the rings in the sequence of levels.
        """
        step = int(self.num_levels / (self.num_rings - 1))

        indices = [0] + list(range(step, self.num_levels - 1, step)) + [self.num_levels - 1]
        indices = jnp.array(indices, dtype=jnp.int64)

        assert indices.size == self.num_rings

        return indices

    def _calculate_ring_indices_ravel(self):
        """
        Compute the indices of the rings in the sequence of levels.
        """
        indices = []
        for index in self.indices_rings:
            start = index * self.num_sides
            end = start + self.num_sides
            indices.extend(range(start, end))

        indices = jnp.array(indices, dtype=jnp.int64)

        return indices

    def _calculate_ring_indices_free_ravel(self):
        """
        Compute the indices of the rings in the sequence of levels.
        """
        indices = []
        for index in self.indices_rings[1:-1]:
            start = index * self.num_sides
            end = start + self.num_sides
            indices.extend(range(start, end))

        indices = jnp.array(indices, dtype=jnp.int64)

        return indices

    def wiggle(self, key):
        """
        Sample transformation vectors from a uniform distribution.
        """
        return self.wiggle_radii(key), self.wiggle_angle(key)

    def wiggle_radii(self, key):
        """
        Sample a 2D transformation vector from a uniform distribution.
        """
        shape = (self.num_rings, 2)
        minval = self.minval[:2]
        maxval = self.maxval[:2]

        return jrn.uniform(key, shape=shape, minval=minval, maxval=maxval)

    def wiggle_angle(self, key):
        """
        Sample a transformation vector from a uniform distribution.
        """
        shape = (self.num_rings,)
        minval = self.minval[2]
        maxval = self.maxval[2]

        return jrn.uniform(key, shape=shape, minval=minval, maxval=maxval)

    def points_on_tube(self, key=None, wiggle=False):
        """
        """
        heights = jnp.linspace(0.0, self.height, self.num_levels)
        radii = jnp.ones(shape=(self.num_levels, 2)) * self.radius
        angles = jnp.ones(shape=(self.num_levels,))

        if wiggle:
            wiggle_radii, wiggle_angle = self.wiggle(key)
            wiggle_radii = wiggle_radii * self.radius
            radii = radii.at[self.indices_rings, :].set(wiggle_radii)
            angles = angles.at[self.indices_rings].set(wiggle_angle)

        points = points_on_ellipses(
            radii[:, 0],
            radii[:, 1],
            heights,
            self.num_sides,
            angles,
        )

        return points

    def points_on_ellipses(self, key):
        """
        """
        heights = jnp.linspace(0.0, self.height, self.num_rings)

        radii, angles = self.wiggle(key)
        radii = radii * self.radius

        points = points_on_ellipses(
            radii[:, 0],
            radii[:, 1],
            heights,
            self.num_sides,
            angles,
        )

        return points

    def _check_array_shapes(self, num_rings, minval, maxval):
        """
        Verify that input shapes are consistent.
        """
        shape = (3, )
        minval_shape = minval.shape
        maxval_shape = maxval.shape

        assert minval_shape == shape, f"{minval_shape} vs. {shape}"
        assert maxval_shape == shape, f"{maxval_shape} vs. {shape}"


class CircularTubePointGenerator(EllipticalTubePointGenerator):
    """
    A generator that outputs point evaluated on a wiggled elliptical tube.
    """
    def wiggle_radii(self, key):
        """
        Sample a 2D transformation vector from a uniform distribution.
        """
        shape = (self.num_rings,)
        minval = self.minval[0]
        maxval = self.maxval[0]

        return jrn.uniform(key, shape=shape, minval=minval, maxval=maxval)

    def points_on_tube(self, key=None, wiggle=False):
        """
        """
        heights = jnp.linspace(0.0, self.height, self.num_levels)
        radii = jnp.ones(shape=(self.num_levels,)) * self.radius
        angles = jnp.ones(shape=(self.num_levels,))

        if wiggle:
            wiggle_radii, wiggle_angle = self.wiggle(key)
            wiggle_radii = wiggle_radii * self.radius
            radii = radii.at[self.indices_rings].set(wiggle_radii)
            angles = angles.at[self.indices_rings].set(wiggle_angle)

        points = points_on_ellipses(
            radii,
            radii,
            heights,
            self.num_sides,
            angles,
        )

        return points

    def points_on_ellipses(self, key):
        """
        """
        heights = jnp.linspace(0.0, self.height, self.num_rings)

        radii, angles = self.wiggle(key)
        radii = radii * self.radius

        points = points_on_ellipses(
            radii,
            radii,
            heights,
            self.num_sides,
            angles,
        )

        return points


# ===============================================================================
# Helper functions
# ===============================================================================

def points_on_ellipse_xy(radius_1, radius_2, num_sides, angle=0.0):
    """
    Sample points on an ellipse.

    Notes
    -----
    The first and last points are not equal.
    """
    angles = 2 * jnp.pi * jnp.linspace(0.0, 1.0, num_sides + 1)
    angles = jnp.reshape(angles, (-1, 1))
    xs = radius_1 * jnp.cos(angles)
    ys = radius_2 * jnp.sin(angles)

    points = jnp.hstack((xs, ys))[:-1]

    # Calculate rotation matrix
    theta = jnp.radians(angle)
    rotation_matrix = jnp.array([
        [jnp.cos(theta), -jnp.sin(theta)],
        [jnp.sin(theta), jnp.cos(theta)]
    ])

    # Rotate points
    points = points @ rotation_matrix.T

    return points


def points_on_ellipse(radius_1, radius_2, height, num_sides, angle=0.0):
    """
    Sample points on an ellipse at a given z coordinate.

    Notes
    -----
    The first and last points are not equal.
    """
    xy = points_on_ellipse_xy(radius_1, radius_2, num_sides, angle)
    z = jnp.ones((num_sides, 1)) * height

    return jnp.hstack((xy, z))


def points_on_ellipses(radius_1, radius_2, heights, num_sides, angles):
    """
    Sample points on an sequence of ellipses distributed over an array of heights.

    Notes
    -----
    The first and last points per ellipse are not equal.
    """
    polygon_fn = vmap(points_on_ellipse, in_axes=(0, 0, 0, None, 0))

    return polygon_fn(radius_1, radius_2, heights, num_sides, angles)


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from compas.geometry import Polygon
    from jax_fdm.visualization import Viewer
    from neural_fofin.generators import EllipticalTubePointGenerator

    height = 10.0
    radius = 2.0
    radius_1 = 1.0
    radius_2 = 1.25

    num_sides = 4
    num_levels = 11  # Use 11 or 21 or 31
    num_rings = 3  # Use 3 or 4, 2 of them will be supported

    xy = points_on_ellipse_xy(radius_1, radius_2, num_sides)
    assert xy.shape == (num_sides, 2)
    xyz = jnp.hstack((xy, jnp.ones((num_sides, 1)) * height))
    assert xyz.shape == (num_sides, 3)

    xyz2 = points_on_ellipse(radius_1, radius_2, height, num_sides)
    assert xyz2.shape == (num_sides, 3)
    assert jnp.allclose(xyz, xyz2)

    heights = jnp.linspace(0, height, num_levels)
    r1 = jnp.ones_like(heights) * radius_1
    r2 = jnp.ones_like(heights) * radius_2
    angles = jnp.zeros_like(heights)
    xyzs = points_on_ellipses(r1, r2, heights, num_sides, angles)
    assert xyzs.shape == (num_levels, num_sides, 3)

    print("\nGenerating")
    generator = EllipticalTubePointGenerator(
        height,
        radius,
        num_sides,
        num_levels,
        num_rings,
        minval=jnp.array([0.5, 0.5, 0.0]),
        maxval=jnp.array([2.0, 2.0, 0.0]),
    )

    print(generator.indices_rings, generator.shape_tube)

    # randomness
    seed = 91
    key = jrn.PRNGKey(seed)
    _, generator_key = jrn.split(key, 2)
    batch_size = 3

    # sample data batch
    xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))
    print(f"{xyz_batch.shape=}")
    xyz_ellipse_batch = vmap(generator.points_on_ellipses)(jrn.split(generator_key, batch_size))

    # xyz_batch = jnp.reshape(xyz_batch, (batch_size, num_levels, num_sides, 3))
    # print(f"{xyz_batch.shape=}")

    for xyzs, xyzs_ellipse in zip(xyz_batch, xyz_ellipse_batch):
        print(f"{xyzs.shape=}")
        xyzs = jnp.reshape(xyzs, generator.shape_rings)
        assert jnp.allclose(xyzs, xyzs_ellipse)
    raise
    #     assert xyzs.shape == (num_rings, num_sides, 3)
    #     print("Generated")

    #     print("Viewing")
    #     viewer = Viewer(width=1600, height=900, show_grid=True)

    #     for xyz in xyzs:
    #         polygon = Polygon(xyz.tolist())
    #         viewer.add(polygon, opacity=0.5)

    #     viewer.show()

    print("Viewing")
    from neural_fofin.builders import build_mesh_from_generator
    from jax_fdm.equilibrium import fdm
    from jax_fdm.datastructures import FDNetwork

    mesh = build_mesh_from_generator(generator)
    mesh.edges_forcedensities(10.0)
    # mesh = fdm(mesh, sparse=False)

    viewer = Viewer(width=1600, height=900, show_grid=True)
    viewer.add(mesh, opacity=0.5)
    viewer.add(FDNetwork.from_mesh(mesh), show_nodes=True, nodesize=0.2)
    viewer.show()
