# the essentials
import os
import numpy as np
from scipy.spatial.distance import directed_hausdorff

# compas
from compas.colors import Color
from compas.geometry import Line

# jax fdm
from jax_fdm.datastructures import FDNetwork, FDMesh

from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import constrained_fdm

from jax_fdm.optimization import LBFGSB
from jax_fdm.optimization import OptimizationRecorder

from jax_fdm.parameters import EdgeForceDensityParameter

from jax_fdm.goals import NodePointGoal

from jax_fdm.losses import RootMeanSquaredError
from jax_fdm.losses import Loss

from jax_fdm.visualization import LossPlotter
from jax_fdm.visualization import Viewer


# ==========================================================================
# Parameters
# ==========================================================================

model = 6 # 1, 3, 6
name = "srf"
name_target = name

q0 = -0.1
px, py, pz = 0.0, 0.0, -0.1  # loads at each node
qmin, qmax = -20.0, -0.0  # min and max force densities

optimizer = LBFGSB  # the optimization algorithm
maxiter = 1000  # optimizer maximum iterations
tol = 1e-6  # optimizer tolerance

record = False  # True to record optimization history of force densities
export = False  # export result to JSON

# ==========================================================================
# Import network
# ==========================================================================

HERE = os.path.dirname(__file__)
FILE_IN = os.path.abspath(os.path.join(HERE, f"{name}_{int(model)}.json"))
mesh = FDMesh.from_json(FILE_IN)

# ==========================================================================
# Import target network
# ==========================================================================

FILE_IN = os.path.abspath(os.path.join(HERE, f"{name}_{int(model)}.json"))
mesh_target = FDMesh.from_json(FILE_IN)

# ==========================================================================
# Define structural system
# ==========================================================================

# data
# anchors = [node for node in mes.nodes() if network.is_leaf(node)]
anchors = mesh.vertices_on_boundary()

mesh.vertices_supports(anchors)
mesh.vertices_loads([px, py, pz], keys=mesh.vertices_free())
mesh.edges_forcedensities(q=q0)

# ==========================================================================
# Export FD network with problem definition
# ==========================================================================

if export:
    FILE_OUT = os.path.join(HERE, f"../../data/json/{name}_base.json")
    mesh.to_json(FILE_OUT)
    print("Problem definition exported to", FILE_OUT)

# ==========================================================================
# Define optimization parameters
# ==========================================================================

parameters = []
for edge in mesh.edges():
    parameter = EdgeForceDensityParameter(edge, qmin, qmax)
    parameters.append(parameter)

# ==========================================================================
# Define goals
# ==========================================================================

# edge lengths
goals = []
for node in mesh.vertices():
    if node in anchors:
        continue
    xyz = mesh_target.vertex_coordinates(node)
    goal = NodePointGoal(node, xyz)
    goals.append(goal)

# ==========================================================================
# Combine error functions and regularizer into custom loss function
# ==========================================================================

squared_error = RootMeanSquaredError(goals, alpha=1.0)
loss = Loss(squared_error)

# ==========================================================================
# Form-find mesh
# ==========================================================================

mesh0 = mesh.copy()
mesh = fdm(mesh)
mesh_fd = mesh.copy()

print(f"Load path: {round(mesh.loadpath(), 3)}")

# ==========================================================================
# Solve constrained form-finding problem
# ==========================================================================

optimizer = optimizer()
recorder = OptimizationRecorder(optimizer) if record else None

mesh = constrained_fdm(mesh0,
                       optimizer=optimizer,
                       loss=loss,
                       parameters=parameters,
                       maxiter=maxiter,
                       tol=tol,
                       callback=recorder)

# ==========================================================================
# Export optimization history
# ==========================================================================

if record and export:
    FILE_OUT = os.path.join(HERE, f"../../data/json/{name}_history.json")
    recorder.to_json(FILE_OUT)
    print("Optimization history exported to", FILE_OUT)

# ==========================================================================
# Plot loss components
# ==========================================================================

if record:
    plotter = LossPlotter(loss, mesh, dpi=150, figsize=(8, 4))
    plotter.plot(recorder.history)
    plotter.show()

# ==========================================================================
# Export JSON
# ==========================================================================

if export:
    FILE_OUT = os.path.join(HERE, f"../../data/json/{name}_optimized.json")
    mesh.to_json(FILE_OUT)
    print("Form found design exported to", FILE_OUT)

# ==========================================================================
# Hausdorff distance
# ==========================================================================

U = np.array([mesh.vertex_coordinates(node) for node in mesh.vertices()])
V = np.array([mesh_target.vertex_coordinates(node) for node in mesh_target.vertices()])
directed_u = directed_hausdorff(U, V)[0]
directed_v = directed_hausdorff(V, U)[0]
hausdorff = max(directed_u, directed_v)

print(f"Hausdorff distances: Directed U: {directed_u}\tDirected V: {directed_v}\tUndirected: {round(hausdorff, 4)}")

# ==========================================================================
# Report stats
# ==========================================================================

mesh.print_stats()

# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer(width=1600, height=900, show_grid=False, viewmode="lighted")

# modify view
# viewer.view.camera.zoom(-5)  # number of steps, negative to zoom out
# viewer.view.camera.rotation[2] = 0.0  # set rotation around z axis to zero
# modify view
viewer.view.camera.position = (30.34, 30.28, 42.94)# (29.109, 29.207, 33.251)
viewer.view.camera.target = (0.956,0.727,1.287) # (0.853, 0.605, 1.404)
viewer.view.camera.distance = 20.0

# optimized mesh
viewer.add(FDNetwork.from_mesh(mesh),
           edgewidth=(0.1, 0.25),           
           edgecolor="force",
           show_loads=True,
           loadscale=5.0,
           show_reactions=True,
           reactionscale=5.0)

viewer.add(mesh,
           show_points=False,
           show_edges=False,
           opacity=0.3)

# reference network
# viewer.add(FDNetwork.from_mesh(mesh_target),
#            as_wireframe=True,
#            show_points=False,
#            linewidth=1.0,
#            color=Color.grey().darkened())

# # draw lines to target
# for node in mesh.vertices():
#     pt = mesh.vertex_coordinates(node)
#     line = Line(pt, mesh_target.vertex_coordinates(node))
#     viewer.add(line, color=Color.grey())

# show le crème
viewer.show()
