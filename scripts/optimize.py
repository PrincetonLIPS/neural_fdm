"""
Predict the force densities and shapes of a batch of target shapes with a pretrained model.
"""
import os
from math import fabs
import yaml

import warnings

from time import perf_counter
from statistics import mean
from statistics import stdev

import jax
from jax import vmap
import jax.numpy as jnp

import jax.random as jrn

import equinox as eqx

from jaxopt import ScipyBoundedMinimize

from compas.colors import Color
from compas.colors import ColorMap

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import datastructure_updated
from jax_fdm.visualization import Viewer

from neural_fofin import DATA

from neural_fofin.builders import build_mesh_from_generator
from neural_fofin.builders import build_data_generator
from neural_fofin.builders import build_connectivity_structure_from_generator
from neural_fofin.builders import build_fd_decoder_parametrized
from neural_fofin.builders import build_loss_function

from neural_fofin.losses import print_loss_summary


# ===============================================================================
# Script function
# ===============================================================================

def match_batch(
        optimizer,
        task,
        param_init=None,
        blow=1e-3,
        bup=20.0,
        seed=None,
        batch_size=None,
        verbose=True,
        save=False,
        view=False,
        slice_start=0,
        slice_end=None,
        edgecolor="force"
):
    """
    Solve the prediction task on a batch target shapes with direct optimization with box constraints.
    The box constraints help generating compression-only or tension-only solutions.

    Parameters
    ___________
    optimizer: `str`
        The name gradient-based optimizer used to solve this task.
        Supported methods are "slsqp" and "lbfgsb".
    task: `str`
        The filepath (without extension) of the YAML file with the task hyperparameters.
    param_init: `float`
        If specified, it determines the starting value of all the model parameters.
        If not, then it samples parameters between b_low and b_up from a uniform distribution.
    b_low: `float`
        The lower bound of the box constraints on the model parameters.
    b_up: `float`
        The lower bound of the box constraints on the model parameters.
    seed: `int`
        The random seed to generate a batch of target shapes.
        If `None`, it defaults to the input hyperparameters file.
    batch_size: `int` or `None`
        The size of the batch of target shapes.
        If `None`, it defaults to the input hyperparameters file.
    verbose: `bool`
        If `True`, print to stdout intermediary results.
    save: `bool`
        If `True`, save the predicted shapes as JSON files.
    view: `bool`
        If `True`, view the predicted shapes.
    slice_start: `int`
        The start of the slice of the batch for saving and viewing.
    slice_end: `int`
        The start of the slice of the batch for saving and viewing.
    edgecolor: `str`
        The color palette for the edges.
        Supported color palettes are "fd" to display force densities, and "force" to show forces.
    """
    START = slice_start
    STOP = slice_end or -1
    CONFIG_NAME = task
    EDGECOLOR = edgecolor  # force, fd
    SAVE = save
    QMIN = blow
    QMAX = bup

    CAMERA_CONFIG = {
        "position": (30.34, 30.28, 42.94),
        "target": (0.956, 0.727, 1.287),
        "distance": 20.0,
    }

    # pick optimizer name
    optimizer_names = {"lbfgsb": "L-BFGS-B", "slsqp": "SLSQP"}
    optimizer_name = optimizer_names[optimizer]

    # load yaml file with hyperparameters
    with open(f"{CONFIG_NAME}.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # unpack parameters
    if seed is None:
        seed = config["seed"]
    training_params = config["training"]
    if batch_size is None:
        batch_size = training_params["batch_size"]

    generator_name = config['generator']['name']
    bounds_name = config['generator']['bounds']
    fd_params = config["fdm"]

    # randomness
    key = jrn.PRNGKey(seed)
    _, generator_key = jax.random.split(key, 2)

    # create data generator
    generator = build_data_generator(config)
    compute_loss = build_loss_function(config, generator)
    structure = build_connectivity_structure_from_generator(generator)
    mesh = build_mesh_from_generator(generator)

    # generate initial model parameters
    q0 = calculate_params_init(mesh, param_init, key, QMIN, QMAX)

    # create model
    print(f"Directly optimizing with {optimizer_name} for {generator_name} dataset with {bounds_name} bounds on seed {seed}")
    model = build_fd_decoder_parametrized(q0, mesh, fd_params)

    # sample data batch
    xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))

    # split mode
    diff_model, static_model = eqx.partition(model, eqx.is_inexact_array)

    # wrap loss function to meet jax and jaxopt's ideosyncracies
    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def compute_loss_diffable(diff_model, xyz_target):
        """
        """
        _model = eqx.combine(diff_model, static_model)
        return compute_loss(_model, structure, xyz_target, aux_data=False)

    # warmstart loss function to eliminate jit compilation time from perf measurements
    _ = compute_loss_diffable(diff_model, xyz_target=xyz_batch[None, 0])

    # define optimization function
    warnings.filterwarnings("ignore")

    opt = ScipyBoundedMinimize(
        fun=compute_loss_diffable,
        method=optimizer_name,
        jit=False,
        tol=1e-6,
        maxiter=5000,
        options={"disp": False},
        value_and_grad=True,
    )

    # define parameter bounds
    bound_low, bound_up = calculate_params_bounds(mesh, q0, QMIN, QMAX)
    bound_low_tree = eqx.tree_at(lambda tree: tree.q, diff_model, replace=(bound_low))
    bound_up_tree = eqx.tree_at(lambda tree: tree.q, diff_model, replace=(bound_up))
    bounds = (bound_low_tree, bound_up_tree)

    # optimize
    print("\nOptimizing shapes in sequence")
    opt_times = []

    loss_values = []
    shape_errors = []
    residual_errors = []
    smooth_errors = []
    loss_lists = [loss_values, shape_errors, residual_errors, smooth_errors]

    were_successful = 0
    if STOP == -1:
        STOP == batch_size

    xyz_slice = xyz_batch[START:STOP]
    num_opts = xyz_slice.shape[0]
    for i, xyz in enumerate(xyz_slice):

        xyz = xyz[None, :]

        # report start losses
        _, loss_terms = compute_loss(model, structure, xyz, aux_data=True)
        if verbose:
            print(f"Shape {i}")
            print_loss_summary(loss_terms, prefix="\tStart")

        # optimize
        start = perf_counter()
        diff_model_opt, opt_res = opt.run(diff_model, bounds, xyz)
        opt_time = perf_counter() - start
        # unite optimal and static submodels
        model_opt = eqx.combine(diff_model_opt, static_model)

        # evaluate loss function at optimum point
        _, loss_terms = compute_loss(model_opt, structure, xyz, aux_data=True)
        if verbose:
            print_loss_summary(loss_terms, prefix="\tEnd")
            print(f"\tOpt success?: {opt_res.success}")
            print(f"\tOpt iters: {opt_res.iter_num}")
            print(f"\tOpt time: {opt_time:.4f} sec")

        if opt_res.success:
            were_successful += 1

        opt_times.append(opt_time)

        for loss_container, loss_term in zip(loss_lists, loss_terms):
            loss_container.append(loss_term.item())

        # assemble datastructure for post-processing
        eqstate_hat, fd_params_hat = model_opt.predict_states(xyz, structure)
        mesh_hat = datastructure_updated(mesh, eqstate_hat, fd_params_hat)
        network_hat = FDNetwork.from_mesh(mesh_hat)
        # if verbose:
        #    network_hat.print_stats()

        # export prediction
        if SAVE:
            filename = f"mesh_{i}"
            filepath = os.path.join(DATA, f"{filename}.json")
            mesh_hat.to_json(filepath)
            print(f"Saved prediction to {filepath}")

        # visualization
        if view:
            # create target mesh
            mesh_target = mesh.copy()
            _xyz = jnp.reshape(xyz, (-1, 3)).tolist()
            for idx, key in mesh.index_key().items():
                mesh_target.vertex_attributes(key, "xyz", _xyz[idx])

            viewer = Viewer(
                width=900,
                height=900,
                show_grid=False,
                viewmode="lighted"
            )

            # modify view
            viewer.view.camera.position = CAMERA_CONFIG["position"]
            viewer.view.camera.target = CAMERA_CONFIG["target"]
            viewer.view.camera.distance = CAMERA_CONFIG["distance"]

            # approximated mesh
            # viewer.add(
            #     mesh_hat,
            #     show_points=False,
            #     show_edges=False,
            #     opacity=0.2
            # )

            # edge colors
            if EDGECOLOR == "force":

                color_end = Color.from_rgb255(12, 119, 184)
                color_start = Color.white()
                cmap = ColorMap.from_two_colors(color_start, color_end)

                edgecolor = {}
                forces = [fabs(network_hat.edge_force(edge)) for edge in network_hat.edges()]
                fmin = min(forces)
                fmax = max(forces)

                for edge in network_hat.edges():
                    force = network_hat.edge_force(edge) * -1.0
                    if force < 0.0:
                        _color = Color.from_rgb255(227, 6, 75)
                    else:
                        value = (force - fmin) / (fmax - fmin)
                        _color = cmap(value)

                    edgecolor[edge] = _color
            else:
                edgecolor = EDGECOLOR

            viewer.add(network_hat,
                       # edgewidth=(0.01, 0.2),
                       edgewith=0.01,
                       edgecolor=edgecolor,
                       show_edges=True,
                       edges=[edge for edge in mesh.edges() if not mesh.is_edge_on_boundary(*edge)],
                       nodes=[node for node in mesh.vertices() if len(mesh.vertex_neighbors(node)) > 2],
                       show_loads=False,
                       loadscale=1.0,
                       show_reactions=True,
                       reactionscale=5.0,
                       reactioncolor=Color.from_rgb255(0, 150, 10),
                       )

            # viewer.add(FDNetwork.from_mesh(mesh_target),
            #            as_wireframe=True,
            #            show_points=False,
            #            linewidth=4.0,
            #            color=Color.black().lightened()
            #            )

            # show le crème
            viewer.show()

    # report optimization statistics
    print(f"\nSuccessful optimizations: {were_successful}/{num_opts}")
    print(f"Optimization time over {num_opts} optimizations (s): {mean(opt_times):.4f} (+-{stdev(opt_times):.4f})")
    print(f"Loss value over {num_opts} optimizations: {mean(loss_values):.4f} (+-{stdev(loss_values):.4f})")
    print(f"Shape error over {num_opts} optimizations: {mean(shape_errors):.4f} (+-{stdev(shape_errors):.4f})")
    print(f"Residual error over {num_opts} optimizations: {mean(residual_errors):.4f} (+-{stdev(residual_errors):.4f})")
    if len(smooth_errors) > 0:
        print(f"Smoothness error over {num_opts} optimizations: {mean(smooth_errors):.4f} (+-{stdev(smooth_errors):.4f})")


# ===============================================================================
# Helper functions
# ===============================================================================


def calculate_params_init(mesh, param_init, key, minval, maxval):
    """
    """
    num_edges = mesh.number_of_edges()

    signs = []
    for edge in mesh.edges():
        sign = -1.0  # compression by default
        # for tower task
        if mesh.edge_attribute(edge, "tag") == "cable":
            sign = 1.0
        signs.append(sign)

    signs = jnp.array(signs)

    if param_init is not None:
        q0 = jnp.ones(num_edges) * param_init
    else:
        q0 = jrn.uniform(key, shape=(num_edges, ), minval=minval, maxval=maxval)

    return q0 * signs


def calculate_params_bounds(mesh, q0, minval, maxval):
    """
    """
    bound_low = []
    bound_up = []
    for edge in mesh.edges():
        # compression by default
        b_low = maxval * -1.0
        b_up = minval * -1.0
        # for tower task
        if mesh.edge_attribute(edge, "tag") == "cable":
            b_low = minval
            b_up = maxval

        bound_low.append(b_low)
        bound_up.append(b_up)

    bound_low = jnp.array(bound_low)
    bound_up = jnp.array(bound_up)

    return bound_low, bound_up


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(match_batch)
