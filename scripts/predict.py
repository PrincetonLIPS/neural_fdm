"""
Predict the force densities and shapes of a batch of target shapes with a pretrained model.
"""

import os
from math import fabs
import yaml

import jax
from jax import vmap
import jax.numpy as jnp

import jax.random as jrn

from compas.colors import Color
from compas.colors import ColorMap

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import datastructure_updated
from jax_fdm.visualization import Viewer

from neural_fofin import DATA

from neural_fofin.builders import build_mesh_from_generator
from neural_fofin.builders import build_data_generator
from neural_fofin.builders import build_connectivity_structure_from_generator
from neural_fofin.builders import build_neural_model

from neural_fofin.losses import compute_loss

from neural_fofin.serialization import load_model


# ===============================================================================
# Script function
# ===============================================================================

def predict_batch(model, save=False, batch_size=None, start=50, stop=53, seed=None, edgecolor="fd", config="config"):
    """
    Predict a batch of target shapes with a pretrained model.

    Parameters
    ___________
    model: `str`
        The model name.
        Supported models are formfinder, autoencoder, and piggy.
        Append the suffix `_pinn` to load model versions that were trained with a PINN loss.
    save: `bool`
        If `True`, it will save the predicted shapes as JSON files.
    batch_size: `int` or `None`
        The size of the batch of target shapes.
        If `None`, it defaults to the input hyperparameters file.
    start: `int`
        The start of the slice of the batch.
    end: `int`
        The end of the slice of the batch.
    seed: `int`
        The random seed to generate a batch of target shapes.
        If `None`, it defaults to the input hyperparameters file.
    edgecolor: `str`
        The color palette for the edges.
        Supported color palettes are "fd" to display force densities, and "force" to show forces.
    config: `str`
        The filepath (without extension) of the YAML config file with the training hyperparameters.
    """
    NAME = model
    START = start
    STOP = stop
    CONFIG_NAME = config
    EDGECOLOR = edgecolor  # force, fd
    SAVE = save

    CAMERA_CONFIG = {
        "position": (30.34, 30.28, 42.94),
        "target": (0.956, 0.727, 1.287),
        "distance": 20.0,
    }

    # load yaml file with hyperparameters
    with open(f"{CONFIG_NAME}.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # unpack parameters
    if seed is None:
        seed = config["seed"]
    training_params = config["training"]
    if batch_size is None:
        batch_size = training_params["batch_size"]

    loss_params = config["loss"]

    # randomness
    key = jrn.PRNGKey(seed)
    model_key, generator_key = jax.random.split(key, 2)

    # create data generator
    generator = build_data_generator(config)
    structure = build_connectivity_structure_from_generator(generator)
    mesh = build_mesh_from_generator(generator)

    # load model
    filepath = os.path.join(DATA, f"{NAME}.eqx")
    _model_name = NAME.split("_")[0]
    model_skeleton = build_neural_model(_model_name, config, generator, model_key)
    model = load_model(filepath, model_skeleton)

    # sample data batch
    xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))

    # make (batched) predictions
    print(f"Making predictions with {NAME}")
    for i in range(START, STOP):
        xyz = xyz_batch[i]

        eqstate_hat, fd_params_hat = model.predict_states(xyz, structure)

        _, loss_terms = compute_loss(
            model,
            structure,
            xyz[None, :],
            loss_params,
            True
        )

        train_loss, shape_error, residual_error = loss_terms
        print(f"Shape {i}\tTrain loss: {train_loss:.4f}\tShape error: {shape_error:.4f}\tResidual error: {residual_error:.4f}")

        mesh_hat = datastructure_updated(mesh, eqstate_hat, fd_params_hat)
        network_hat = FDNetwork.from_mesh(mesh_hat)

        # export prediction
        if SAVE:
            filename = f"mesh_{i}"
            filepath = os.path.join(DATA, f"{filename}.json")
            mesh_hat.to_json(filepath)
            print(f"Saved prediction to {filepath}")

        # Create target mesh
        mesh_target = mesh.copy()
        _xyz = jnp.reshape(xyz, (-1, 3)).tolist()
        for idx, key in mesh.index_key().items():
            mesh_target.vertex_attributes(key, "xyz", _xyz[idx])

        # visualization
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
        viewer.add(
            mesh_hat,
            show_points=False,
            show_edges=False,
            opacity=0.2
        )

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
                   edgewidth=(0.01, 0.3),
                   edgecolor=edgecolor,
                   show_edges=True,
                   edges=[edge for edge in mesh.edges() if not mesh.is_edge_on_boundary(*edge)],
                   nodes=[node for node in mesh.vertices() if len(mesh.vertex_neighbors(node)) > 2],
                   show_loads=False,
                   loadscale=1.0,
                   show_reactions=True,
                   reactionscale=1.0,
                   reactioncolor=Color.from_rgb255(0, 150, 10),
                   )

        viewer.add(FDNetwork.from_mesh(mesh_target),
                   as_wireframe=True,
                   show_points=False,
                   linewidth=4.0,
                   color=Color.black().lightened()
                   )

        # show le crème
        viewer.show()


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(predict_batch)
