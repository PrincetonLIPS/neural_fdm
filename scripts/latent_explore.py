"""
Predict the force densities and shapes of a batch of target shapes with a pretrained model.
"""

import os
from math import fabs
from statistics import mean, stdev
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.random as jrn
import matplotlib.pyplot as plt
import yaml
from compas.colors import Color, ColorMap
from compas.geometry import Line, Polygon
from jax import jit, vmap
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import datastructure_updated
from jax_fdm.visualization import Viewer
from neural_fofin import DATA
from neural_fofin.builders import (
    build_connectivity_structure_from_generator,
    build_data_generator,
    build_loss_function,
    build_mesh_from_generator,
    build_neural_model,
)
from neural_fofin.losses import print_loss_summary
from neural_fofin.serialization import load_model
from sklearn.decomposition import PCA

# ===============================================================================
# Script function
# ===============================================================================


def predict_batch(
    model_name,
    task_name,
    seed=None,
    batch_size=None,
    time_batch_inference=True,
    predict_in_sequence=True,
    slice=(0, -1),  # (50, 53) for bezier
    view=False,
    save=False,
    edgecolor="force",
):
    """
    Predict a batch of target shapes with a pretrained model.

    Parameters
    ___________
    model_name: `str`
        The model name.
        Supported models are formfinder, autoencoder, and piggy.
        Append the suffix `_pinn` to load model versions that were trained with a PINN loss.
    task_name: `str`
        The name of the YAML config file with the task hyperparameters.
    seed: `int`
        The random seed to generate a batch of target shapes.
        If `None`, it defaults to the input hyperparameters file.
    batch_size: `int` or `None`
        The size of the batch of target shapes.
        If `None`, it defaults to the input hyperparameters file.
    time_batch_inference: `bool`
        If `True`, report the inference time over a data batch, averaged over 10 jitted runs.
    predict_in_sequence: `bool`
        If `True`, predict every shape in the prescribed slice of the data batch.
    slice: `tuple`
        The start of the slice of the batch for saving and viewing.
    view: `bool`
        If `True`, view the predicted shapes.
    save: `bool`
        If `True`, save the predicted shapes as JSON files.
    edgecolor: `str`
        The color palette for the edges.
        Supported color palettes are "fd" to display force densities, and "force" to show forces.
    """
    START, STOP = slice
    if STOP == -1:
        STOP = batch_size

    EDGECOLOR = edgecolor  # force, fd

    CAMERA_CONFIG = {
        "position": (30.34, 30.28, 42.94),
        "target": (0.956, 0.727, 1.287),
        "distance": 20.0,
    }

    # load yaml file with hyperparameters
    with open(f"{task_name}.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # unpack parameters
    if seed is None:
        seed = config["seed"]
    training_params = config["training"]
    if batch_size is None:
        batch_size = training_params["batch_size"]

    generator_name = config["generator"]["name"]
    bounds_name = config["generator"]["bounds"]

    # randomness
    key = jrn.PRNGKey(seed)
    model_key, generator_key = jax.random.split(key, 2)

    # create data generator
    generator = build_data_generator(config)
    structure = build_connectivity_structure_from_generator(config, generator)
    mesh = build_mesh_from_generator(config, generator)
    compute_loss = build_loss_function(config, generator)

    # print info
    print(
        f"Making predictions with {model_name} on {generator_name} dataset with {bounds_name} bounds\n"
    )
    print(
        f"Structure size: {structure.num_vertices} vertices, {structure.num_edges} edges"
    )

    # load model
    filepath = os.path.join(DATA, f"{model_name}_{task_name}.eqx")
    _model_name = model_name.split("_")[0]
    model_skeleton = build_neural_model(_model_name, config, generator, model_key)
    model = load_model(filepath, model_skeleton)

    # sample data batch
    xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))

    encoding_fn = jit(vmap(model.encode))
    out = encoding_fn(xyz_batch)
    print(f"Encoded shapes: {out.shape}")
    # PCA on the shapes
    pca = PCA(n_components=2)
    pca.fit(out)
    # TODO: Plot the PCA of the embedding space i.e. the latent dim representation

    out_new = pca.transform(out)
    # plot the shapes on the 2D plane
    fig, ax = plt.subplots()
    ax.scatter(out_new[:, 0], out_new[:, 1], color="C0", label="Original shapes")
    # Sample a new batch but with noise
    xyz_batch_noise = vmap(generator)(jrn.split(generator_key, batch_size))

    # add noise
    noise = jax.random.normal(jrn.split(generator_key)[0], shape=xyz_batch.shape)

    xyz_batch_noise += 2.0 * (noise - 0.5)
    # encode the noisy shapes
    noisy_embed = encoding_fn(xyz_batch_noise)
    # use previous pca to project the noisy shapes
    noisy_pca = pca.transform(noisy_embed)
    ax.scatter(noisy_pca[:, 0], noisy_pca[:, 1], color="C1", label="Noisy shapes")
    ax.legend()
    plt.show()


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":
    from fire import Fire

    Fire(predict_batch)
