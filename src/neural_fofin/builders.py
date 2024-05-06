import jax

import optax

import jax.numpy as jnp

from functools import partial

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumMeshStructure

from neural_fofin.generators import BezierSurfacePointGenerator
from neural_fofin.generators import BezierSurfaceAsymmetricPointGenerator
from neural_fofin.generators import BezierSurfaceSymmetricPointGenerator
from neural_fofin.generators import BezierSurfaceSymmetricDoublePointGenerator

from neural_fofin.generators import CircularTubePointGenerator
from neural_fofin.generators import EllipticalTubePointGenerator
from neural_fofin.generators import TubePointGenerator

from neural_fofin.mesh import create_mesh_from_bezier_generator
from neural_fofin.mesh import create_mesh_from_tube_generator

from neural_fofin.losses import compute_loss
from neural_fofin.losses import compute_loss_shape_residual
from neural_fofin.losses import compute_loss_residual_smoothness

from neural_fofin.models import AutoEncoder
from neural_fofin.models import AutoEncoderPiggy
from neural_fofin.models import MLPEncoder
from neural_fofin.models import FDDecoder
from neural_fofin.models import FDDecoderParametrized
from neural_fofin.models import MLPDecoder
from neural_fofin.models import MLPDecoderXL


# ===============================================================================
# Tower shape generator bounds
# ===============================================================================

def ellipse_minmax_values():
    """
    The boundary values for an ellipse.
    """
    # radius 1, radius 2, rotation
    # radii are scale factors relative to the base radius of a tower
    minval = [0.5, 0.5, 0.0]
    maxval = [2.0, 2.0, 0.0]

    return minval, maxval


def ellipse_rotated_minmax_values():
    """
    The boundary values for rotated ellipse.
    """
    # radius 1, radius 2, rotation
    # radii are scale factors relative to the base radius of a tower
    minval = [0.5, 0.5, -45.0]
    maxval = [2.0, 2.0, 45.0]

    return minval, maxval


def get_tower_generator_minmax_values(name, bounds):
    """
    """
    experiments = {
        "straight": ellipse_minmax_values,
        "twisted": ellipse_rotated_minmax_values,
    }

    values_fn = experiments.get(bounds)
    if not values_fn:
        raise KeyError(f"Experiment bounds: {bounds} is currently unsupported!")

    # generate values on a quarter tile
    minval, maxval = values_fn()

    # array-ify
    minval = jnp.array(minval)
    maxval = jnp.array(maxval)

    return minval, maxval


# ===============================================================================
# Bezier shape generator bounds
# ===============================================================================

def pillow_minmax_values():
    """
    The boundary XYZ values for the pillow tile.
    """
    minval = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    maxval = [
        [0.0, 0.0, 10.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    return minval, maxval


def dome_minmax_values():
    """
    The boundary XYZ values for the dome tile.
    """
    minval = [
        [0.0, 0.0, 1.0],
        [-5.0, 0.0, 0.0],
        [0.0, -5.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    maxval = [
        [0.0, 0.0, 10.0],
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    return minval, maxval


def saddle_minmax_values():
    """
    The boundary XYZ values for the dome tile.
    """
    minval = [
        [0.0, 0.0, 1.0],
        [-5.0, 0.0, 0.0],
        [0.0, -5.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    maxval = [
        [0.0, 0.0, 10.0],
        [5.0, 0.0, 10.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    return minval, maxval


def get_bezier_generator_minmax_values(name, bounds):
    experiments = {
        "pillow": pillow_minmax_values,
        "dome": dome_minmax_values,
        "saddle": saddle_minmax_values
    }

    values_fn = experiments.get(bounds)
    if not values_fn:
        raise KeyError(f"Experiment bounds: {bounds} is currently unsupported!")

    # generate values on a quarter tile
    minval, maxval = values_fn()

    # concatenate bounds based on whether generator is symmetric or 2-symmetric
    name_parts = name.split("_")
    if "symmetric" in name_parts:
        if "double" not in name_parts:
            minval = minval + minval
            maxval = maxval + maxval
    elif "asymmetric" in name_parts:
        minval = minval + minval + minval + minval
        maxval = maxval + maxval + maxval + maxval

    # array-ify
    minval = jnp.array(minval)
    maxval = jnp.array(maxval)

    return minval, maxval


# ===============================================================================
# Data generators
# ===============================================================================

def build_tube_point_generator(generator_params):
    """
    """
    # unpack parameters
    name = generator_params["name"]
    bounds = generator_params["bounds"]

    height = generator_params["height"]
    radius = generator_params["radius"]
    num_sides = generator_params["num_sides"]
    num_levels = generator_params["num_levels"]
    num_rings = generator_params["num_rings"]

    # wiggle bounds for task
    minval, maxval = get_tower_generator_minmax_values(name, bounds)

    # Create data generator
    generators = {
        "ellipse": EllipticalTubePointGenerator,
        "circle": CircularTubePointGenerator,
    }

    name = generator_params["name"].split("_")[-1]
    generator = generators.get(name)
    if not generator:
        raise ValueError(f"Generator {name} is not supported yet!")

    return generator(height, radius, num_sides, num_levels, num_rings, minval, maxval)


def build_bezier_point_generator(generator_params):
    """
    """
    # unpack parameters
    name = generator_params["name"]
    num_u = generator_params["num_uv"]
    num_v = generator_params["num_uv"]
    size = generator_params["size"]
    num_pts = generator_params["num_points"]
    bounds = generator_params["bounds"]

    # wiggle bounds for task
    minval, maxval = get_bezier_generator_minmax_values(name, bounds)

    # Create data generator
    u = jnp.linspace(0.0, 1.0, num_u)
    v = jnp.linspace(0.0, 1.0, num_v)

    # Create data generator
    generators = {
        "bezier_symmetric": BezierSurfaceSymmetricPointGenerator,
        "bezier_symmetric_double": BezierSurfaceSymmetricDoublePointGenerator,
        "bezier_asymmetric": BezierSurfaceAsymmetricPointGenerator
    }

    generator = generators.get(name)
    if not generator:
        raise ValueError(f"Generator {name} is not supported yet!")

    return generator(size, num_pts, u, v, minval, maxval)


def build_data_generator(config):
    """
    TODO: Pick generator based on task name
    """
    # unpack parameters
    generator_params = config["generator"]

    # pick generator function
    generator_builders = {
        "bezier": build_bezier_point_generator,
        "tower": build_tube_point_generator
    }

    name = generator_params["name"].split("_")[0]

    generator_builder = generator_builders.get(name)
    if not generator_builder:
        raise ValueError(f"Generator {name} is not supported yet!")

    # build bezier generator
    return generator_builder(generator_params)


# ===============================================================================
# Mesh
# ===============================================================================

def build_mesh_from_generator(generator):
    """
    Generate a JAX FDM mesh according to the generator type.
    """
    if isinstance(generator, BezierSurfacePointGenerator):
        mesh_builder = create_mesh_from_bezier_generator
    elif isinstance(generator, TubePointGenerator):
        mesh_builder = create_mesh_from_tube_generator
    else:
        raise ValueError(f"Cannot make meshes with generator {generator}!")

    return mesh_builder(generator)


# ===============================================================================
# Structure (Graph)
# ===============================================================================

def build_connectivity_structure_from_generator(generator):
    """
    """
    # generate base FD mesh
    mesh = build_mesh_from_generator(generator)

    return EquilibriumMeshStructure.from_mesh(mesh)


# ===============================================================================
# Activation functions
# ===============================================================================

def get_activation_fn(name):
    """
    """
    functions = {
        "elu": jax.nn.elu,
        "relu": jax.nn.relu,
        "softplus": jax.nn.softplus
    }

    activation_fn = functions.get(name)
    if not activation_fn:
        raise KeyError(f"Activation name: {name} is currently unsupported!")

    return activation_fn


# ===============================================================================
# Optimizers
# ===============================================================================

def get_optimizer_fn(name):
    """
    Fetch the optimizer function.
    """
    optimizers = {
        "adam": optax.adam,
        "sgd": optax.sgd,
    }

    optimizer_fn = optimizers.get(name)
    if not optimizer_fn:
        raise KeyError(f"Optimize name: {name} is currently unsupported!")

    return optimizer_fn


def build_optimizer(config):
    """
    Construct an optimizer.
    """
    hyperparams = config["optimizer"]

    name = hyperparams["name"]
    learning_rate = hyperparams["learning_rate"]
    assert isinstance(learning_rate, float)

    optimizer_fn = get_optimizer_fn(name)
    optimizer = optimizer_fn(learning_rate=learning_rate)

    return optimizer


# ===============================================================================
# Loss functions
# ===============================================================================

def build_loss_function(config, generator):
    """
    Build a loss function.
    """
    task_name = config["generator"]["name"]
    loss_params = config["loss"]

    if "bezier" in task_name:
        _loss_fn = compute_loss_shape_residual

    elif "tower" in task_name:
        loss_params["residual"]["indices"] = generator.indices_rings_free_ravel
        _loss_fn = compute_loss_residual_smoothness

    loss_fn = partial(
        compute_loss,
        loss_fn=_loss_fn,
        loss_params=loss_params
    )

    return loss_fn


# ===============================================================================
# Force density solver
# ===============================================================================

def build_fd_model():
    """
    Dense, because batching rule is undefined to vmap a sparse model.
    """
    fd_model = EquilibriumModel(
        tmax=1,
        eta=1e-6,
        is_load_local=False,
        itersolve_fn=None,
        implicit_diff=True,
        verbose=False
        )

    return fd_model


def calculate_edges_mask(mesh):
    """
    Mask out to indicate what mesh edges are fully supported.
    """
    mask_edges = []
    for edge in mesh.edges():
        mask_val = 1.0
        if mesh.is_edge_fully_supported(edge):
            mask_val = 0.0
            # NOTE: for tower task
            if mesh.edge_attribute(edge, "tag") == "ring":
                if not mesh.is_edge_on_boundary(*edge):
                    mask_val = 1.0
        mask_edges.append(mask_val)

    return jnp.array(mask_edges, dtype=jnp.int64)


def calculate_edges_stress_signs(mesh):
    """
    Integer array to indicate what mesh edges are in compression and in tension
    """
    signs = []
    for edge in mesh.edges():
        sign = -1  # compression by default
        # for tower task
        if mesh.edge_attribute(edge, "tag") == "cable":
            sign = 1
        signs.append(sign)

    return jnp.array(signs, dtype=jnp.int64)


# ===============================================================================
# Decoders
# ===============================================================================

def build_fd_decoder_parametrized(q0, mesh, params):
    """
    """
    # unpack hyperparams
    load = params["load"]

    # create FD model
    fd_model = build_fd_model()

    # get mask of supported edges
    mask_edges = calculate_edges_mask(mesh)

    # instantiate FD decoder
    decoder = FDDecoderParametrized(
        q0,
        fd_model,
        load,
        mask_edges
        )

    return decoder


def build_fd_decoder(mesh, params):
    """
    """
    # unpack hyperparams
    load = params["load"]

    # create FD model
    fd_model = build_fd_model()

    # get mask of supported edges
    mask_edges = calculate_edges_mask(mesh)

    # instantiate FD decoder
    decoder = FDDecoder(
        fd_model,
        load,
        mask_edges
        )

    return decoder


def build_neural_decoder(mesh, key, params):
    """
    """
    # unpack hyperparameters
    nn_params, fd_params = params

    # get neural network params
    include_xl = nn_params["include_params_xl"]
    hidden_layer_size = nn_params["hidden_layer_size"]
    hidden_layer_num = nn_params["hidden_layer_num"]
    activation_name = nn_params["activation_fn_name"]

    # get load
    load = fd_params["load"]

    # mesh quantities
    num_vertices = mesh.number_of_vertices()
    num_edges = mesh.number_of_edges()
    num_vertices_free = len(list(mesh.vertices_free()))
    num_vertices_fixed = len(list(mesh.vertices_fixed()))

    # get mask of supported edges
    mask_edges = calculate_edges_mask(mesh)

    # define size of input layer
    in_size = num_edges
    decoder_cls = MLPDecoder

    if include_xl:
        in_size += num_vertices_fixed * 3
        in_size += num_vertices
        decoder_cls = MLPDecoderXL

    # instantiate MLP
    decoder = decoder_cls(
        load=load,
        mask_edges=mask_edges,
        in_size=in_size,
        out_size=num_vertices_free * 3,
        width_size=hidden_layer_size,
        depth=hidden_layer_num,
        activation=get_activation_fn(activation_name),
        key=key
        )

    return decoder


# ===============================================================================
# Encoders
# ===============================================================================

def build_neural_encoder(mesh, key, params, generator):
    """
    """
    # unpack hyperparameters
    nn_params, generator_params = params

    hidden_layer_size = nn_params["hidden_layer_size"]
    hidden_layer_num = nn_params["hidden_layer_num"]
    activation_name = nn_params["activation_fn_name"]
    final_activation_name = nn_params["final_activation_fn_name"]

    # mesh quantities
    num_vertices = mesh.number_of_vertices()
    num_edges = mesh.number_of_edges()

    # get edges stress signs
    edges_signs = calculate_edges_stress_signs(mesh)

    # define input size
    in_size = num_vertices
    is_tower_task = "tower" in generator_params["name"]
    if is_tower_task:
        in_size = generator_params["num_rings"] * generator_params["num_sides"]
    in_size *= 3

    # define slices
    slice_out = False
    slice_indices = None
    if is_tower_task:
        slice_out = True
        slice_indices = generator.indices_rings_ravel

    # instantiate MLP
    encoder = MLPEncoder(
        edges_signs=edges_signs,
        slice_out=slice_out,
        slice_indices=slice_indices,
        in_size=in_size,
        out_size=num_edges,
        width_size=hidden_layer_size,
        depth=hidden_layer_num,
        activation=get_activation_fn(activation_name),
        final_activation=get_activation_fn(final_activation_name),
        key=key
        )

    return encoder


# ===============================================================================
# Autoencoder models
# ===============================================================================

def build_neural_formfinder(mesh, key, params, generator):
    """
    """
    # Unpack hyperparams
    nn_params, fd_params = params

    # Create MLP encoder
    encoder = build_neural_encoder(mesh, key, nn_params, generator)

    # Build FD decoder
    decoder = build_fd_decoder(mesh, fd_params)

    # Assemble autoencoder
    model = AutoEncoder(encoder, decoder)

    return model


def build_neural_autoencoder(mesh, key, params, generator):
    """
    """
    # Unpack hyperparams
    enc_params, dec_params = params

    # Create MLP encoder
    encoder = build_neural_encoder(mesh, key, enc_params, generator)

    # Build MLP decoder
    decoder = build_neural_decoder(mesh, key, dec_params)

    # Assemble autoencoder
    model = AutoEncoder(encoder, decoder)

    return model


def build_neural_autoencoder_piggy(mesh, key, params, generator):
    """
    """
    # Unpack hyperparams

    enc_params, dec_params, fd_params = params

    # Create MLP encoder
    encoder = build_neural_encoder(mesh, key, enc_params, generator)

    # Build FD decoder
    decoder = build_fd_decoder(mesh, fd_params)

    # Build MLP decoder
    decoder_piggy = build_neural_decoder(mesh, key, dec_params)

    # Assemble autoencoder
    model = AutoEncoderPiggy(encoder, decoder, decoder_piggy)

    return model


def build_neural_model(name, config, generator, model_key):
    """
    """
    # generate base FD mesh
    mesh = build_mesh_from_generator(generator)

    # build model
    fd_params = config["fdm"]
    decoder_params = config["decoder"]
    encoder_params = config["encoder"]
    generator_params = config["generator"]

    encoder_params = (encoder_params, generator_params)

    # select model
    if name == "formfinder":
        build_fn = build_neural_formfinder
        params = (encoder_params, fd_params)
    elif name == "autoencoder":
        build_fn = build_neural_autoencoder
        params = (encoder_params, (decoder_params, fd_params))
    elif name == "piggy":
        build_fn = build_neural_autoencoder_piggy
        params = (encoder_params, (decoder_params, fd_params), fd_params)
    else:
        raise ValueError(f"Model name {name} is unsupported")

    return build_fn(mesh, model_key, params, generator)
