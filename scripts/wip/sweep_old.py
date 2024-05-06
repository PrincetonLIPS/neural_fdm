import numpy as np

import wandb
import yaml

import jax
from jax import vmap

import jax.random as jrn
import jax.numpy as jnp

import equinox as eqx

import optax

import matplotlib.pyplot as plt
import seaborn as sns

from compas.geometry import Translation
from compas.datastructures import Network

from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel

from jax_fdm.equilibrium import datastructure_updated
from jax_fdm.equilibrium import fdm

from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization import NotebookViewer as Viewer

from models import AutoEncoder
from models import ForceDensityModel
from models import ForceDensityWithShapeBasedLoads

from generator import create_mesh_from_grid
from generator import PointGrid
from generator import BezierSurfacePointGenerator

from train import train_model
from train import compute_loss


# global constants
# NOTE: do no using globals at home!
GRID_SIZE = 10.0
GRID_NUM_PTS = 4
GRID_INDICES = [15, 13, 5, 7, 14, 12, 4, 6, 10, 8, 0, 2, 11, 9, 1, 3]


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


def create_generator_minmax_values(name):
    experiments = {
        "pillow": pillow_minmax_values,
        "dome": dome_minmax_values,
        "saddle": dome_minmax_values
    }

    values_fn = experiments.get(name)
    if not values_fn:
        raise KeyError(f"Experiment name: {name} is currently unsupported!")
    
    return values_fn


def get_activation_fn(name):
    functions = {
        "elu": jax.nn.elu,
        "relu": jax.nn.relu,
        "softplus": jax.nn.softplus
    }

    activation_fn = functions.get(name)
    if not activation_fn:
        raise KeyError(f"Activation name: {name} is currently unsupported!")
    
    return activation_fn
    

def get_optimizer_fn(name):
    optimizers = {
        "adam": optax.adam,
        "sgd": optax.sgd,        
    }

    optimizer_fn = optimizers.get(name)
    if not optimizer_fn:
        raise KeyError(f"Optimize name: {name} is currently unsupported!")
    
    return optimizer_fn
    

def get_fd_solver_fn(name):
    solvers = {
        "constant": ForceDensityModel,
        "shape_based_loads": ForceDensityWithShapeBasedLoads,
    }

    solver_fn = solvers.get(name)
    if not solver_fn:
        raise KeyError(f"FD model name: {name} is currently unsupported!")
    
    return solver_fn
    

def create_data_generator(experiment_name, grid, u, v):
    # wiggle bounds for task
    minmax_fn = create_generator_minmax_values(experiment_name)
    minval, maxval = minmax_fn()

    # array-ify
    minval = jnp.array(minval)
    maxval = jnp.array(maxval)

    # Create data generator
    generator = BezierSurfacePointGenerator(grid, u, v, minval, maxval)

    return generator


def calculate_edges_mask(mesh):
    mask_edges = []
    for edge in mesh.edges():
        mask_val = 1.0
        if mesh.is_edge_fully_supported(edge):
            mask_val = 0.0
        mask_edges.append(mask_val)

    return jnp.array(mask_edges)


def log_to_wandb(model, opt_state, loss_val):
    wandb.log({"train_loss_val": loss_val.item()})


def moving_average(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def plot_smoothed_loss(loss_values, window_size):
    # Calculate the moving average
    smooth_loss = moving_average(loss_values, window_size)

    # Adjust the length of original loss values to match the smoothed array
    adjusted_loss_values = loss_values[:len(smooth_loss)]

    # Plotting
    plt.figure(figsize=(10, 6))
    color = "tab:blue"
    plt.plot(adjusted_loss_values, alpha=0.5, color=color, label='Loss')
    plt.plot(smooth_loss, color=color)
    plt.title('Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    # plt.ylim(1e-1, 1000)
    plt.legend()
    plt.show()


def sweep():
    """
    Sweep-ey, deep-ey.
    """
    # # Generator parameters
    # experiment_name = "pillow"
    # num_u = 7
    # num_v = 7

    # # FDM parameters
    # fd_name = "shape_based_loads"
    # pz = -0.5

    # # MLP parameters
    # hidden_layer_size = 128
    # hidden_layer_num = 3
    # activation_name = "elu"
    # final_activation_name = "softplus"
    
    # # optimizer parameters
    # optimizer_name = "adam"
    # learning_rate = 1e-6

    # # training parameters
    # seed = 91
    # batch_size = 64
    # steps = 20000

    # # Define callback for logging
    # callback = None

    # wandb
    with open("./config_sweep.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    wandb.init(config=config)
        
    # Generator parameters
    experiment_name = wandb.config.experiment_name    
    num_u = wandb.config.num_uv
    num_v = wandb.config.num_uv

    # FDM parameters
    fd_name = wandb.config.fd_name
    pz = wandb.config.pz

    # MLP parameters
    hidden_layer_size = wandb.config.hidden_layer_size
    hidden_layer_num = wandb.config.hidden_layer_num
    activation_name = wandb.config.activation_fn
    final_activation_name = wandb.config.final_activation_fn
    
    # optimizer parameters
    optimizer_name = wandb.config.optimizer
    learning_rate = wandb.config.learning_rate

    # training parameters
    batch_size = wandb.config.batch_size
    steps = wandb.config.steps
    seed = wandb.config.seed

    # Define callback for logging
    callback = log_to_wandb

    # randomness
    key = jrn.PRNGKey(seed)
    model_key, generator_key = jax.random.split(key, 2)

    # create grid of control points
    grid = PointGrid(GRID_SIZE, GRID_NUM_PTS, GRID_INDICES)

    # evaluation coordinates on bezier surface
    u = jnp.linspace(0.0, 1.0, num_u)
    v = jnp.linspace(0.0, 1.0, num_v)

    # Create data generator
    generator = create_data_generator(experiment_name, grid, u, v)

    # generate base FD mesh
    load = [0.0, 0.0, pz]
    mesh = create_mesh_from_grid(grid, u, v, load)

    # get mask of supported edges
    mask_edges = calculate_edges_mask(mesh)

    # query initial parameters from FD Mesh
    structure = EquilibriumMeshStructure.from_mesh(mesh)

    # Create FDM model
    fdm_model = EquilibriumModel(
        tmax=1,
        eta=1e-6,
        is_load_local=False,
        itersolve_fn=None,
        implicit_diff=True,
        verbose=False
        )

    # Get loads
    if fd_name == "constant":
        fdm_params = EquilibriumParametersState.from_datastructure(mesh) 
        loads = fdm_params.loads.nodes
    elif fd_name == "shape_based_loads":
        loads = jnp.array([load for _ in range(mesh.number_of_faces())])
    else:
        raise ValueError(f"FD solver name {fd_name} is unavailable")
    
    # Build FD decoder
    fd_solver = get_fd_solver_fn(fd_name)
    decoder = fd_solver(
        fdm_model,
        loads,                            
        mask_edges
        )

    # Create MLP encoder
    num_vertices = mesh.number_of_vertices()
    num_edges = mesh.number_of_edges()

    encoder = eqx.nn.MLP(
        in_size=num_vertices * 3,
        out_size=num_edges,
        width_size=hidden_layer_size,
        depth=hidden_layer_num,
        activation=get_activation_fn(activation_name),
        final_activation=get_activation_fn(final_activation_name),  # NOTE: needs softplus to ensure positive encoder output
        key=model_key    
        )

    # Assemble autoencoder
    model = AutoEncoder(encoder, decoder)

    # Instantiate optimizer
    optimizer_fn = get_optimizer_fn(optimizer_name)
    optimizer = optimizer_fn(learning_rate=learning_rate)

    # Warm start
    print("Warmstarting")
    xyz = vmap(generator)(jrn.split(generator_key, batch_size))
    start_loss = compute_loss(model, structure, xyz)
    print(f"Start loss: {start_loss:.6f}")

    # train model
    print("Training")
    train_data = train_model(
        model,
        structure,
        optimizer,
        generator,
        num_steps=steps,
        batch_size=batch_size,
        key=generator_key,
        callback=callback
        )

    trained_model, trained_opt_state, loss_vals = train_data
    print(f"Last loss: {loss_vals[-1]:.6f}")
    
    # plot_smoothed_loss(loss_vals, 100)

if __name__ == "__main__":
    # rock and roll
    sweep()