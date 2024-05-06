import os
import time
import yaml

import jax
from jax import vmap
import jax.random as jrn

from neural_fofin import DATA

from neural_fofin.training import train_model

from neural_fofin.plotting import plot_smoothed_losses

from neural_fofin.builders import build_loss_function
from neural_fofin.builders import build_data_generator
from neural_fofin.builders import build_connectivity_structure_from_generator
from neural_fofin.builders import build_neural_model
from neural_fofin.builders import build_optimizer

from neural_fofin.serialization import save_model


# ===============================================================================
# Script function
# ===============================================================================

def train(model, task, plot=True, save=True):
    """
    Train a model to approximate a family of arbitrary shapes with mechanically-feasible geometries.

    Parameters
    ___________
    model: `str`
        The model name.
        Supported models are formfinder, autoencoder, and piggy.
    task: `str`
        The name of the YAML config file with the task hyperparameters.
    plot: `bool`
        If `True`, plot the loss curves.
    save: `bool`
        If `True`, save the trained model and the loss histories.
    """
    MODEL_NAME = model
    SAVE = save
    PLOT = plot
    CONFIG_NAME = task

    # load yaml file with hyperparameters
    with open(f"{CONFIG_NAME}.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # train model
    train_data = train_model_from_config(model, config, callback=None)
    trained_model, trained_opt_states, loss_history = train_data

    # plot loss curves
    if PLOT:
        print("\nPlotting")
        loss_labels = ["loss", "shape", "residual", "smoothness"]
        plot_smoothed_losses(loss_history,
                             window_size=50,
                             labels=loss_labels)

    # save models
    if SAVE:
        print("\nSaving results")

        _filename = f"{MODEL_NAME}"
        loss_params = config["loss"]
        if loss_params["residual"]["include"] > 0 and MODEL_NAME != "formfinder":
            _filename += "_pinn"
        _filename += f"_{CONFIG_NAME}"

        _filepath = os.path.join(DATA, f"{_filename}.eqx")
        save_model(_filepath, trained_model)
        print(f"Saved model to {_filepath}")

        for i, _label in enumerate(loss_labels):
            _filename_loss = f"losses_{_filename}_{_label}.txt"

            _filepath = os.path.join(DATA, _filename_loss)
            with open(_filepath, "w") as file:
                for values in loss_history:
                    _value = values[i].item()
                    file.write(f"{_value}\n")

            print(f"Saved loss history to {_filepath}")


# ===============================================================================
# Helper functions
# ===============================================================================

def train_model_from_config(model, config, callback=None):
    """
    Train a model to approximate a family of arbitrary shapes with mechanically-feasible geometries.

    Parameters
    ___________
    model: `str`
        The model name.
        Supported models are formfinder, autoencoder, and piggy.
    config: `dict`
        A dictionary with the hyperparameters configuration.
    callback: `Callable`
        A callback function to call at every train step.
    """
    MODEL_NAME = model

    # unpack parameters
    seed = config["seed"]
    training_params = config["training"]
    batch_size = training_params["batch_size"]
    steps = training_params["steps"]
    generator_name = config['generator']['name']
    bounds_name = config['generator']['bounds']

    # randomness
    key = jrn.PRNGKey(seed)
    model_key, generator_key = jax.random.split(key, 2)

    # create experiment
    print(f"\nTraining {MODEL_NAME} on {generator_name} dataset with {bounds_name} bounds")
    generator = build_data_generator(config)
    structure = build_connectivity_structure_from_generator(generator)
    optimizer = build_optimizer(config)
    compute_loss = build_loss_function(config, generator)
    model = build_neural_model(MODEL_NAME, config, generator, model_key)

    # sample initial data batch
    xyz = vmap(generator)(jrn.split(generator_key, batch_size))

    # warmstart
    start_loss = compute_loss(model, structure, xyz)
    print(f"Structure size: {structure.num_vertices} vertices, {structure.num_edges} edges")
    print(f"{MODEL_NAME} start loss: {start_loss:.6f}")

    # train models
    print("\nTraining")
    start = time.perf_counter()
    train_data = train_model(
        model,
        structure,
        optimizer,
        generator,
        loss_fn=compute_loss,
        num_steps=steps,
        batch_size=batch_size,
        key=generator_key,
        callback=callback
        )
    end = time.perf_counter()

    print("\nTraining completed")
    print(f"Training time: {end - start:.4f} s")

    trained_model, trained_opt_states, loss_history = train_data

    end_loss = compute_loss(trained_model, structure, xyz)
    print(f"{MODEL_NAME} last loss: {end_loss}")

    return train_data


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(train)
