from jax import vmap

import jax.numpy as jnp

from neural_fofin.models import AutoEncoderPiggy

from neural_fofin.helpers import vertices_residuals_from_xyz


def compute_loss(
        model,
        structure,
        x,
        loss_fn,
        loss_params,
        aux_data=False,
        piggy_mode=False
):
    """
    Compute the model loss according to the model type.
    """
    predict_fn = vmap(model, in_axes=(0, None, None, None))
    x_hat, data_hat = predict_fn(x, structure, True, piggy_mode)

    # TODO: make _loss_fn an input, and make isinstance check before running this function
    _loss_fn = _compute_loss
    if isinstance(model, AutoEncoderPiggy):
        _loss_fn = _compute_loss_piggy

    return _loss_fn(loss_fn, loss_params, x, x_hat, data_hat, structure, aux_data, piggy_mode)


def _compute_loss(
        loss_fn,
        loss_params,
        x,
        x_hat,
        params_hat,
        structure,
        aux_data,
        piggy_mode=False
):
    """
    Compute the model loss of an autoencoder.
    """
    return loss_fn(
        x,
        x_hat,
        params_hat,
        structure,
        loss_params,
        aux_data
    )


def _compute_loss_piggy(loss_fn, loss_params, x, x_data_hat, y_data_hat, structure, aux_data, piggy_mode=True):
    """
    Compute the model loss of a piggy autoencoder.
    """
    x_hat, x_params_hat = x_data_hat

    if not piggy_mode:
        loss_data = loss_fn(
            x,
            x_hat,
            x_params_hat,
            structure,
            loss_params,
            aux_data
        )
    else:
        y_hat, y_params_hat = y_data_hat
        q, xyz_fixed, loads = y_params_hat

        loss_data = loss_fn(
            x_hat,
            y_hat,
            y_params_hat,
            structure,
            loss_params,
            aux_data
            )

    return loss_data


def compute_loss_residual_smoothness(x, x_hat, params_hat, structure, loss_params, aux_data, *args):
    """
    Compute the model loss.

    Parameters
    ----------
    x : target
    x_hat : prediction
    params_hat : parameters prediction (aux data)
    structure : the connectivity graph of the structure
    loss_params : the scaling parameters to combine the loss' error terms
    aux_data : if true, returns auxiliary data
    """
    loss_shape = jnp.array([0.0])

    # include support ring vertices to compute residual error on
    residual_params = loss_params["residual"]
    indices_rings = residual_params["indices"]
    indices = structure.indices_free
    indices = jnp.concatenate((indices, indices_rings))
    factor_residual = residual_params["weight"] / residual_params["scale"]
    loss_residual = compute_error_residual(x_hat, params_hat, structure, indices)
    loss_residual = factor_residual * loss_residual

    smooth_params = loss_params["energy"]
    factor_smooth = smooth_params["weight"] / smooth_params["scale"]
    loss_smooth = compute_error_smoothness(x_hat, params_hat, structure)
    loss_smooth = factor_smooth * loss_smooth

    loss = 0.0
    if residual_params["include"]:
        loss = loss + loss_residual
    if smooth_params["include"]:
        loss = loss + loss_smooth

    loss_terms = (
        loss,
        loss_shape,
        loss_residual,
        loss_smooth
    )

    if aux_data:
        return loss, loss_terms

    return loss


def compute_loss_shape_residual_smoothness(x, x_hat, params_hat, structure, loss_params, aux_data, *args):
    """
    Compute the model loss.

    Parameters
    ----------
    x : target
    x_hat : prediction
    params_hat : parameters prediction (aux data)
    structure : the connectivity graph of the structure
    loss_params : the scaling parameters to combine the loss' error terms
    aux_data : if true, returns auxiliary data
    """
    shape_params = loss_params["shape"]
    factor_shape = shape_params["weight"] / shape_params["scale"]

    # select only points on rings to compute shape error
    dims = shape_params["dims"]
    indices = shape_params["indices"]

    def slice_x(_x):
        return jnp.reshape(_x, dims)[indices, :].ravel()

    x_slice = vmap(slice_x)(x)
    x_hat_slice = vmap(slice_x)(x_hat)
    assert x_slice.shape == x_hat_slice.shape

    loss_shape = compute_error_shape_l2(x_slice, x_hat_slice)
    loss_shape = factor_shape * loss_shape

    indices = structure.indices_free
    residual_params = loss_params["residual"]
    factor_residual = residual_params["weight"] / residual_params["scale"]
    loss_residual = compute_error_residual(x_hat, params_hat, structure, indices)
    loss_residual = factor_residual * loss_residual

    smooth_params = loss_params["energy"]
    factor_smooth = smooth_params["weight"] / smooth_params["scale"]
    loss_smooth = compute_error_smoothness(x_hat, params_hat, structure)
    loss_smooth = factor_smooth * loss_smooth

    loss = 0.0
    if shape_params["include"]:
        loss = loss + loss_shape
    if residual_params["include"]:
        loss = loss + loss_residual
    if smooth_params["include"]:
        loss = loss + loss_smooth

    loss_terms = (
        loss,
        loss_shape,
        loss_residual,
        loss_smooth
    )

    if aux_data:
        return loss, loss_terms

    return loss


def compute_loss_shape_residual(x, x_hat, params_hat, structure, loss_params, aux_data, *args):
    """
    Compute the model loss.

    Parameters
    ----------
    x : target
    x_hat : prediction
    params_hat : parameters prediction (aux data)
    structure : the connectivity graph of the structure
    loss_params : the scaling parameters to combine the loss' error terms
    aux_data : if true, returns auxiliary data
    """
    shape_params = loss_params["shape"]
    factor_shape = shape_params["weight"] / shape_params["scale"]
    loss_shape = compute_error_shape_l1(x, x_hat)
    loss_shape = factor_shape * loss_shape

    indices = structure.indices_free
    residual_params = loss_params["residual"]
    factor_residual = residual_params["weight"] / residual_params["scale"]
    loss_residual = compute_error_residual(x_hat, params_hat, structure, indices)
    loss_residual = factor_residual * loss_residual

    loss = 0.0
    if shape_params["include"]:
        loss = loss + loss_shape
    if residual_params["include"]:
        loss = loss + loss_residual

    loss_terms = (
        loss,
        loss_shape,
        loss_residual
    )

    if aux_data:
        return loss, loss_terms

    return loss


def compute_error_shape_l1(x, x_hat):
    """
    Calculate the shape reconstruction error
    """
    error = jnp.abs(x - x_hat)
    batch_error = jnp.sum(error, axis=-1)

    return jnp.mean(batch_error, axis=-1)


def compute_error_shape_l2(x, x_hat):
    """
    Calculate the shape reconstruction error
    """
    error = jnp.square(x - x_hat)
    batch_error = jnp.sum(error, axis=-1)

    return jnp.mean(batch_error, axis=-1)


def compute_error_residual(x_hat, params_hat, structure, indices):
    """
    Calculate the residual error.
    """
    def calculate_residuals(_x_hat, _params_hat):
        """
        _x_hat: the shape predicted by the model
        _data_hat: the aux data produced by the model

        NOTE: Not using jnp.linalg.norm because we hitted NaNs.
        """
        q_hat, xyz_fixed, loads = _params_hat
        residual_vectors = vertices_residuals_from_xyz(
            q_hat,
            loads,
            _x_hat,
            structure
        )
        residual_vectors_free = jnp.ravel(residual_vectors[indices, :])

        # return jnp.linalg.norm(residual_vectors_free, axis=-1)
        # return jnp.sqrt(jnp.sum(jnp.square(residual_vectors_free), axis=-1))
        # return jnp.square(residual_vectors_free)
        return jnp.square(residual_vectors_free)

    error = vmap(calculate_residuals)(x_hat, params_hat)
    # batch_error = jnp.sum(error, axis=-1)
    batch_error = jnp.sqrt(jnp.sum(error, axis=-1))

    return jnp.mean(batch_error, axis=-1)


def compute_error_smoothness(x_hat, params_hat, structure):
    """
    Calculate the shape smoothness (fairness) error.
    """
    error = vmap(vertices_smoothness, in_axes=(0, None))(x_hat, structure)
    batch_error = jnp.sqrt(jnp.sum(error, axis=-1))
    # batch_error = jnp.sum(error, axis=-1)

    return jnp.mean(batch_error, axis=-1)


def vertices_smoothness(xyz, structure):
    """
    Compute the shape smoothness energy of the vertices of a structure.
    """
    xyz = jnp.reshape(xyz, (-1, 3))

    def vertex_fairness(xyz_vertex, adjacency_vertex):
        """
        Compute the fairness of an n-gon vertex neighborhood.
        """
        xyz_nbrs = adjacency_vertex @ xyz / jnp.sum(adjacency_vertex, axis=-1)

        fvector = xyz_vertex - xyz_nbrs
        assert fvector.shape == xyz_vertex.shape

        return jnp.sum(jnp.square(fvector))

    indices = structure.indices_free
    adjacency_free = structure.adjacency[indices, :]
    xyz_free = xyz[indices, :]

    vertices_fairness_fn = vmap(vertex_fairness)

    return vertices_fairness_fn(xyz_free, adjacency_free)


def print_loss_summary(loss_terms, labels=None, prefix=None):
    """
    """
    if not labels:
        labels = ["Loss", "Shape error", "Residual error", "Smooth error"]

    msg_parts = []
    if prefix:
        msg_parts.append(prefix)

    for term, label in zip(loss_terms, labels):
        part = f"{label}: {term.item():.4f}"
        msg_parts.append(part)

    msg = "\t".join(msg_parts)
    print(msg)
