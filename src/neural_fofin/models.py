import jax.numpy as jnp

import equinox as eqx

from jax.lax import stop_gradient

from jaxtyping import Array, Float, Bool

from jax_fdm.equilibrium import EquilibriumModel

from neural_fofin.helpers import calculate_area_loads
from neural_fofin.helpers import calculate_equilibrium_state
from neural_fofin.helpers import calculate_fd_params_state


# ===============================================================================
# Autoencoders
# ===============================================================================

class AutoEncoder(eqx.Module):
    """
    An autoencoder that couples a neural network with a form-finding solver.
    """
    encoder: eqx.Module
    decoder: eqx.Module

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, x, structure, aux_data=False, *args, **kwargs):
        """
        Predict a funicular shape given a target shape for a structure.
        """
        # NOTE: x must be a flat vector
        q = self.encoder(x)
        x_hat = self.decoder(q, x, structure, aux_data)

        return x_hat

    def encode(self, x):
        """
        Generate the latent representation of an input vector.
        """
        return self.encoder(x)

    def decode(self, q, *args, **kwargs):
        """
        Map a latent representation back to input space.
        """
        return self.decoder(q, *args, **kwargs)

    def predict_states(self, x, structure):
        """
        Predict equilibrium and parameter states for visualization.
        """
        # Predict shape
        x_hat, params = self(x, structure, True)

        return build_states(x_hat, params, structure)


class AutoEncoderPiggy(AutoEncoder):
    """
    An autoencoder with a piggybacking decoder.
    """
    decoder_piggy: eqx.Module

    def __init__(self, encoder, decoder, decoder_piggy):
        super().__init__(encoder, decoder)
        self.decoder_piggy = decoder_piggy

    def __call__(self, x, structure, aux_data=False, piggy_mode=True):
        """
        Make prediction.
        """
        q = self.encoder(x)
        x_hat = self.decoder(q, x, structure, aux_data)

        if piggy_mode:
            q = stop_gradient(q)
            x_hat = stop_gradient(x_hat)

        y_hat = self.decoder_piggy(q, x, structure, aux_data)

        return x_hat, y_hat

    def decode(self, q, *args, **kwargs):
        """
        Map a latent representation back to input space.
        """
        return self.decoder_piggy(q, *args, **kwargs)

    def predict_states(self, x, structure):
        """
        To interface with JAX FDM visualization.
        """
        # Predict shape
        _, pred_piggy = self(x, structure, True)
        x_hat, params = pred_piggy

        return build_states(x_hat, params, structure)


# ===============================================================================
# Encoders
# ===============================================================================

class Encoder(eqx.Module):
    """
    An encoder.
    """
    edges_signs: Array
    # If we want to learn a mapping w.r.t. a slice of the points output by the generator
    slice_out: Bool
    slice_indices: Array

    def __init__(
            self,
            edges_signs,
            slice_out=False,
            slice_indices=None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.edges_signs = edges_signs
        self.slice_out = slice_out
        self.slice_indices = slice_indices

    def __call__(self, x):
        """
        Map x to q.
        """
        if self.slice_out:
            x = jnp.reshape(x, (-1, 3))
            x = x[self.slice_indices, :]
            x = jnp.ravel(x)

        return super().__call__(x)


class MLPEncoder(Encoder, eqx.nn.MLP):
    """
    A MLP encoder.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x):
        """
        Transform a shape from Euclidean to latent representation.
        """
        # MLP prediction (must be positive due to softplus activation)
        q_hat = super().__call__(x)

        # NOTE: negative q denotes compression, positive tension.
        return q_hat * self.edges_signs


# ===============================================================================
# Decoders
# ===============================================================================

class Decoder(eqx.Module):
    """
    A decoder.
    """
    load: Float
    mask_edges: Array

    def __init__(self, load, mask_edges, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.load = load
        self.mask_edges = mask_edges

    def __call__(self, q, x, structure, aux_data=False):
        """
        """
        # gather parameters
        q = self.get_q(q)
        xyz_fixed = self.get_xyz_fixed(x, structure)
        loads = self.get_loads(x, structure)

        # predict x
        x_hat = self.get_xyz((q, xyz_fixed, loads), structure)

        if aux_data:
            data = (q, xyz_fixed, loads)
            return x_hat, data

        return x_hat

    def get_q(self, q_hat):
        """
        TODO: A better model should not be first predicting and then masking edges.
        """
        return q_hat * self.mask_edges

    def get_xyz_fixed(self, x, structure):
        """
        """
        indices = structure.indices_fixed
        x = jnp.reshape(x, (-1, 3))

        return x[indices, :]

    def get_loads(self, x, structure):
        """
        Calculate applied vertex loads.
        """
        if self.load:
            return calculate_area_loads(x, structure, self.load)

        return jnp.zeros((structure.num_vertices, 3))

    def get_xyz(self, params, structure):
        """
        """
        raise NotImplementedError


# ===============================================================================
# Physics-based decoders
# ===============================================================================

class FDDecoder(Decoder):
    """
    A fixed force density decoder.
    """
    model: EquilibriumModel

    def __init__(self, model, *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def get_xyz(self, params, structure):
        """
        """
        q, xyz_fixed, loads = params
        # NOTE: to predict only free vertices, use instead
        # self.model.nodes_free_positions(q, xyz_fixed, loads_nodes, structure)
        x_hat = self.model.equilibrium(q,
                                       xyz_fixed,
                                       loads,
                                       structure)

        return jnp.ravel(x_hat)


class FDDecoderParametrized(FDDecoder):
    """
    A parametrized force density decoder.
    """
    q: Array

    def __init__(self, q, *args, **kwargs):
        self.q = q
        super().__init__(*args, **kwargs)

    def __call__(self, x, structure, aux_data=False, *args, **kwargs):
        """
        """
        return super().__call__(self.q, x, structure, aux_data)

    def predict_states(self, x, structure):
        """
        Predict equilibrium and parameter states for visualization.
        """
        # Predict shape
        x_hat, params = self(x, structure, True)

        return build_states(x_hat, params, structure)


# ===============================================================================
# Neural decoders
# ===============================================================================

class MLPDecoder(Decoder, eqx.nn.MLP):
    """
    A MLP decoder maps q to xyz.
    NOTE: Should the inheritance order be reversed?
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_xyz(self, params, structure):
        """
        """
        # unpack parameters
        q, x_fixed, lodas = params

        # predict x
        x_free = self._get_xyz(params)

        # Concatenate the position of the free and the fixed nodes
        indices = structure.indices_freefixed
        x_free = jnp.reshape(x_free, (-1, 3))
        x_hat = jnp.concatenate((x_free, x_fixed))[indices, :]

        return jnp.ravel(x_hat)

    def _get_xyz(self, params):
        """
        """
        # unpack parameters
        q, x_fixed, loads = params

        # NOTE: using this exotic way to call __call__ to map q to x
        # due to multiple inheritance
        return eqx.nn.MLP.__call__(self, q)


class MLPDecoderXL(MLPDecoder):
    """
    A MLP decoder that maps q, xyz_fixed, and loads to xyz.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_xyz(self, params):
        """
        """
        # unpack parameters
        q, x_fixed, loads = params

        # concatenate long array
        x_fixed = jnp.ravel(x_fixed)
        loads_z = loads[:, 2]  # only z component, x and y are always 0
        params = jnp.concatenate((q, x_fixed, loads_z))

        return eqx.nn.MLP.__call__(self, params)


# ===============================================================================
# Helpers
# ===============================================================================

def build_states(xyz_hat, params, structure):
    """
    Assemble equilibrium and parameter states for visualization.
    """
    # Unpack aux data
    q, xyz_fixed, loads = params

    # Equilibrium parameters
    fd_params_state = calculate_fd_params_state(
        q,
        xyz_fixed,
        loads
    )

    # Equilibrium state
    xyz_hat = jnp.reshape(xyz_hat, (-1, 3))

    eq_state = calculate_equilibrium_state(
        q,
        xyz_hat,  # xyz_free | xyz_fixed
        loads,
        structure
    )

    return eq_state, fd_params_state
