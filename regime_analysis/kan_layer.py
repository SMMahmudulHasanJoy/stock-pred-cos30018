"""
kan_layer.py
A compact Kolmogorov-Arnold Network built on Keras/TensorFlow.

Why not pykan: pykan requires PyTorch, and the PyTorch wheel index is not
reachable from the environment these runs were executed in. This module
reimplements the same construction pykan uses so the comparison stays faithful:

    phi(x) = w_base * silu(x) + w_spline * sum_i c_i B_i(x)

with B_i a B-spline basis of order k on a uniform grid, one learnable univariate
function per (input, output) pair, and edges summed at each node. Defaults match
the published configuration: grid = 3 intervals, k = 3, so grid + k = 6 basis
functions per edge. Training uses full-batch L-BFGS for a fixed number of
iterations, mirroring pykan's fit(opt="LBFGS", steps=...).
"""

import numpy as np
import tensorflow as tf


def build_knots(grid_size, spline_order, lower=-1.0, upper=1.0):
    """Uniform extended knot vector covering [lower, upper]."""
    step = (upper - lower) / grid_size
    knots = np.arange(-spline_order, grid_size + spline_order + 1, dtype=np.float64)
    knots = knots * step + lower
    return tf.constant(knots, dtype=tf.float32)


def bspline_basis(x, knots, spline_order):
    """
    Cox-de Boor recursion.

    x     : (batch, in_dim)
    knots : (grid_size + 2 * spline_order + 1,)
    returns (batch, in_dim, grid_size + spline_order)
    """
    x = tf.expand_dims(x, -1)
    left = knots[:-1]
    right = knots[1:]
    bases = tf.cast((x >= left) & (x < right), tf.float32)

    for order in range(1, spline_order + 1):
        lower_knots = knots[: -(order + 1)]
        upper_knots = knots[order:-1]
        denom_left = tf.maximum(upper_knots - lower_knots, 1e-8)
        term_left = (x - lower_knots) / denom_left * bases[..., :-1]

        lower_shift = knots[1:-order]
        upper_shift = knots[order + 1 :]
        denom_right = tf.maximum(upper_shift - lower_shift, 1e-8)
        term_right = (upper_shift - x) / denom_right * bases[..., 1:]

        bases = term_left + term_right
    return bases


class KANLayer(tf.keras.layers.Layer):
    """One KAN layer: learnable univariate spline on every edge, summed at nodes."""

    def __init__(self, out_dim, grid_size=3, spline_order=3, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.seed = seed

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.in_dim = in_dim
        self.n_basis = self.grid_size + self.spline_order
        self.knots = build_knots(self.grid_size, self.spline_order)

        initialiser = tf.keras.initializers.GlorotUniform(seed=self.seed)
        scale = 1.0 / np.sqrt(in_dim)

        self.spline_coef = self.add_weight(
            name="spline_coef",
            shape=(in_dim, self.out_dim, self.n_basis),
            initializer=tf.keras.initializers.RandomNormal(
                stddev=0.1 * scale, seed=self.seed
            ),
            trainable=True,
        )
        self.spline_scale = self.add_weight(
            name="spline_scale",
            shape=(in_dim, self.out_dim),
            initializer=initialiser,
            trainable=True,
        )
        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(in_dim, self.out_dim),
            initializer=initialiser,
            trainable=True,
        )
        self.node_bias = self.add_weight(
            name="node_bias",
            shape=(self.out_dim,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # keep activations inside the spline domain
        x = tf.tanh(inputs)

        base = tf.nn.silu(x)
        base_out = tf.einsum("bi,io->bo", base, self.base_weight)

        basis = bspline_basis(x, self.knots, self.spline_order)
        spline = tf.einsum("bif,iof->bio", basis, self.spline_coef)
        spline_out = tf.einsum("bio,io->bo", spline, self.spline_scale)

        return base_out + spline_out + self.node_bias


def build_kan(input_dim, hidden_dim, output_dim, grid_size=3, spline_order=3, seed=0):
    """width = [input_dim, hidden_dim, output_dim], as in the published setup."""
    inputs = tf.keras.Input(shape=(input_dim,))
    hidden = KANLayer(hidden_dim, grid_size, spline_order, seed=seed)(inputs)
    outputs = KANLayer(output_dim, grid_size, spline_order, seed=seed + 1)(hidden)
    return tf.keras.Model(inputs, outputs)


def fit_lbfgs(model, x, y, max_iterations=10):
    """
    Full-batch L-BFGS, matching pykan's fit(opt="LBFGS", steps=max_iterations).
    Gradients come from TensorFlow; the optimiser itself is scipy's L-BFGS-B.
    """
    from scipy.optimize import minimize

    variables = model.trainable_variables
    shapes = [v.shape.as_list() for v in variables]
    sizes = [int(np.prod(s)) for s in shapes]

    def assign(flat):
        offset = 0
        for var, size, shape in zip(variables, sizes, shapes):
            var.assign(tf.reshape(tf.constant(flat[offset : offset + size], tf.float32), shape))
            offset += size

    x_tensor = tf.constant(x, tf.float32)
    y_tensor = tf.constant(y, tf.float32)

    def loss_and_grad(flat):
        assign(flat)
        with tf.GradientTape() as tape:
            prediction = model(x_tensor, training=True)
            loss = tf.reduce_mean(tf.square(prediction - y_tensor))
        grads = tape.gradient(loss, variables)
        flat_grad = np.concatenate(
            [
                (np.zeros(size, np.float64) if g is None else g.numpy().astype(np.float64).ravel())
                for g, size in zip(grads, sizes)
            ]
        )
        return float(loss.numpy()), flat_grad

    initial = np.concatenate([v.numpy().astype(np.float64).ravel() for v in variables])
    result = minimize(
        loss_and_grad,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iterations, "maxcor": 10},
    )
    assign(result.x)
    return {"final_loss": float(result.fun), "iterations": int(result.nit)}


# --------------------------------------------------------------------------
# Recurrent KAN (referee 1, major comment 2)
# --------------------------------------------------------------------------
class KANRNNCell(tf.keras.layers.Layer):
    """
    A KAN layer used as a recurrent cell, so the network sees the look-back
    window one day at a time and keeps temporal order, like the LSTM does.

        h_t = KAN([x_t, h_{t-1}])

    The cell concatenates the current day's features with the previous hidden
    state and passes them through one KANLayer (learnable spline on every
    edge). The hidden state is bounded by the tanh inside KANLayer. This is
    the simplest "recurrent wrapper" around a KAN; TKAN-style designs add
    LSTM-like gating on top of the same idea.
    """

    def __init__(self, units, grid_size=3, spline_order=3, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.output_size = units
        self.kan = KANLayer(units, grid_size, spline_order, seed=seed)

    def build(self, input_shape):
        self.kan.build((None, int(input_shape[-1]) + self.units))
        super().build(input_shape)

    def call(self, inputs, states):
        prev = states[0]
        h = tf.tanh(self.kan(tf.concat([inputs, prev], axis=-1)))
        return h, [h]


def build_recurrent_kan(lookback, n_features, horizon, units=20, grid_size=3,
                        spline_order=3, seed=0):
    """
    Sequence model: KANRNNCell over the window, then a KAN read-out to the
    H-step forecast. Trained with the same optimiser and budget as the LSTM
    (RMSprop, 25 epochs, batch 30) so the comparison with LSTM differs only
    in what sits inside the recurrent cell.
    """
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(lookback, n_features))
    h = tf.keras.layers.RNN(KANRNNCell(units, grid_size, spline_order, seed=seed))(inputs)
    outputs = KANLayer(horizon, grid_size, spline_order, seed=seed + 1)(h)
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss="mean_squared_error", optimizer="RMSprop")
    return model


def build_mlp(input_dim, horizon, width=32, seed=0):
    """
    Plain MLP on the same flattened (L * |F|) vector KAN receives: the
    like-for-like baseline referee 1 asked for. Trained with Adam and early
    stopping on the training loss, which is a competent (not tuned) baseline.
    """
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(width, activation="tanh"),
        tf.keras.layers.Dense(width, activation="tanh"),
        tf.keras.layers.Dense(horizon),
    ])
    model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(1e-3))
    return model
