import jax
import jax.numpy as jnp
from jax import lax


def init_conv(key, in_channels, out_channels, kernel_size=3):
    scale = jnp.sqrt(2.0 / (kernel_size * kernel_size * in_channels))
    w = scale * jax.random.normal(
        key,
        (kernel_size, kernel_size, in_channels, out_channels)
    )
    b = jnp.zeros(out_channels)

    return {"w": w, "b": b}


def conv(x, params, stride=1):
    y = lax.conv_general_dilated(
        x,
        params["w"],
        window_strides=(stride, stride),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )

    return y + params["b"]


def init_dense(key, in_dim, out_dim):
    scale = jnp.sqrt(2.0 / in_dim)
    w = scale * jax.random.normal(key, (in_dim, out_dim))
    b = jnp.zeros(out_dim)

    return {"w": w, "b": b}


def dense(x, params):
    return x @ params["w"] + params["b"]


def init_norm(channels):
    return {
        "gamma": jnp.ones(channels),
        "beta": jnp.zeros(channels)
    }


def group_norm(x, params, groups=8, eps=1e-5):
    b, h, w, c = x.shape

    x_grouped = x.reshape(b, h, w, groups, c // groups)

    mean = x_grouped.mean(axis=(1, 2, 4), keepdims=True)
    var = x_grouped.var(axis=(1, 2, 4), keepdims=True)

    x_grouped = (x_grouped - mean) / jnp.sqrt(var + eps)
    x = x_grouped.reshape(b, h, w, c)

    gamma = params["gamma"][None, None, None, :]
    beta = params["beta"][None, None, None, :]

    return x * gamma + beta


def init_time_embedding(key, time_dim):
    k1, k2 = jax.random.split(key)

    return {
        "dense1": init_dense(k1, 5, time_dim),
        "dense2": init_dense(k2, time_dim, time_dim)
    }


def time_embedding(t, params):
    t = t.reshape(t.shape[0], 1)

    emb = jnp.concatenate(
        [
            t,
            jnp.sin(2 * jnp.pi * t),
            jnp.cos(2 * jnp.pi * t),
            jnp.sin(4 * jnp.pi * t),
            jnp.cos(4 * jnp.pi * t)
        ],
        axis=-1
    )

    emb = dense(emb, params["dense1"])
    emb = jax.nn.silu(emb)
    emb = dense(emb, params["dense2"])

    return emb


def init_resblock(key, in_channels, out_channels, time_dim):
    k1, k2, k3, k4 = jax.random.split(key, 4)

    params = {
        "norm1": init_norm(in_channels),
        "conv1": init_conv(k1, in_channels, out_channels),
        "time": init_dense(k2, time_dim, out_channels),
        "norm2": init_norm(out_channels),
        "conv2": init_conv(k3, out_channels, out_channels),
        "skip": None
    }

    if in_channels != out_channels:
        params["skip"] = init_conv(k4, in_channels, out_channels, 1)

    return params


def resblock(x, t_emb, params):
    residual = x

    h = group_norm(x, params["norm1"])
    h = jax.nn.silu(h)
    h = conv(h, params["conv1"])

    time = dense(t_emb, params["time"])
    h = h + time[:, None, None, :]

    h = group_norm(h, params["norm2"])
    h = jax.nn.silu(h)
    h = conv(h, params["conv2"])

    if params["skip"] is not None:
        residual = conv(residual, params["skip"])

    return residual + h


def init_model(key, base_channels=64, time_dim=128):
    def next_key():
        nonlocal key
        key, subkey = jax.random.split(key)
        return subkey

    c = base_channels

    return {
        "time": init_time_embedding(next_key(), time_dim),

        "input": init_conv(next_key(), 3, c),

        "down1_1": init_resblock(next_key(), c, c, time_dim),
        "down1_2": init_resblock(next_key(), c, c, time_dim),

        "down1": init_conv(next_key(), c, 2 * c),

        "down2_1": init_resblock(next_key(), 2 * c, 2 * c, time_dim),
        "down2_2": init_resblock(next_key(), 2 * c, 2 * c, time_dim),

        "down2": init_conv(next_key(), 2 * c, 4 * c),

        "mid1": init_resblock(next_key(), 4 * c, 4 * c, time_dim),
        "mid2": init_resblock(next_key(), 4 * c, 4 * c, time_dim),

        "up1_conv": init_conv(next_key(), 6 * c, 2 * c),
        "up1_1": init_resblock(next_key(), 2 * c, 2 * c, time_dim),
        "up1_2": init_resblock(next_key(), 2 * c, 2 * c, time_dim),

        "up2_conv": init_conv(next_key(), 3 * c, c),
        "up2_1": init_resblock(next_key(), c, c, time_dim),
        "up2_2": init_resblock(next_key(), c, c, time_dim),

        "output_norm": init_norm(c),
        "output": init_conv(next_key(), c, 3)
    }


def model_apply(params, x, t, return_features=False):
    t_emb = time_embedding(t, params["time"])

    x = conv(x, params["input"])

    h1 = resblock(x, t_emb, params["down1_1"])
    h1 = resblock(h1, t_emb, params["down1_2"])

    x = conv(h1, params["down1"], stride=2)

    h2 = resblock(x, t_emb, params["down2_1"])
    h2 = resblock(h2, t_emb, params["down2_2"])

    x = conv(h2, params["down2"], stride=2)

    x = resblock(x, t_emb, params["mid1"])
    x = resblock(x, t_emb, params["mid2"])

    if return_features:
        return x

    x = jax.image.resize(
        x,
        (x.shape[0], 16, 16, x.shape[-1]),
        method="nearest"
    )

    x = jnp.concatenate([x, h2], axis=-1)
    x = conv(x, params["up1_conv"])

    x = resblock(x, t_emb, params["up1_1"])
    x = resblock(x, t_emb, params["up1_2"])

    x = jax.image.resize(
        x,
        (x.shape[0], 32, 32, x.shape[-1]),
        method="nearest"
    )

    x = jnp.concatenate([x, h1], axis=-1)
    x = conv(x, params["up2_conv"])

    x = resblock(x, t_emb, params["up2_1"])
    x = resblock(x, t_emb, params["up2_2"])

    x = group_norm(x, params["output_norm"])
    x = jax.nn.silu(x)

    return conv(x, params["output"])
