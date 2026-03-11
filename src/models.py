from __future__ import annotations

from typing import Iterable, Sequence

import tensorflow as tf
import keras


def tcn_residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout: float,
    name: str,
) -> tf.Tensor:
    shortcut = x

    y = keras.layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        name=f"{name}_conv1",
    )(x)
    y = keras.layers.LayerNormalization(name=f"{name}_ln1")(y)
    y = keras.layers.Activation("swish", name=f"{name}_act1")(y)
    y = keras.layers.Dropout(dropout, name=f"{name}_drop1")(y)

    y = keras.layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        name=f"{name}_conv2",
    )(y)
    y = keras.layers.LayerNormalization(name=f"{name}_ln2")(y)
    y = keras.layers.Activation("swish", name=f"{name}_act2")(y)
    y = keras.layers.Dropout(dropout, name=f"{name}_drop2")(y)

    if shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv1D(filters, 1, padding="same", name=f"{name}_proj")(shortcut)

    return keras.layers.Add(name=f"{name}_add")([shortcut, y])


def build_tcn(
    input_shape: Sequence[int],
    num_classes: int,
    filters: int = 64,
    kernel_size: int = 3,
    dilations: Iterable[int] = (1, 2, 4, 8),
    dropout: float = 0.2,
) -> keras.Model:
    if num_classes <= 1:
        raise ValueError(f"num_classes debe ser > 1, recibido: {num_classes}")

    inp = keras.Input(shape=input_shape, name="input")
    x = inp
    for i, d in enumerate(dilations):
        x = tcn_residual_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=int(d),
            dropout=dropout,
            name=f"tcn_b{i}",
        )

    x = keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = keras.layers.Dense(filters, activation="swish", name="head_dense")(x)
    x = keras.layers.Dropout(dropout, name="head_drop")(x)
    out = keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)

    return keras.Model(inp, out, name="lse_tcn")


def model_size_params(model_size: str) -> dict:
    model_size = model_size.lower()
    if model_size == "small":
        return {"filters": 64, "kernel_size": 3, "dilations": (1, 2, 4, 8), "dropout": 0.2}
    if model_size == "medium":
        return {"filters": 96, "kernel_size": 3, "dilations": (1, 2, 4, 8, 16), "dropout": 0.25}
    raise ValueError(f"model_size no soportado: {model_size}")