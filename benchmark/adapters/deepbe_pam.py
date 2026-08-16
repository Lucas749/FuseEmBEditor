"""DeepBE-PAM adapter.

Source: https://github.com/NahyeKim/DeepBE
"""

import json

from ..config import (
    CONTEXT_LENGTH,
    DEEPBE_ARCH_MODEL,
    DEEPBE_ARCH_PATH,
    DEEPBE_PARAMS,
    DNA_BASES,
)


def build_deepbe():
    import tensorflow as tf

    architecture = json.loads(DEEPBE_ARCH_PATH.read_text())[DEEPBE_ARCH_MODEL]
    conv_layer = next(layer for layer in architecture["layers"] if layer["type"] == "Conv1D")
    pool_layer = next(
        layer for layer in architecture["layers"]
        if layer["type"] == "AveragePooling1D"
    )
    dense_layers = [
        layer for layer in architecture["layers"]
        if layer["type"] == "Dense"
    ]
    inputs = tf.keras.Input(shape=(CONTEXT_LENGTH, len(DNA_BASES)), name="sequence")
    hidden = tf.keras.layers.Conv1D(
        conv_layer["filters"],
        conv_layer["kernel_size"],
        padding=conv_layer["padding"],
        activation=conv_layer["activation"],
    )(inputs)
    hidden = tf.keras.layers.AveragePooling1D(
        pool_size=pool_layer["pool_size"],
        strides=pool_layer["strides"],
    )(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    for layer in dense_layers[:-1]:
        hidden = tf.keras.layers.Dense(layer["units"], activation=layer["activation"])(hidden)
        hidden = tf.keras.layers.Dropout(DEEPBE_PARAMS["dropout"])(hidden)
    outputs = tf.keras.layers.Dense(
        dense_layers[-1]["units"],
        activation=dense_layers[-1]["activation"],
        name="efficiency",
    )(hidden)
    model = tf.keras.Model(inputs, outputs, name="DeepBE_PAM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(DEEPBE_PARAMS["learning_rate"]),
        loss="mae",
    )
    return model
