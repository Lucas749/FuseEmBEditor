"""Direct architecture loaders used by the paper benchmarks."""

from .config import (
    BEDICT_BASES,
    BEDICT_INPUT_LENGTH,
    BEDICT_PARAMS,
    IGRNA_ARCH_PATH,
    IGRNA_KERAS_INITIALIZERS,
    IGRNA_KERAS_LAYERS,
    IGRNA_PARAMS,
)
from .models.bedict_v2.CNN import PredictionCNN


def build_bedict():
    return PredictionCNN(
        k=BEDICT_PARAMS["kernel_size"],
        input_dim=BEDICT_INPUT_LENGTH,
        dictionary=len(BEDICT_BASES),
        mlp_embedder_config=None,
    )


def build_igrna():
    import tensorflow as tf

    custom_objects = {
        name: getattr(tf.keras.layers, name)
        for name in IGRNA_KERAS_LAYERS
    }
    custom_objects.update({
        name: getattr(tf.keras.initializers, name)
        for name in IGRNA_KERAS_INITIALIZERS
    })
    custom_objects["Sequential"] = tf.keras.Sequential
    model = tf.keras.models.model_from_json(
        IGRNA_ARCH_PATH.read_text(),
        custom_objects=custom_objects,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(IGRNA_PARAMS["learning_rate"]),
        loss="mse",
    )
    return model
