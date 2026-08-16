"""Single-output CRISPRon-BE adapter.

Source: https://github.com/RTH-tools/crispron-BE
"""

from types import SimpleNamespace

from ..config import CONTEXT_LENGTH, CRISPRON_PARAMS, DNA_BASES


def build_crispron(n_sources):
    import tensorflow as tf

    from ..models.crispron_be.DeepCRISPRonBE_train import get_pooling

    sequence = tf.keras.Input(shape=(CONTEXT_LENGTH, len(DNA_BASES)), name="sequence")
    off_score = tf.keras.Input(shape=(1,), name="crisproff_score")
    on_score = tf.keras.Input(shape=(1,), name="crispron_score")
    source = tf.keras.Input(shape=(n_sources,), name="source")

    branches = []
    for filters, kernel in zip(
        CRISPRON_PARAMS["conv_filters"],
        CRISPRON_PARAMS["conv_kernels"],
    ):
        branch = tf.keras.layers.Conv1D(filters, kernel, activation="relu")(sequence)
        branch = tf.keras.layers.Dropout(CRISPRON_PARAMS["dropout"])(branch)
        branch = get_pooling(
            SimpleNamespace(pt="avg", pfs=CRISPRON_PARAMS["pool_size"]),
            1,
        )(branch)
        branches.append(tf.keras.layers.Flatten()(branch))

    hidden = tf.keras.layers.Concatenate()(branches)
    hidden = tf.keras.layers.Dense(CRISPRON_PARAMS["hidden_dims"][0], activation="relu")(hidden)
    hidden = tf.keras.layers.Dropout(CRISPRON_PARAMS["dropout"])(hidden)
    hidden = tf.keras.layers.Concatenate()([hidden, off_score, on_score, source])
    for units in CRISPRON_PARAMS["hidden_dims"][1:]:
        hidden = tf.keras.layers.Dense(units, activation="relu")(hidden)
        hidden = tf.keras.layers.Dropout(CRISPRON_PARAMS["dropout"])(hidden)
    outputs = tf.keras.layers.Dense(1, name="efficiency")(hidden)
    model = tf.keras.Model(
        [sequence, off_score, on_score, source],
        outputs,
        name="CRISPRon_BE",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CRISPRON_PARAMS["learning_rate"]),
        loss="mse",
    )
    return model
