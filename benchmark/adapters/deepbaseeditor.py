"""TensorFlow adapter for the original DeepBaseEditor architecture.

Source: https://github.com/MyungjaeSong/Paired-Library/tree/DeepCRISPR.info/DeepBaseEditor
"""

from types import SimpleNamespace

from ..config import DNA_BASES, get_deepbase_params


def build_deepbaseeditor(dataset):
    import tensorflow as tf

    params = get_deepbase_params(dataset)
    length = params["input_length"]
    filters = params["conv_filters"]
    hidden_dims = params["hidden_dims"]
    inputs = tf.compat.v1.placeholder(
        tf.float32,
        [None, 1, length, len(DNA_BASES)],
    )
    targets = tf.compat.v1.placeholder(tf.float32, [None, 1])
    is_training = tf.compat.v1.placeholder(tf.bool)

    weights = tf.Variable(
        tf.random.truncated_normal(
            [1, params["kernel_size"], len(DNA_BASES), filters],
            stddev=0.03,
        ),
        name="conv1_W",
    )
    bias = tf.Variable(
        tf.random.truncated_normal([filters]),
        name="conv1_b",
    )
    hidden = tf.nn.relu(
        tf.nn.conv2d(inputs, weights, [1, 1, 1, 1], padding="VALID") + bias
    )
    hidden = tf.cond(
        is_training,
        lambda: tf.nn.dropout(hidden, rate=params["dropout"]),
        lambda: hidden,
    )
    flattened_dim = (
        length - params["kernel_size"] + 1
    ) * filters
    hidden = tf.reshape(hidden, [-1, flattened_dim])
    with tf.compat.v1.variable_scope("Fully_Connected_Layer1"):
        weights = tf.compat.v1.get_variable(
            "W_fcl1",
            shape=[flattened_dim, hidden_dims[0]],
        )
        bias = tf.compat.v1.get_variable(
            "B_fcl1",
            shape=[hidden_dims[0]],
        )
        hidden = tf.nn.relu(tf.nn.bias_add(tf.matmul(hidden, weights), bias))
        hidden = tf.cond(
            is_training,
            lambda: tf.nn.dropout(hidden, rate=params["dropout"]),
            lambda: hidden,
        )
    if len(hidden_dims) == 2:
        with tf.compat.v1.variable_scope("Fully_Connected_Layer2"):
            weights = tf.compat.v1.get_variable(
                "W_fcl2",
                shape=hidden_dims,
            )
            bias = tf.compat.v1.get_variable(
                "B_fcl2",
                shape=[hidden_dims[1]],
            )
            hidden = tf.nn.relu(tf.nn.bias_add(tf.matmul(hidden, weights), bias))
            hidden = tf.cond(
                is_training,
                lambda: tf.nn.dropout(hidden, rate=params["dropout"]),
                lambda: hidden,
            )
    with tf.compat.v1.variable_scope("Output_Layer"):
        output_dim = hidden_dims[-1]
        weights = tf.compat.v1.get_variable("W_out", shape=[output_dim, 1])
        bias = tf.compat.v1.get_variable("B_out", shape=[1])
        outputs = tf.nn.bias_add(tf.matmul(hidden, weights), bias)

    loss = tf.reduce_mean(tf.square(targets - outputs))
    optimizer = tf.compat.v1.train.AdamOptimizer(params["learning_rate"]).minimize(loss)
    return SimpleNamespace(
        inputs=inputs,
        targets=targets,
        is_training=is_training,
        outputs=outputs,
        loss=loss,
        optimizer=optimizer,
    )
