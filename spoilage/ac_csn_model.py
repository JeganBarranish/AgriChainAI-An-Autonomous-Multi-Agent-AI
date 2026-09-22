"""
AC-CSN model builders — product-specific architectures.

[SUPERVISED neural networks]

Paneer: full temporal dual-branch AC-CSN (existing design).
Dried fruit: compact dual-branch dense network (same conceptual fusion,
            sized for n≈24 dried storage cells).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_paneer_ac_csn(n_env: int, n_ops: int, seq_len: int, n_features: int) -> tf.keras.Model:
    """Existing Paneer AC-CSN architecture (preserved)."""
    seq_input = layers.Input(shape=(seq_len, n_features), name="sequence_input")
    env_input = layers.Input(shape=(n_env,), name="environmental_input")
    ops_input = layers.Input(shape=(n_ops,), name="operational_input")

    t = layers.GRU(12, name="temporal_encoder")(seq_input)
    t = layers.Dropout(0.3)(t)
    e = layers.Dense(8, activation="relu", name="environmental_repr")(env_input)
    o = layers.Dense(6, activation="relu", name="operational_repr")(ops_input)
    fused = layers.Concatenate(name="feature_fusion")([t, e, o])
    x = layers.Dense(16, activation="relu")(fused)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(3, activation="softmax", name="spoilage_class")(x)

    model = models.Model(
        inputs=[seq_input, env_input, ops_input],
        outputs=output,
        name="Paneer_AC_CSN",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_dryfruit_ac_csn(n_common: int, n_product: int) -> tf.keras.Model:
    """
    Dried-fruit AC-CSN-inspired compact model.

    Common branch: storage duration + packaging one-hot
    Product branch: fruit + dryer type one-hot
    Fusion → dense → softmax(3)
    """
    common_in = layers.Input(shape=(n_common,), name="common_environment_input")
    product_in = layers.Input(shape=(n_product,), name="product_specific_input")

    c = layers.Dense(8, activation="relu", name="common_encoder")(common_in)
    p = layers.Dense(8, activation="relu", name="product_encoder")(product_in)
    fused = layers.Concatenate(name="fusion")([c, p])
    x = layers.Dense(12, activation="relu")(fused)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(8, activation="relu")(x)
    output = layers.Dense(3, activation="softmax", name="spoilage_class")(x)

    model = models.Model(inputs=[common_in, product_in], outputs=output, name="DryFruit_AC_CSN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
