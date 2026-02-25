import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import (
    Input,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    concatenate,
)


def weighted_crossentropy(y_true, y_pred):
    class_weights = tf.constant([[[[0.6, 1.0, 5.0]]]], dtype=tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true, logits=y_pred
    )
    weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    weighted_losses = weights * unweighted_losses
    return tf.reduce_mean(weighted_losses)


def build_unet(patch_size, num_inputs=1):
    tf.keras.backend.clear_session()

    if num_inputs < 1:
        raise ValueError("num_inputs must be >= 1")

    if num_inputs == 1:
        inputs = [Input((patch_size, patch_size, 3), name="input")]
        x = inputs[0]
    else:
        inputs = [
            Input((patch_size, patch_size, 3), name=f"input_{i + 1}")
            for i in range(num_inputs)
        ]
        x = concatenate(inputs)

    conv1 = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    conv1 = Conv2D(16, (3, 3), activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(128, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(64, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(32, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(conv8)
    up9 = concatenate([up9, conv1])
    conv9 = Conv2D(16, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(16, (3, 3), activation="relu", padding="same")(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(3, (1, 1), activation="softmax", dtype="float32")(conv9)
    return Model(inputs=inputs, outputs=conv10)


def initialize_from_checkpoint(checkpoint_path, patch_size, num_inputs=7):
    source_model = tf.keras.models.load_model(
        checkpoint_path, custom_objects={"weighted_crossentropy": weighted_crossentropy}
    )
    target_model = build_unet(patch_size, num_inputs=num_inputs)
    _transfer_weights(source_model, target_model, num_inputs=num_inputs)
    return target_model


def _transfer_weights(source_model, target_model, num_inputs):
    source_layers = [layer for layer in source_model.layers if layer.weights]
    target_layers = [layer for layer in target_model.layers if layer.weights]

    if len(source_layers) != len(target_layers):
        raise ValueError(
            "Layer count mismatch between source and target models: "
            f"{len(source_layers)} vs {len(target_layers)}"
        )

    for layer_idx, (src, tgt) in enumerate(zip(source_layers, target_layers)):
        src_weights = src.get_weights()
        if not src_weights:
            continue

        if layer_idx == 0 and isinstance(tgt, Conv2D):
            kernel, bias = src_weights
            if kernel.shape[2] * num_inputs != tgt.get_weights()[0].shape[2]:
                raise ValueError(
                    "First conv input channels mismatch when expanding weights."
                )
            expanded_kernel = np.concatenate([kernel] * num_inputs, axis=2)
            tgt.set_weights([expanded_kernel, bias])
            continue

        tgt_weights = tgt.get_weights()
        if len(src_weights) != len(tgt_weights):
            raise ValueError(
                f"Weight count mismatch at layer {layer_idx}: "
                f"{len(src_weights)} vs {len(tgt_weights)}"
            )
        for src_w, tgt_w in zip(src_weights, tgt_weights):
            if src_w.shape != tgt_w.shape:
                raise ValueError(
                    f"Weight shape mismatch at layer {layer_idx}: "
                    f"{src_w.shape} vs {tgt_w.shape}"
                )
        tgt.set_weights(src_weights)
