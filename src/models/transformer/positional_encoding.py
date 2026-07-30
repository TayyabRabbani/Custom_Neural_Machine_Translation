# src/models/transformer/positional_encoding.py

import numpy as np
import tensorflow as tf


def positional_encoding(length, d_model):
    half = d_model // 2
    positions = np.arange(length).reshape(length, 1)
    depths = np.arange(half).reshape(1,half)/ half
    angle = positions * (1.0 / (10000.0 ** depths))
    pos = np.concatenate([np.sin(angle), np.cos(angle)], axis=-1)
    return tf.cast(pos[np.newaxis, ...], tf.float32)


if __name__ == "__main__":
    print("Positional encoding shape:", positional_encoding(40, 256).shape)
