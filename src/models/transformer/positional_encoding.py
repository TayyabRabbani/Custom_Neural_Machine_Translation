# src/models/transformer/positional_encoding.py

import numpy as np
import tensorflow as tf


class PositionalEncoding(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        max_length,
        d_model
    ):
        super().__init__()
        self.supports_masking = True

        self.pos_encoding = (
            self.create_positional_encoding(
                max_length,
                d_model
            )
        )

    def get_angles(
        self,
        position,
        index,
        d_model
    ):

        angle_rates = (
            1 /
            np.power(
                10000,
                (
                    2 * (index // 2)
                ) / np.float32(
                    d_model
                )
            )
        )

        return position * angle_rates

    def create_positional_encoding(
        self,
        position,
        d_model
    ):

        angle_rads = (
            self.get_angles(
                np.arange(position)[
                    :, np.newaxis
                ],

                np.arange(d_model)[
                    np.newaxis, :
                ],

                d_model
            )
        )

        angle_rads[:, 0::2] = np.sin(
            angle_rads[:, 0::2]
        )

        angle_rads[:, 1::2] = np.cos(
            angle_rads[:, 1::2]
        )

        pos_encoding = (
            angle_rads[
                np.newaxis, ...
            ]
        )

        return tf.cast(
            pos_encoding,
            dtype=tf.float32
        )

    def call(self, x):

        seq_length = tf.shape(x)[1]

        return (
            x
            + self.pos_encoding[
                :,
                :seq_length,
                :
            ]
        )


if __name__ == "__main__":

    layer = PositionalEncoding(
        max_length=40,
        d_model=256
    )

    dummy = tf.random.normal(
        (64, 40, 256)
    )

    output = layer(dummy)

    print(output.shape)