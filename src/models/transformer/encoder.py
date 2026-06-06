# src/models/transformer/encoder.py

import tensorflow as tf

from src.models.transformer.attention import (
    MultiHeadAttention
)

from src.models.transformer.positional_encoding import (
    PositionalEncoding
)


def point_wise_feed_forward_network(
    d_model,
    dff
):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(
            dff,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            d_model
        )
    ])


class EncoderLayer(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        d_model,
        num_heads,
        dff,
        dropout_rate=0.1
    ):
        super().__init__()
        self.supports_masking = True

        self.mha = (
            MultiHeadAttention(
                d_model,
                num_heads
            )
        )

        self.ffn = (
            point_wise_feed_forward_network(
                d_model,
                dff
            )
        )

        self.layernorm1 = (
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6
            )
        )

        self.layernorm2 = (
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6
            )
        )

        self.dropout1 = (
            tf.keras.layers.Dropout(
                dropout_rate
            )
        )

        self.dropout2 = (
            tf.keras.layers.Dropout(
                dropout_rate
            )
        )

    def call(
        self,
        x,
        training=False,
        mask=None
    ):

        attention_output, _ = (
            self.mha(
                x,
                x,
                x,
                mask
            )
        )

        attention_output = (
            self.dropout1(
                attention_output,
                training=training
            )
        )

        out1 = (
            self.layernorm1(
                x + attention_output
            )
        )

        ffn_output = (
            self.ffn(out1)
        )

        ffn_output = (
            self.dropout2(
                ffn_output,
                training=training
            )
        )

        out2 = (
            self.layernorm2(
                out1 + ffn_output
            )
        )

        return out2


class Encoder(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        vocab_size,
        max_length,
        dropout_rate=0.1
    ):
        super().__init__()
        self.supports_masking = True

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = (
            tf.keras.layers.Embedding(
                vocab_size,
                d_model
            )
        )

        self.pos_encoding = (
            PositionalEncoding(
                max_length,
                d_model
            )
        )

        self.encoder_layers = [

            EncoderLayer(
                d_model,
                num_heads,
                dff,
                dropout_rate
            )

            for _ in range(
                num_layers
            )
        ]

        self.dropout = (
            tf.keras.layers.Dropout(
                dropout_rate
            )
        )

    def call(
        self,
        x,
        training=False,
        mask=None
    ):

        seq_length = tf.shape(x)[1]

        x = self.embedding(x)

        x *= tf.math.sqrt(
            tf.cast(
                self.d_model,
                tf.float32
            )
        )

        x = self.pos_encoding(x)

        x = self.dropout(
            x,
            training=training
        )

        for layer in (
            self.encoder_layers
        ):

            x = layer(
                x,
                training=training,
                mask=mask
            )

        return x


if __name__ == "__main__":

    encoder = Encoder(
        num_layers=3,
        d_model=256,
        num_heads=8,
        dff=1024,
        vocab_size=10000,
        max_length=40
    )

    dummy_input = (
        tf.random.uniform(
            (64, 40),
            maxval=10000,
            dtype=tf.int32
        )
    )

    output = encoder(
        dummy_input
    )

    print(
        "Encoder Output Shape:",
        output.shape
    )