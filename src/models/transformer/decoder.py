# src/models/transformer/decoder.py

import tensorflow as tf

from src.models.transformer.attention import (
    MultiHeadAttention
)

from src.models.transformer.positional_encoding import (
    PositionalEncoding
)

from src.models.transformer.encoder import (
    point_wise_feed_forward_network
)


def create_look_ahead_mask(size):

    mask = (
        1
        - tf.linalg.band_part(
            tf.ones(
                (size, size)
            ),
            -1,
            0
        )
    )

    return mask


class DecoderLayer(
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

        # masked self-attention
        self.mha1 = (
            MultiHeadAttention(
                d_model,
                num_heads
            )
        )

        # encoder-decoder attention
        self.mha2 = (
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

        self.layernorm3 = (
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

        self.dropout3 = (
            tf.keras.layers.Dropout(
                dropout_rate
            )
        )

    def call(
        self,
        x,
        encoder_output,
        training=False,
        look_ahead_mask=None,
        padding_mask=None
    ):

        # masked self attention
        attn1, _ = (
            self.mha1(
                x,
                x,
                x,
                look_ahead_mask
            )
        )

        attn1 = self.dropout1(
            attn1,
            training=training
        )

        out1 = self.layernorm1(
            x + attn1
        )

        # encoder-decoder attention
        attn2, attention_weights = (
            self.mha2(
                out1,
                encoder_output,
                encoder_output,
                padding_mask
            )
        )

        attn2 = self.dropout2(
            attn2,
            training=training
        )

        out2 = self.layernorm2(
            out1 + attn2
        )

        # feed forward
        ffn_output = self.ffn(
            out2
        )

        ffn_output = self.dropout3(
            ffn_output,
            training=training
        )

        out3 = self.layernorm3(
            out2 + ffn_output
        )

        return out3, attention_weights


class Decoder(
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

        self.decoder_layers = [

            DecoderLayer(
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
        encoder_output,
        training=False,
        look_ahead_mask=None,
        padding_mask=None
    ):

        seq_length = tf.shape(
            x
        )[1]

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

        attention_weights = {}

        for i, layer in enumerate(
            self.decoder_layers
        ):

            x, block = layer(
                x,
                encoder_output,
                training=training,
                look_ahead_mask=
                look_ahead_mask,
                padding_mask=
                padding_mask
            )

            attention_weights[
                f"decoder_layer_"
                f"{i + 1}"
            ] = block

        return x, attention_weights


if __name__ == "__main__":

    decoder = Decoder(
        num_layers=3,
        d_model=256,
        num_heads=8,
        dff=1024,
        vocab_size=10000,
        max_length=40
    )

    encoder_output = (
        tf.random.normal(
            (64, 40, 256)
        )
    )

    decoder_input = (
        tf.random.uniform(
            (64, 39),
            maxval=10000,
            dtype=tf.int32
        )
    )

    look_ahead_mask = (
        create_look_ahead_mask(
            39
        )
    )

    output, attention = (
        decoder(
            decoder_input,
            encoder_output,
            look_ahead_mask=
            look_ahead_mask
        )
    )

    print(
        "Decoder Output Shape:",
        output.shape
    )