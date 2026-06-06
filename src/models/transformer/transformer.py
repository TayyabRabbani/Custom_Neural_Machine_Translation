# src/models/transformer/transformer.py

import tensorflow as tf

from src.models.transformer.encoder import (
    Encoder
)

from src.models.transformer.decoder import (
    Decoder,
    create_look_ahead_mask
)


class Transformer(
    tf.keras.Model
):

    def __init__(
        self,
        vocab_size,
        num_layers=3,
        d_model=256,
        num_heads=8,
        dff=1024,
        max_length=40,
        dropout_rate=0.1
    ):
        super().__init__()

        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=vocab_size,
            max_length=max_length,
            dropout_rate=dropout_rate
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=vocab_size,
            max_length=max_length,
            dropout_rate=dropout_rate
        )

        self.final_layer = (
            tf.keras.layers.Dense(
                vocab_size
            )
        )

    def create_padding_mask(
        self,
        seq
    ):

        mask = tf.cast(
            tf.math.equal(
                seq,
                0
            ),
            tf.float32
        )

        return mask[
            :,
            tf.newaxis,
            tf.newaxis,
            :
        ]

    def call(
        self,
        inputs,
        training=False
    ):

        encoder_inputs = (
            inputs[
                "encoder_inputs"
            ]
        )

        decoder_inputs = (
            inputs[
                "decoder_inputs"
            ]
        )

        encoder_padding_mask = (
            self.create_padding_mask(
                encoder_inputs
            )
        )

        decoder_padding_mask = (
            self.create_padding_mask(
                encoder_inputs
            )
        )

        # 1. Create padding mask for the decoder inputs
        dec_target_padding_mask = (
            self.create_padding_mask(
                decoder_inputs
            )
        )

        seq_length = tf.shape(
            decoder_inputs
        )[1]

        # 2. Create the look-ahead mask
        look_ahead_mask = (
            create_look_ahead_mask(
                seq_length
            )
        )

        # 3. Combine them to ensure padded tokens aren't attended to
        combined_mask = tf.maximum(
            look_ahead_mask,
            dec_target_padding_mask
        )

        encoder_output = (
            self.encoder(
                encoder_inputs,
                training=training,
                mask=
                encoder_padding_mask
            )
        )

        decoder_output, attention_weights = (
            self.decoder(
                decoder_inputs,
                encoder_output,
                training=training,
                look_ahead_mask=
                combined_mask,          # <-- Updated here
                padding_mask=
                decoder_padding_mask    # <-- Encoder padding mask for cross-attention
            )
        )

        logits = (
            self.final_layer(
                decoder_output
            )
        )

        return logits


if __name__ == "__main__":

    vocab_size = 10000

    model = Transformer(
        vocab_size=vocab_size
    )

    dummy_batch = {

        "encoder_inputs":
            tf.random.uniform(
                (64, 40),
                maxval=vocab_size,
                dtype=tf.int32
            ),

        "decoder_inputs":
            tf.random.uniform(
                (64, 39),
                maxval=vocab_size,
                dtype=tf.int32
            )
    }

    output = model(
        dummy_batch
    )

    print(
        "Output Shape:",
        output.shape
    )

    model.summary()