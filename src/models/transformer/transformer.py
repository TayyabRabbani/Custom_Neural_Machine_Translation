# src/models/transformer/transformer.py

import tensorflow as tf

from src.models.transformer.encoder import Encoder
from src.models.transformer.decoder import Decoder, create_look_ahead_mask


class Transformer(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        num_layers=4,
        d_model=256,
        num_heads=8,
        dff=1024,
        max_length=40,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=vocab_size,
            max_length=max_length,
            dropout_rate=dropout_rate,
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=vocab_size,
            max_length=max_length,
            dropout_rate=dropout_rate,
        )

        self.final_layer = tf.keras.layers.Dense(vocab_size)

    @staticmethod
    def padding_mask(seq):
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def encode(self, encoder_inputs, training=False):
        pad_mask = self.padding_mask(encoder_inputs)
        encoder_output = self.encoder(
            encoder_inputs, training=training, padding_mask=pad_mask
        )
        return encoder_output, pad_mask

    def decode(self, decoder_inputs, encoder_output, encoder_pad_mask, training=False):
        seq_length = tf.shape(decoder_inputs)[1]
        combined_mask = tf.maximum(
            create_look_ahead_mask(seq_length),
            self.padding_mask(decoder_inputs),
        )
        decoder_output, attn = self.decoder(
            decoder_inputs,
            encoder_output,
            training=training,
            combined_mask=combined_mask,
            padding_mask=encoder_pad_mask,
        )
        return self.final_layer(decoder_output), attn

    def call(self, inputs, training=False):
        encoder_output, pad_mask = self.encode(
            inputs["encoder_inputs"], training=training
        )
        logits, _ = self.decode(
            inputs["decoder_inputs"], encoder_output, pad_mask, training=training
        )
        return logits


if __name__ == "__main__":
    vocab_size = 10000
    model = Transformer(vocab_size=vocab_size)
    dummy = {
        "encoder_inputs": tf.random.uniform((64, 40), maxval=vocab_size, dtype=tf.int32),
        "decoder_inputs": tf.random.uniform((64, 39), maxval=vocab_size, dtype=tf.int32),
    }
    print("Output Shape:", model(dummy).shape)
    model.summary()
