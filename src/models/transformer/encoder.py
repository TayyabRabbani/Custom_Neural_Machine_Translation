# src/models/transformer/encoder.py

import tensorflow as tf

from src.models.transformer.attention import MultiHeadAttention
from src.models.transformer.positional_encoding import positional_encoding


def feed_forward(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation="relu"),
        tf.keras.layers.Dense(d_model),
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward(d_model, dff)

        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.drop1 = tf.keras.layers.Dropout(dropout_rate)
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False, padding_mask=None):
        attn, _ = self.mha(x, x, x, attn_mask=padding_mask)
        out1 = self.ln1(x + self.drop1(attn, training=training))

        ffn_out = self.ffn(out1)
        return self.ln2(out1 + self.drop2(ffn_out, training=training))


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        vocab_size,
        max_length,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos = positional_encoding(max_length, d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        self.drop = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False, padding_mask=None):
        seq_length = tf.shape(x)[1]

        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos[:, :seq_length, :]
        x = self.drop(x, training=training)

        for layer in self.enc_layers:
            x = layer(x, training=training, padding_mask=padding_mask)

        return x


if __name__ == "__main__":
    encoder = Encoder(
        num_layers=4, d_model=256, num_heads=8, dff=1024,
        vocab_size=10000, max_length=40,
    )
    dummy = tf.random.uniform((64, 40), maxval=10000, dtype=tf.int32)
    print("Encoder Output Shape:", encoder(dummy).shape)
