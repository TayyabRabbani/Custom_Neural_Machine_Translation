# src/models/transformer/decoder.py


import tensorflow as tf

from src.models.transformer.attention import MultiHeadAttention
from src.models.transformer.positional_encoding import positional_encoding
from src.models.transformer.encoder import feed_forward


def create_look_ahead_mask(size):
    return 1.0 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.mha1 = MultiHeadAttention(d_model, num_heads)   # masked self-attention
        self.mha2 = MultiHeadAttention(d_model, num_heads)   # cross-attention
        self.ffn = feed_forward(d_model, dff)

        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.drop1 = tf.keras.layers.Dropout(dropout_rate)
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)
        self.drop3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, encoder_output, training=False, combined_mask=None, padding_mask=None):
        attn1, _ = self.mha1(x, x, x, attn_mask=combined_mask)
        out1 = self.ln1(x + self.drop1(attn1, training=training))

        attn2, attn_weights = self.mha2(
            out1, encoder_output, encoder_output, attn_mask=padding_mask
        )
        out2 = self.ln2(out1 + self.drop2(attn2, training=training))

        ffn_out = self.ffn(out2)
        out3 = self.ln3(out2 + self.drop3(ffn_out, training=training))

        return out3, attn_weights


class Decoder(tf.keras.layers.Layer):
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

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        self.drop = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, encoder_output, training=False, combined_mask=None, padding_mask=None):
        seq_length = tf.shape(x)[1]

        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos[:, :seq_length, :]
        x = self.drop(x, training=training)

        attn_weights = None
        for layer in self.dec_layers:
            x, attn_weights = layer(
                x,
                encoder_output,
                training=training,
                combined_mask=combined_mask,
                padding_mask=padding_mask,
            )

        return x, attn_weights


if __name__ == "__main__":
    decoder = Decoder(
        num_layers=4, d_model=256, num_heads=8, dff=1024,
        vocab_size=10000, max_length=40,
    )
    encoder_output = tf.random.normal((64, 40, 256))
    decoder_input = tf.random.uniform((64, 39), maxval=10000, dtype=tf.int32)
    combined = create_look_ahead_mask(39)
    output, attn = decoder(decoder_input, encoder_output, combined_mask=combined)
    print("Decoder Output Shape:", output.shape)
