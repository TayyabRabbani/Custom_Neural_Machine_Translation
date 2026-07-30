# src/models/transformer/attention.py

import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.out = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, attn_mask=None):
        batch_size = tf.shape(query)[0]

        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)

        logits = tf.matmul(query, key, transpose_b=True)
        logits = logits / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if attn_mask is not None:
            logits += attn_mask * -1e9

        weights = tf.nn.softmax(logits, axis=-1)

        context = tf.matmul(weights, value)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.d_model))

        return self.out(context), weights


if __name__ == "__main__":
    layer = MultiHeadAttention(d_model=256, num_heads=8)
    dummy = tf.random.normal((64, 40, 256))
    output, attention = layer(dummy, dummy, dummy)
    print("Output Shape:", output.shape)
    print("Attention Shape:", attention.shape)
