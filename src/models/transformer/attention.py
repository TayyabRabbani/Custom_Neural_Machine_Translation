# src/models/transformer/attention.py

import tensorflow as tf


def scaled_dot_product_attention(
    query,
    key,
    value,
    mask=None
):
    """
    query: (batch, heads, seq_q, depth)
    key:   (batch, heads, seq_k, depth)
    value: (batch, heads, seq_v, depth)
    """

    matmul_qk = tf.matmul(
        query,
        key,
        transpose_b=True
    )

    dk = tf.cast(
        tf.shape(key)[-1],
        tf.float32
    )

    scaled_logits = (
        matmul_qk
        / tf.math.sqrt(dk)
    )

    if mask is not None:

        scaled_logits += (
            mask * -1e9
        )

    attention_weights = tf.nn.softmax(
        scaled_logits,
        axis=-1
    )

    output = tf.matmul(
        attention_weights,
        value
    )

    return output, attention_weights


class MultiHeadAttention(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        d_model,
        num_heads
    ):
        super().__init__()
        self.supports_masking = True

        self.num_heads = num_heads
        self.d_model = d_model

        assert (
            d_model
            % num_heads
            == 0
        )

        self.depth = (
            d_model
            // num_heads
        )

        self.wq = tf.keras.layers.Dense(
            d_model
        )

        self.wk = tf.keras.layers.Dense(
            d_model
        )

        self.wv = tf.keras.layers.Dense(
            d_model
        )

        self.dense = (
            tf.keras.layers.Dense(
                d_model
            )
        )

    def split_heads(
        self,
        x,
        batch_size
    ):

        x = tf.reshape(
            x,
            (
                batch_size,
                -1,
                self.num_heads,
                self.depth
            )
        )

        return tf.transpose(
            x,
            perm=[0, 2, 1, 3]
        )

    def call(
        self,
        query,
        key,
        value,
        mask=None
    ):

        batch_size = tf.shape(
            query
        )[0]

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        query = self.split_heads(
            query,
            batch_size
        )

        key = self.split_heads(
            key,
            batch_size
        )

        value = self.split_heads(
            value,
            batch_size
        )

        scaled_attention, attention_weights = (
            scaled_dot_product_attention(
                query,
                key,
                value,
                mask
            )
        )

        scaled_attention = tf.transpose(
            scaled_attention,
            perm=[0, 2, 1, 3]
        )

        concat_attention = tf.reshape(
            scaled_attention,
            (
                batch_size,
                -1,
                self.d_model
            )
        )

        output = self.dense(
            concat_attention
        )

        return output, attention_weights


if __name__ == "__main__":

    layer = MultiHeadAttention(
        d_model=256,
        num_heads=8
    )

    dummy = tf.random.normal(
        (64, 40, 256)
    )

    output, attention = layer(
        dummy,
        dummy,
        dummy
    )

    print(
        "Output Shape:",
        output.shape
    )

    print(
        "Attention Shape:",
        attention.shape
    )