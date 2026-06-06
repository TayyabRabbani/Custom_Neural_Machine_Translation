# src/training/losses.py

import tensorflow as tf


class MaskedLoss:

    def __init__(
            self,
            pad_token_id=0
    ):

        self.pad_token_id = \
            pad_token_id

        self.loss_fn = \
            tf.keras.losses\
            .SparseCategoricalCrossentropy(
                from_logits=True,
                reduction="none"
            )

    def __call__(
            self,
            y_true,
            y_pred
    ):

        loss = self.loss_fn(
            y_true,
            y_pred
        )

        mask = tf.cast(
            y_true
            != self.pad_token_id,
            tf.float32
        )

        loss *= mask

        return (
            tf.reduce_sum(loss)
            /
            tf.reduce_sum(mask)
        )


if __name__ == "__main__":

    loss_fn = MaskedLoss()

    y_true = tf.constant([
        [1, 2, 3, 0, 0]
    ])

    y_pred = tf.random.normal(
        (1, 5, 10000)
    )

    loss = loss_fn(
        y_true,
        y_pred
    )

    print(
        "Masked Loss:",
        loss.numpy()
    )