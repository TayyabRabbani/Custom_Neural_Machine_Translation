# src/models/lstm_seq2seq.py

import tensorflow as tf


class Encoder(tf.keras.Model):

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_units
    ):

        super().__init__()

        self.embedding = \
            tf.keras.layers.Embedding(
                vocab_size,
                embedding_dim,
                mask_zero=True
            )

        self.lstm = \
            tf.keras.layers.LSTM(
                hidden_units,
                return_sequences=True,
                return_state=True
            )

    def call(
            self,
            x,
            training=False
    ):

        x = self.embedding(x)

        output, h, c = \
            self.lstm(
                x,
                training=training
            )

        return output, h, c


class Decoder(tf.keras.Model):

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_units
    ):

        super().__init__()

        self.embedding = \
            tf.keras.layers.Embedding(
                vocab_size,
                embedding_dim,
                mask_zero=True
            )

        self.lstm = \
            tf.keras.layers.LSTM(
                hidden_units,
                return_sequences=True,
                return_state=True
            )

        self.fc = \
            tf.keras.layers.Dense(
                vocab_size
            )

    def call(
            self,
            x,
            hidden,
            cell,
            training=False
    ):

        x = self.embedding(x)

        output, h, c = \
            self.lstm(
                x,
                initial_state=[
                    hidden,
                    cell
                ],
                training=training
            )

        logits = self.fc(
            output
        )

        return logits, h, c


class Seq2Seq(tf.keras.Model):

    def __init__(
            self,
            vocab_size,
            embedding_dim=256,
            hidden_units=512
    ):

        super().__init__()

        self.encoder = Encoder(
            vocab_size,
            embedding_dim,
            hidden_units
        )

        self.decoder = Decoder(
            vocab_size,
            embedding_dim,
            hidden_units
        )

    def call(
            self,
            inputs,
            training=False
    ):

        encoder_inputs = \
            inputs[
                "encoder_inputs"
            ]

        decoder_inputs = \
            inputs[
                "decoder_inputs"
            ]

        _, h, c = \
            self.encoder(
                encoder_inputs,
                training=training
            )

        logits, _, _ = \
            self.decoder(
                decoder_inputs,
                h,
                c,
                training=training
            )

        return logits


if __name__ == "__main__":

    vocab_size = 10000

    model = Seq2Seq(
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

    output = model(dummy_batch)

    print(
        "Output Shape:",
        output.shape
    )

    model.summary()