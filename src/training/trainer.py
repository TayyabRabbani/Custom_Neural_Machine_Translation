# src/training/trainer.py

import tensorflow as tf

from src.models.lstm_seq2seq import Seq2Seq
from src.training.losses import MaskedLoss
from src.data.dataloader import (
    TranslationDataLoader
)
from src.data.preprocessing import (
    DatasetLoader
)
from tokenizers import Tokenizer


class Trainer:

    def __init__(self):

        self.tokenizer = \
            Tokenizer.from_file(
                "artifacts/tokenizer.json"
            )

        self.vocab_size = \
            self.tokenizer.get_vocab_size()

        self.model = Seq2Seq(
            vocab_size=self.vocab_size,
            embedding_dim=256,
            hidden_units=512
        )

        self.loss_fn = \
            MaskedLoss(
                pad_token_id=0
            )

        self.optimizer = \
            tf.keras.optimizers.Adam(
                learning_rate=1e-3
            )
        self.checkpoint_path = (
            "checkpoints/lstm/"
            "best.weights.h5"
        )

    @tf.function
    def train_step(
            self,
            x,
            y
    ):

        with tf.GradientTape() as tape:

            logits = self.model(
                x,
                training=True
            )

            loss = self.loss_fn(
                y,
                logits
            )

        gradients = tape.gradient(
            loss,
            self.model.trainable_variables
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.model.trainable_variables
            )
        )

        return loss

    def train(
            self,
            train_dataset,
            val_dataset,
            epochs=3
    ):

        best_val_loss = float("inf")

        for epoch in range(epochs):

            print(
                f"\nEpoch "
                f"{epoch + 1}/{epochs}"
            )

            total_loss = 0
            steps = 0

            for x, y in train_dataset:

                loss = self.train_step(
                    x,
                    y
                )

                total_loss += \
                    loss.numpy()

                steps += 1

                if steps % 20 == 0:
                    print(
                        f"Step "
                        f"{steps}"
                        f" | Loss:"
                        f" {loss:.4f}"
                    )

            train_loss = \
                total_loss / steps

            val_loss = \
                self.validate(
                    val_dataset
                )

            print(
                f"\nTrain Loss:"
                f" {train_loss:.4f}"
            )

            print(
                f"Validation Loss:"
                f" {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                self.model.save_weights(
                    self.checkpoint_path
                )

                print(
                    "Best model saved."
                )

    def validate(
            self,
            val_dataset
    ):

        total_loss = 0
        steps = 0

        for x, y in val_dataset:
            logits = self.model(
                x,
                training=False
            )

            loss = self.loss_fn(
                y,
                logits
            )

            total_loss += \
                loss.numpy()

            steps += 1

        return total_loss / steps

if __name__ == "__main__":
    
    loader = DatasetLoader(
        max_samples=2000
    )

    train_pairs, val_pairs, _ = \
        loader.load_dataset()

    data_loader = \
        TranslationDataLoader(
            batch_size=32
        )

    train_dataset = \
        data_loader.create_tf_dataset(
            train_pairs
        )

    val_dataset = \
        data_loader.create_tf_dataset(
            val_pairs,
            shuffle=False
        )

    trainer = Trainer()

    trainer.train(
        train_dataset,
        val_dataset,
        epochs=3
    )