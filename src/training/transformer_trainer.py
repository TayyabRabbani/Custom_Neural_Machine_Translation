# src/training/transformer_trainer.py

import json
import os

import tensorflow as tf
from tqdm import tqdm
from tokenizers import Tokenizer

from src.models.transformer.transformer import (
    Transformer
)

from src.training.losses import (
    MaskedLoss
)


class CustomSchedule(
    tf.keras.optimizers.schedules
    .LearningRateSchedule
):
    """
    Transformer learning rate
    schedule from the paper
    """

    def __init__(
        self,
        d_model,
        warmup_steps=4000
    ):
        super().__init__()

        self.d_model = tf.cast(
            d_model,
            tf.float32
        )

        self.warmup_steps = (
            warmup_steps
        )

    def __call__(
        self,
        step
    ):

        step = tf.cast(
            step,
            tf.float32
        )

        arg1 = tf.math.rsqrt(
            step
        )

        arg2 = (
            step *
            (
                self.warmup_steps
                ** -1.5
            )
        )

        return (
            tf.math.rsqrt(
                self.d_model
            )
            *
            tf.math.minimum(
                arg1,
                arg2
            )
        )


class TransformerTrainer:

    def __init__(
        self,
        tokenizer_path=
        "artifacts/tokenizer.json",

        checkpoint_dir=
        "checkpoints/transformer",

        d_model=256,
        num_layers=3,
        num_heads=8,
        dff=1024,
        max_length=40
    ):

        self.tokenizer = (
            Tokenizer.from_file(
                tokenizer_path
            )
        )

        self.vocab_size = (
            self.tokenizer
            .get_vocab_size()
        )

        self.model = (
            Transformer(
                vocab_size=
                self.vocab_size,

                num_layers=
                num_layers,

                d_model=
                d_model,

                num_heads=
                num_heads,

                dff=dff,

                max_length=
                max_length
            )
        )

        self.loss_fn = (
            MaskedLoss(
                pad_token_id=0
            )
        )

        self.learning_rate = (
            CustomSchedule(
                d_model=
                d_model
            )
        )

        self.optimizer = (
            tf.keras.optimizers.Adam(
                learning_rate=
                self.learning_rate,

                beta_1=0.9,
                beta_2=0.98,
                epsilon=1e-9
            )
        )

        self.checkpoint_dir = (
            checkpoint_dir
        )

        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True
        )

        self.best_weights_path = (
            os.path.join(
                self.checkpoint_dir,
                "best.weights.h5"
            )
        )

        self.final_weights_path = (
            os.path.join(
                self.checkpoint_dir,
                "final.weights.h5"
            )
        )

        self.history_path = (
            os.path.join(
                self.checkpoint_dir,
                "history.json"
            )
        )

        self.history = {
            "train_loss": [],
            "val_loss": []
        }

    def train_step(
        self,
        batch
    ):

        x, y = batch

        with tf.GradientTape() as tape:

            logits = (
                self.model(
                    {
                        "encoder_inputs":
                        x[
                            "encoder_inputs"
                        ],

                        "decoder_inputs":
                        x[
                            "decoder_inputs"
                        ]
                    },
                    training=True
                )
            )

            loss = (
                self.loss_fn(
                    y,
                    logits
                )
            )

        gradients = (
            tape.gradient(
                loss,
                self.model
                .trainable_variables
            )
        )

        gradients, _ = (
            tf.clip_by_global_norm(
                gradients,
                1.0
            )
        )

        self.optimizer\
            .apply_gradients(

            zip(
                gradients,
                self.model
                .trainable_variables
            )
        )

        return loss

    def validate(
        self,
        val_dataset
    ):

        total_loss = 0
        steps = 0

        val_bar = tqdm(
            val_dataset,
            desc="Validation",
            unit="batch"
        )

        for batch in val_bar:

            x, y = batch

            logits = (
                self.model(
                    {
                        "encoder_inputs":
                        x[
                            "encoder_inputs"
                        ],

                        "decoder_inputs":
                        x[
                            "decoder_inputs"
                        ]
                    },
                    training=False
                )
            )

            loss = (
                self.loss_fn(
                    y,
                    logits
                )
            )

            total_loss += (
                loss.numpy()
            )

            steps += 1

            val_bar.set_postfix(
                loss=
                f"{loss:.4f}"
            )

        return (
            total_loss / steps
        )

    def save_history(
        self
    ):

        with open(
            self.history_path,
            "w"
        ) as file:

            json.dump(
                self.history,
                file,
                indent=4
            )

    def train(
        self,
        train_dataset,
        val_dataset,
        epochs=7
    ):

        best_val_loss = (
            float("inf")
        )

        for epoch in range(
            epochs
        ):

            print(
                f"\nEpoch "
                f"{epoch + 1}"
                f"/{epochs}"
            )

            total_loss = 0
            steps = 0

            train_bar = tqdm(
                train_dataset,
                desc="Training",
                unit="batch"
            )

            for batch in train_bar:

                loss = (
                    self.train_step(
                        batch
                    )
                )

                total_loss += (
                    loss.numpy()
                )

                steps += 1

                train_bar.set_postfix(
                    loss=
                    f"{loss:.4f}"
                )

            train_loss = (
                total_loss / steps
            )

            val_loss = (
                self.validate(
                    val_dataset
                )
            )

            self.history[
                "train_loss"
            ].append(
                float(
                    train_loss
                )
            )

            self.history[
                "val_loss"
            ].append(
                float(
                    val_loss
                )
            )

            print(
                f"\nTrain Loss:"
                f" {train_loss:.4f}"
            )

            print(
                f"Validation Loss:"
                f" {val_loss:.4f}"
            )

            if (
                val_loss
                < best_val_loss
            ):

                best_val_loss = (
                    val_loss
                )

                self.model\
                    .save_weights(

                    self
                    .best_weights_path
                )

                print(
                    "Best model "
                    "saved."
                )

            self.save_history()

        self.model.save_weights(
            self.final_weights_path
        )

        print(
            "\nTraining Complete"
        )

        print(
            "Best Weights:"
        )

        print(
            self.best_weights_path
        )