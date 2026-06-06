# src/data/dataloader.py

import tensorflow as tf
import numpy as np

from tokenizers import Tokenizer

from src.data.preprocessing import DatasetLoader


class TranslationDataLoader:

    def __init__(
            self,
            tokenizer_path="artifacts/tokenizer.json",
            max_length=40,
            batch_size=64
    ):

        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = \
            Tokenizer.from_file(
                tokenizer_path
            )

        self.pad_id = (
            self.tokenizer
            .token_to_id("[PAD]")
        )

    def encode_sentence(
            self,
            sentence
    ):

        # START + END already added
        ids = self.tokenizer.encode(
            sentence
        ).ids

        ids = ids[:self.max_length]

        padding = (
            self.max_length
            - len(ids)
        )

        ids += (
            [self.pad_id]
            * padding
        )

        return ids

    def prepare_dataset(
            self,
            pairs
    ):

        encoder_inputs = []
        decoder_inputs = []
        decoder_targets = []

        for pt, en in pairs:

            encoder = \
                self.encode_sentence(pt)

            decoder = \
                self.encode_sentence(en)

            decoder_input = \
                decoder[:-1]

            decoder_target = \
                decoder[1:]

            encoder_inputs.append(
                encoder
            )

            decoder_inputs.append(
                decoder_input
            )

            decoder_targets.append(
                decoder_target
            )

        return (
            np.array(
                encoder_inputs,
                dtype=np.int32
            ),

            np.array(
                decoder_inputs,
                dtype=np.int32
            ),

            np.array(
                decoder_targets,
                dtype=np.int32
            )
        )

    def create_tf_dataset(
            self,
            pairs,
            shuffle=True
    ):

        enc, dec_in, dec_tar = \
            self.prepare_dataset(
                pairs
            )

        dataset = (
            tf.data.Dataset
            .from_tensor_slices(
                (
                    {
                        "encoder_inputs":
                            enc,

                        "decoder_inputs":
                            dec_in
                    },
                    dec_tar
                )
            )
        )

        if shuffle:

            dataset = \
                dataset.shuffle(
                    20000
                )

        dataset = dataset.batch(
            self.batch_size
        )

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

        return dataset


if __name__ == "__main__":

    loader = DatasetLoader(
        max_samples=50000
    )

    train_pairs, _, _ = \
        loader.load_dataset()

    data_loader = \
        TranslationDataLoader()

    train_dataset = \
        data_loader.create_tf_dataset(
            train_pairs
        )

    for batch in train_dataset.take(1):

        x, y = batch

        print(
            "Encoder:",
            x[
                "encoder_inputs"
            ].shape
        )

        print(
            "Decoder:",
            x[
                "decoder_inputs"
            ].shape
        )

        print(
            "Target:",
            y.shape
        )

        print("\nExample Encoder Input:")
        print(
            x[
                "encoder_inputs"
            ][0][:20]
        )