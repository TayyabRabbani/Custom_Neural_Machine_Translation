# src/evaluation/transformer_inference.py

import tensorflow as tf

from tokenizers import (
    Tokenizer
)

from src.models.transformer.transformer import (
    Transformer
)


class TransformerTranslator:

    def __init__(
        self,
        tokenizer_path=
        "artifacts/tokenizer.json",

        weights_path=
        "checkpoints/transformer/"
        "best.weights.h5",

        max_length=40
    ):

        self.max_length = (
            max_length
        )

        self.tokenizer = (
            Tokenizer.from_file(
                tokenizer_path
            )
        )

        self.vocab_size = (
            self.tokenizer
            .get_vocab_size()
        )

        self.pad_id = (
            self.tokenizer
            .token_to_id(
                "[PAD]"
            )
        )

        self.start_id = (
            self.tokenizer
            .token_to_id(
                "[START]"
            )
        )

        self.end_id = (
            self.tokenizer
            .token_to_id(
                "[END]"
            )
        )

        self.model = (
            Transformer(
                vocab_size=
                self.vocab_size
            )
        )

        dummy_batch = {

            "encoder_inputs":
                tf.zeros(
                    (1, 40),
                    dtype=tf.int32
                ),

            "decoder_inputs":
                tf.zeros(
                    (1, 39),
                    dtype=tf.int32
                )
        }

        _ = self.model(
            dummy_batch
        )

        self.model.load_weights(
            weights_path
        )

        print(
            "Transformer loaded."
        )

    def encode_sentence(
        self,
        sentence
    ):

        ids = (
            self.tokenizer
            .encode(sentence)
            .ids
        )

        ids = ids[
            :self.max_length
        ]

        padding = (
            self.max_length
            - len(ids)
        )

        ids += (
            [self.pad_id]
            * padding
        )

        return tf.constant(
            [ids],
            dtype=tf.int32
        )

    def translate(
        self,
        sentence
    ):

        encoder_input = (
            self.encode_sentence(
                sentence
            )
        )

        generated_ids = [
            self.start_id
        ]

        for step in range(
            self.max_length
        ):

            decoder_input = (
                tf.constant(
                    [generated_ids],
                    dtype=tf.int32
                )
            )

            logits = (
                self.model(
                    {
                        "encoder_inputs":
                        encoder_input,

                        "decoder_inputs":
                        decoder_input
                    },
                    training=False
                )
            )

            next_token = (
                tf.argmax(
                    logits[
                        :,
                        -1,
                        :
                    ],
                    axis=-1
                )
                .numpy()[0]
            )

            if (
                next_token
                == self.end_id
            ):
                break

            if (
                next_token
                == self.start_id
            ):
                continue

            generated_ids.append(
                int(next_token)
            )

        translation = (
            self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )
        )

        translation = (
            translation.replace(
                "Ġ",
                " "
            )
        )

        translation = (
            " ".join(
                translation.split()
            )
        )

        return translation


if __name__ == "__main__":

    translator = (
        TransformerTranslator()
    )

    test_sentences = [

        "Eu gosto de futebol.",

        "Ela está estudando.",

        "Você fala inglês?",

        "Eu estou cansado.",

        "Nós vamos à escola.",

        "Onde está meu carro?",

        "Eu gosto de chocolate.",

        "Você pode me ajudar?",

        "Eu não entendo.",

        "Ela gosta de música."
    ]

    for sentence in (
        test_sentences
    ):

        translation = (
            translator.translate(
                sentence
            )
        )

        print(
            f"\nPT: {sentence}"
        )

        print(
            f"EN: {translation}"
        )

        print(
            "-" * 50
        )