# src/evaluation/inference.py

import tensorflow as tf
from src.models.lstm_seq2seq import Seq2Seq
from tokenizers import Tokenizer


class Translator:
    def __init__(
        self,
        tokenizer_path="artifacts/tokenizer.json",
        weights_path="checkpoints/lstm/" "best.weights.h5",
        max_length=40,
    ):

        self.max_length = max_length

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.vocab_size = self.tokenizer.get_vocab_size()

        self.pad_id = self.tokenizer.token_to_id("[PAD]")

        self.start_id = self.tokenizer.token_to_id("[START]")

        self.end_id = self.tokenizer.token_to_id("[END]")

        self.model = Seq2Seq(vocab_size=self.vocab_size)

        # Build model once
        dummy_input = {
            "encoder_inputs": tf.zeros((1, 40), dtype=tf.int32),
            "decoder_inputs": tf.zeros((1, 39), dtype=tf.int32),
        }

        self.model(dummy_input)

        self.model.load_weights(weights_path)

        print("Model loaded.")

    def encode_sentence(self, sentence):

        ids = self.tokenizer.encode(sentence).ids

        ids = ids[: self.max_length]

        padding = self.max_length - len(ids)

        ids += [self.pad_id] * padding

        return tf.constant([ids], dtype=tf.int32)

    def translate(self, sentence):

        encoder_input = self.encode_sentence(sentence)

        # Encode sentence
        _, hidden, cell = self.model.encoder(encoder_input, training=False)

        generated_ids = [self.start_id]

        for step in range(self.max_length):
            # Feed ONLY the most recently generated token
            decoder_input = tf.constant([[generated_ids[-1]]], dtype=tf.int32)

            logits, hidden, cell = self.model.decoder(
                decoder_input, hidden, cell, training=False
            )

            # Because sequence length is now 1, we just take the 0th index for sequence length
            next_token = tf.argmax(logits[:, 0, :], axis=-1).numpy()[0]

            # Ignore repeated START token
            if next_token == self.start_id:
                continue

            # Stop at END
            if next_token == self.end_id:
                break

            generated_ids.append(int(next_token))

        translation = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        translation = translation.replace("Ġ", " ")

        translation = " ".join(translation.split())

        return translation


if __name__ == "__main__":

    translator = Translator()

    test_sentences = [
        "Eu gosto de futebol.",
        "Ela está estudando.",
        "Você fala inglês?",
        "Eu amo aprendizado profundo.",
    ]

    for sentence in test_sentences:

        translation = translator.translate(sentence)

        print(f"\nPT: {sentence}")

        print(f"EN: {translation}")

        print("-" * 50)
