# src/data/tokenizer.py

from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

from src.data.preprocessing import DatasetLoader


class BPETokenizerTrainer:

    def __init__(
            self,
            vocab_size=10000,
            save_path="artifacts/tokenizer.json"
    ):

        self.vocab_size = vocab_size
        self.save_path = Path(save_path)

        self.special_tokens = [
            "[PAD]",
            "[UNK]",
            "[START]",
            "[END]"
        ]

    def train(self, train_pairs):

        tokenizer = Tokenizer(
            BPE(
                unk_token="[UNK]"
            )
        )

        # ByteLevel pre-tokenizer encodes spaces as Ġ;
        # ByteLevelDecoder reverses this correctly.
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens
        )

        corpus = []

        print(
            "Preparing corpus..."
        )

        for pt, en in train_pairs:

            corpus.append(pt)
            corpus.append(en)

        print(
            "Training tokenizer..."
        )

        tokenizer.train_from_iterator(
            corpus,
            trainer=trainer
        )

        tokenizer.post_processor = TemplateProcessing(
            single="[START] $A [END]",
            pair="[START] $A [END] $B:1 [END]:1",
            special_tokens=[
                (
                    "[START]",
                    tokenizer.token_to_id("[START]")
                ),
                (
                    "[END]",
                    tokenizer.token_to_id("[END]")
                )
            ]
        )

        self.save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        tokenizer.save(
            str(self.save_path)
        )

        print(
            f"Tokenizer saved to:"
        )

        print(self.save_path)

        return tokenizer

    @staticmethod
    def load(save_path="artifacts/tokenizer.json"):

        tokenizer = Tokenizer.from_file(
            str(save_path)
        )

        return tokenizer

    def configure_padding(self, tokenizer, max_length):

        tokenizer.enable_padding(
            pad_id=tokenizer.token_to_id("[PAD]"),
            pad_token="[PAD]",
            length=max_length
        )

        tokenizer.enable_truncation(
            max_length=max_length
        )

        return tokenizer

    def test_tokenizer(
            self,
            tokenizer
    ):

        sentence = (
            "Eu gosto de estudar "
            "aprendizado profundo."
        )

        encoded = tokenizer.encode(
            sentence
        )

        decoded = tokenizer.decode(
            encoded.ids,
            skip_special_tokens=True
        )

        print("\nTest Sentence:")
        print(sentence)

        print("\nToken IDs:")
        print(encoded.ids)

        print("\nTokens:")
        print(encoded.tokens)

        print("\nDecoded:")
        print(decoded)


if __name__ == "__main__":

    loader = DatasetLoader(
        max_samples=50000
    )

    train_pairs, _, _ = \
        loader.load_dataset()

    trainer = \
        BPETokenizerTrainer(
            vocab_size=10000
        )

    tokenizer = trainer.train(
        train_pairs
    )

    trainer.test_tokenizer(
        tokenizer
    )