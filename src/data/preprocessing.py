# src/data/preprocessing.py

from pathlib import Path
import pandas as pd

import random

random.seed(42)


class DatasetLoader:

    def __init__(
            self,
            file_path="src/data/por.txt",
            max_samples=50000
    ):

        self.file_path = Path(file_path)
        self.max_samples = max_samples

    def load_dataset(self):

        pairs = []

        with open(
                self.file_path,
                "r",
                encoding="utf-8"
        ) as f:

            for line in f:

                parts = line.strip().split("\t")

                # Need at least English + Portuguese
                if len(parts) < 2:
                    continue

                en = parts[0].strip()
                pt = parts[1].strip()

                # Skip empty rows
                if not en or not pt:
                    continue

                pairs.append((pt, en))

        print(
            f"Original dataset size: {len(pairs)}"
        )

        # Keep manageable size
        random.shuffle(pairs)
        pairs = pairs[:self.max_samples]

        print(
            f"Using first {len(pairs)} samples"
        )

        # Split dataset
        train_size = int(
            0.8 * len(pairs)
        )

        val_size = int(
            0.1 * len(pairs)
        )

        train_pairs = pairs[:train_size]

        val_pairs = pairs[
            train_size:
            train_size + val_size
        ]

        test_pairs = pairs[
            train_size + val_size:
        ]

        print("\nDataset Loaded")
        print(
            f"Train: {len(train_pairs)}"
        )

        print(
            f"Validation: {len(val_pairs)}"
        )

        print(
            f"Test: {len(test_pairs)}"
        )

        return (
            train_pairs,
            val_pairs,
            test_pairs
        )

    def inspect_samples(
            self,
            pairs,
            n=5
    ):

        print(
            "\nSample Translation Pairs:\n"
        )

        for pt, en in pairs[:n]:

            print(
                f"Portuguese: {pt}"
            )

            print(
                f"English:   {en}"
            )

            print("-" * 80)

    def sentence_statistics(
            self,
            pairs
    ):

        pt_lengths = []
        en_lengths = []

        for pt, en in pairs:

            pt_lengths.append(
                len(pt.split())
            )

            en_lengths.append(
                len(en.split())
            )

        stats = {
            "pt_mean_length":
                sum(pt_lengths)
                / len(pt_lengths),

            "en_mean_length":
                sum(en_lengths)
                / len(en_lengths),

            "pt_max_length":
                max(pt_lengths),

            "en_max_length":
                max(en_lengths)
        }

        return pd.DataFrame(
            [stats]
        )


if __name__ == "__main__":

    loader = DatasetLoader(
        max_samples=50000
    )

    train_pairs, val_pairs, test_pairs = \
        loader.load_dataset()

    loader.inspect_samples(
        train_pairs
    )

    stats = loader.sentence_statistics(
        train_pairs
    )

    print(
        "\nDataset Statistics:"
    )

    print(stats)