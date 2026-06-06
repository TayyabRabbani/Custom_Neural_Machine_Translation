# configs/data_config.py

DATASET_NAME = "ted_hrlr_translate/pt_to_en"

VOCAB_SIZE = 10000
MAX_LENGTH = 40
BATCH_SIZE = 64
BUFFER_SIZE = 20000

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[START]",
    "[END]"
]

TOKENIZER_PATH = "artifacts/tokenizer.json"

PROCESSED_DATA_DIR = "artifacts/processed"