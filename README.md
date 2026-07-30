# Custom Neural Machine Translation System

A from-scratch implementation of **Neural Machine Translation (NMT)** for
**Portuguese → English** translation, built entirely in **TensorFlow / Keras**
without any pretrained translation models:

- **Seq2Seq LSTM** (with and without Bahdanau attention)
- **Transformer** (full "Attention Is All You Need" architecture, from scratch)

---

## Project Overview

This project compares recurrent and attention-based architectures for NMT under
constrained compute. It was built to understand, end to end:

- Sequence-to-sequence learning and teacher forcing
- Subword tokenization (BPE)
- Bahdanau (additive) attention
- The Transformer: multi-head attention, positional encoding, masking, and the
  warmup learning-rate schedule
- Training dynamics and translation quality of LSTM vs. Transformer

---

## Features

### Data Pipeline
- Custom preprocessing for the Portuguese–English corpus
- Subword **Byte-Pair-Encoding (BPE)** tokenizer (shared PT/EN vocab, 10k tokens)
- Special tokens: `[PAD]`, `[UNK]`, `[START]`, `[END]`
- `tf.data` pipeline with padding, batching, and prefetch

### Models
**Seq2Seq LSTM** — encoder/decoder with teacher forcing and greedy decoding;
an attention variant adds Bahdanau attention over the encoder states.

**Transformer** (from scratch) — positional encoding, multi-head self-attention,
encoder/decoder stacks, position-wise feed-forward, residual connections, layer
normalization, padding + look-ahead masking, and the Vaswani warmup LR schedule.

---

## Dataset

Portuguese–English sentence pairs (Tatoeba / Anki `por.txt`,
**~169,000 pairs**, tab-separated `EN \t PT`). Split 80 / 10 / 10 into
train / validation / test with a fixed seed. The dataset file is not committed
(see `.gitignore`); download `por.txt` and place it at `src/data/por.txt`.

---

## Results

Transformer trained on the **full corpus** (Kaggle, Tesla T4, 20 epochs):

| Model | Data | Val loss | Val token acc | Test BLEU | Test chrF |
|-------|------|---------:|--------------:|----------:|----------:|
| Seq2Seq LSTM (full) | full corpus | ~1.78 | – | – | – |
| **Transformer** | full corpus | **0.75** | **0.86** | **64.0** | **75.3** |

BLEU / chrF are corpus scores (sacrebleu) over 1,000 held-out test sentences.
Because this is short conversational text, BLEU in the 60s is expected and is
not directly comparable to WMT-style benchmarks; many BLEU "misses" are valid
paraphrases.

**Transformer config:** `d_model=256`, `layers=4`, `heads=8`, `dff=1024`,
`max_len=40`, `dropout=0.1`, vocab `10000`, Adam(0.9, 0.98, 1e-9) with 4000 warmup steps.

### Sample translations (Transformer)

| Portuguese | Model output |
|------------|--------------|
| `Eu gosto de futebol.` | I like soccer. |
| `Ela está estudando.` | She's studying. |
| `Você fala inglês?` | Do you speak English? |
| `Onde está meu carro?` | Where's my car? |
| `Você pode me ajudar?` | Can you help me? |
| `Tom gosta de caçar.` | Tom likes hunting. |

---

## Project Structure

```text
NMT-Research/
├── artifacts/
│   ├── tokenizer.json                  # original shared BPE tokenizer
│   └── tokenizer_transformer.json      # tokenizer from the full Transformer run
├── configs/
│   └── data_config.py
├── kaggle/
│   ├── transformer_nmt_kaggle.ipynb    # self-contained Kaggle training notebook
│   └── build_nb.py                     # regenerates the notebook
├── src/
│   ├── data/                           # preprocessing, tokenizer, dataloader
│   ├── models/
│   │   ├── lstm_seq2seq.py
│   │   ├── attention_seq2seq.py
│   │   └── transformer/                # attention, encoder, decoder, positional enc.
│   ├── training/                       # trainers + masked loss + LR schedule
│   └── evaluation/                     # inference scripts
├── backend/                            # FastAPI app + LangChain HF reference
│   ├── main.py
│   └── reference.py
├── run_server.py                       # launches the backend
├── infer_transformer.py                # local inference + BLEU for the trained model
├── checkpoints/                        # weights (gitignored — see Weights below)
├── requirements.txt
├── requirements-serve.txt              # serving deps (fastapi, transformers, langchain…)
└── README.md
```

---

## Setup

```bash
git clone https://github.com/TayyabRabbani/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

The models load Keras-3 `.weights.h5` checkpoints, so use **TensorFlow 2.19**
(the version used for training on Kaggle). On Python 3.13, TF 2.19 has no wheel —
use a Python 3.10–3.12 environment.

---

## Train

The Transformer is trained on Kaggle via the self-contained notebook
[`kaggle/transformer_nmt_kaggle.ipynb`](kaggle/transformer_nmt_kaggle.ipynb):
upload `por.txt` as a dataset, enable the GPU, and run all cells. It trains the
tokenizer, builds the model, trains on the full corpus, and reports BLEU.

The original LSTM / Transformer training modules are also runnable locally:

```bash
python -m src.training.trainer                # LSTM
python -m src.training.transformer_trainer    # Transformer
```

---

## Inference + Evaluation

Local inference and BLEU for the trained Transformer. Place the trained weights
at `checkpoints/transformer/transformer_best.weights.h5` and the tokenizer at
`artifacts/tokenizer_transformer.json`, then:

```bash
# demo translations + BLEU/chrF on N held-out test sentences (default 1000)
python infer_transformer.py 1000

# interactive: type Portuguese, get English (run with no output redirect)
python infer_transformer.py
```

---

## Web App / API (FastAPI)

A FastAPI backend serves a side-by-side comparison: **your trained Transformer**
vs. a **Hugging Face reference model** (`Helsinki-NLP/opus-mt-ROMANCE-en`) exposed
through a LangChain `Runnable`. It includes a minimal web UI.

Install serving deps and run from the project root:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-serve.txt
python run_server.py          # http://127.0.0.1:8000
```

- `GET  /`          → web UI (type Portuguese, see both translations)
- `POST /translate` → `{"text": "..."}` ⇒ `{model_translation, reference_translation}`
- `GET  /docs`      → Swagger UI
- `GET  /health`    → status + reference model id

The reference model is configurable: set `HF_REF_MODEL` (and `HF_SRC_TOKEN` for
multilingual Marian models that need a source-language token).

```
backend/
├── main.py        # FastAPI app + web UI
└── reference.py   # HF model via LangChain (PT -> EN)
run_server.py      # launcher
```

---

## Weights

Model checkpoints (`*.h5`, `checkpoints/`) are **not** committed — they are large
binaries. To share trained weights, attach them to a GitHub Release or use
Git LFS rather than committing them to the repo.

---

## Future Improvements

- Beam-search decoding
- Attention-weight visualization
- Separate PT / EN tokenizers and larger vocabularies
- Transformer scaling experiments

---

## Author

**MD. Tayyab Rabbani**

- GitHub: https://github.com/TayyabRabbani
- LinkedIn: https://www.linkedin.com/in/md-tayyab-rabbani-757653291
