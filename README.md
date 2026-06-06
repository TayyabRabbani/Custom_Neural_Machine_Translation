# Custom Neural Machine Translation System

A from-scratch implementation of **Neural Machine Translation (NMT)** for **Portuguese → English translation** using both:

- **Seq2Seq LSTM**
- **Transformer Architecture**

Built entirely in **TensorFlow** without pretrained translation models.

---

## Project Overview

This project compares recurrent and attention-based architectures for Neural Machine Translation under constrained compute settings.

The goal was to understand:

- Sequence-to-sequence learning
- Tokenization strategies
- Teacher forcing
- Transformer attention mechanisms
- Training dynamics between LSTM and Transformer architectures

---

## Features

### Data Pipeline
- Custom preprocessing pipeline
- Portuguese-English dataset handling
- Dynamic batching and padding
- TensorFlow `tf.data` pipeline

### Tokenization
- Custom **Byte Pair Encoding (BPE)** tokenizer
- Shared vocabulary for Portuguese and English
- Special tokens:
  - `[PAD]`
  - `[UNK]`
  - `[START]`
  - `[END]`

### Models Implemented

#### 1. Seq2Seq LSTM
- Encoder–Decoder architecture
- Teacher forcing
- Greedy decoding inference

#### 2. Transformer
Implemented from scratch:

- Positional Encoding
- Multi-Head Self Attention
- Encoder Stack
- Decoder Stack
- Feed Forward Network
- Residual Connections
- Layer Normalization
- Masked Attention
- Warmup Learning Rate Scheduler (Vaswani et al.)

---

## Dataset

**Portuguese-English Translation Dataset**

- **50K sentence pairs**
- Conversational Portuguese ↔ English translations

---

## Training Environment

### Local
- **GPU:** RTX 3050 (8GB)

### Cloud
- **Platform:** Kaggle
- **GPU:** Tesla T4

---

## Results

| Model | Validation Loss |
|--------|----------------|
| Seq2Seq LSTM | ~4.7 |
| Transformer | **~1.55** |

The Transformer significantly outperformed the vanilla LSTM model while maintaining comparable parameter counts.

---

## Sample Translations

### Transformer Outputs

| Portuguese | Predicted English |
|------------|-------------------|
| `Eu gosto de futebol.` | `I like soccer.` |
| `Ela está estudando.` | `She is studying.` |
| `Você fala inglês?` | `Can you speak English?` |
| `Onde está meu carro?` | `Where's my car?` |
| `Eu não entendo.` | `I don't understand.` |

---

## Project Structure

```text
custom-neural-machine-translation/
│
├── artifacts/
│   └── tokenizer.json
│
├── configs/
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   │   ├── lstm_seq2seq.py
│   │   └── transformer/
│   └── training/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/custom-neural-machine-translation.git
cd custom-neural-machine-translation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Training

### LSTM

```bash
python -m src.training.trainer
```

### Transformer

```bash
python -m src.training.transformer_trainer
```

---

## Run Inference

### LSTM

```bash
python -m src.evaluation.inference
```

### Transformer

```bash
python -m src.evaluation.transformer_inference
```

---

## Future Improvements

- BLEU score benchmarking
- Beam search decoding
- Larger dataset training
- Attention visualization
- Transformer scaling experiments

---

## Author

**MD. Tayyab Rabbani**

- GitHub: https://github.com/TayyabRabbani
- LinkedIn: https://www.linkedin.com/in/md-tayyab-rabbani-757653291