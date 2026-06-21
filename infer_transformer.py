# Standalone local inference for the Kaggle-trained Transformer.
# Place the downloaded files here (or edit ART_DIR):
#   kaggle/from_kaggle/tokenizer.json
#   kaggle/from_kaggle/transformer_best.weights.h5
#
# Run:  python infer_transformer.py
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import numpy as np
import tensorflow as tf
from tokenizers import Tokenizer

# ---- must match the Kaggle full-run config (notebook CFG defaults) ----
MAX_LENGTH = 40
D_MODEL    = 256
NUM_LAYERS = 4
NUM_HEADS  = 8
DFF        = 1024

TOKENIZER_PATH = os.path.join("artifacts", "tokenizer_transformer.json")
WEIGHTS_PATH   = os.path.join("checkpoints", "transformer", "transformer_best.weights.h5")
DATA_PATH      = os.path.join("src", "data", "por.txt")

print("TensorFlow:", tf.__version__)


# ---------------- model (identical to the notebook) ----------------
def positional_encoding(length, depth):
    half = depth // 2
    positions = np.arange(length)[:, None]
    depths = np.arange(half)[None, :] / half
    angle = positions * (1.0 / (10000.0 ** depths))
    pos = np.concatenate([np.sin(angle), np.cos(angle)], axis=-1)
    return tf.cast(pos[None, ...], tf.float32)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kw):
        super().__init__(**kw)
        assert d_model % num_heads == 0
        self.d_model, self.num_heads = d_model, num_heads
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.out = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, b):
        x = tf.reshape(x, (b, -1, self.num_heads, self.depth))
        return tf.transpose(x, [0, 2, 1, 3])

    def call(self, query, key, value, attn_mask=None):
        b = tf.shape(query)[0]
        q = self.split_heads(self.wq(query), b)
        k = self.split_heads(self.wk(key), b)
        v = self.split_heads(self.wv(value), b)
        logits = tf.matmul(q, k, transpose_b=True)
        logits = logits / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        if attn_mask is not None:
            logits += attn_mask * -1e9
        weights = tf.nn.softmax(logits, axis=-1)
        ctx = tf.matmul(weights, v)
        ctx = tf.transpose(ctx, [0, 2, 1, 3])
        ctx = tf.reshape(ctx, (b, -1, self.d_model))
        return self.out(ctx), weights


def feed_forward(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation="relu"),
        tf.keras.layers.Dense(d_model),
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kw):
        super().__init__(**kw)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward(d_model, dff)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(rate)
        self.drop2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False, padding_mask=None):
        a, _ = self.mha(x, x, x, attn_mask=padding_mask)
        o1 = self.ln1(x + self.drop1(a, training=training))
        f = self.ffn(o1)
        return self.ln2(o1 + self.drop2(f, training=training))


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 vocab_size, max_len, rate=0.1, **kw):
        super().__init__(**kw)
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos = positional_encoding(max_len, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.drop = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False, padding_mask=None):
        L = tf.shape(x)[1]
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos[:, :L, :]
        x = self.drop(x, training=training)
        for layer in self.enc_layers:
            x = layer(x, training=training, padding_mask=padding_mask)
        return x


def look_ahead_mask(size):
    return 1.0 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kw):
        super().__init__(**kw)
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward(d_model, dff)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(rate)
        self.drop2 = tf.keras.layers.Dropout(rate)
        self.drop3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_out, training=False,
             combined_mask=None, padding_mask=None):
        a1, _ = self.mha1(x, x, x, attn_mask=combined_mask)
        o1 = self.ln1(x + self.drop1(a1, training=training))
        a2, attn = self.mha2(o1, enc_out, enc_out, attn_mask=padding_mask)
        o2 = self.ln2(o1 + self.drop2(a2, training=training))
        f = self.ffn(o2)
        o3 = self.ln3(o2 + self.drop3(f, training=training))
        return o3, attn


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 vocab_size, max_len, rate=0.1, **kw):
        super().__init__(**kw)
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos = positional_encoding(max_len, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.drop = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_out, training=False,
             combined_mask=None, padding_mask=None):
        L = tf.shape(x)[1]
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos[:, :L, :]
        x = self.drop(x, training=training)
        attn = None
        for layer in self.dec_layers:
            x, attn = layer(x, enc_out, training=training,
                            combined_mask=combined_mask, padding_mask=padding_mask)
        return x, attn


class Transformer(tf.keras.Model):
    def __init__(self, vocab_size, num_layers, d_model, num_heads,
                 dff, max_len, rate=0.1, **kw):
        super().__init__(**kw)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               vocab_size, max_len, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               vocab_size, max_len, rate)
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    @staticmethod
    def padding_mask(seq):
        m = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return m[:, tf.newaxis, tf.newaxis, :]

    def encode(self, enc_in, training=False):
        pm = self.padding_mask(enc_in)
        return self.encoder(enc_in, training=training, padding_mask=pm), pm

    def decode(self, dec_in, enc_out, enc_pad_mask, training=False):
        L = tf.shape(dec_in)[1]
        combined = tf.maximum(look_ahead_mask(L), self.padding_mask(dec_in))
        dec_out, attn = self.decoder(dec_in, enc_out, training=training,
                                     combined_mask=combined,
                                     padding_mask=enc_pad_mask)
        return self.final_layer(dec_out), attn

    def call(self, inputs, training=False):
        enc_out, pm = self.encode(inputs["encoder_inputs"], training=training)
        logits, _ = self.decode(inputs["decoder_inputs"], enc_out, pm,
                                training=training)
        return logits


# ---------------- load ----------------
assert os.path.exists(TOKENIZER_PATH), f"missing {TOKENIZER_PATH}"
assert os.path.exists(WEIGHTS_PATH), f"missing {WEIGHTS_PATH}"

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.get_vocab_size()
PAD_ID   = tokenizer.token_to_id("[PAD]")
START_ID = tokenizer.token_to_id("[START]")
END_ID   = tokenizer.token_to_id("[END]")

model = Transformer(VOCAB_SIZE, NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, MAX_LENGTH)
model({"encoder_inputs": tf.zeros((1, MAX_LENGTH), tf.int32),
       "decoder_inputs": tf.zeros((1, MAX_LENGTH - 1), tf.int32)})
model.load_weights(WEIGHTS_PATH)
print(f"Loaded weights. vocab={VOCAB_SIZE}")

# sanity: a correctly loaded model gives low loss on a known sentence pair
# (high loss here => version mismatch, same failure mode as the old checkpoint)


def encode_sentence(s):
    ids = tokenizer.encode(s).ids[:MAX_LENGTH]
    return tf.constant([ids + [PAD_ID] * (MAX_LENGTH - len(ids))], tf.int32)


def translate(sentence, max_len=MAX_LENGTH):
    enc_out, pm = model.encode(encode_sentence(sentence), training=False)
    out = [START_ID]
    for _ in range(max_len):
        logits, _ = model.decode(tf.constant([out], tf.int32), enc_out, pm,
                                 training=False)
        nxt = int(tf.argmax(logits[0, -1]))
        if nxt == END_ID:
            break
        out.append(nxt)
    return tokenizer.decode(out[1:], skip_special_tokens=True).strip()


def load_test_pairs():
    # reproduce the notebook's seed-42 split -> identical held-out test set
    import random
    pairs = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            en, pt = parts[0].strip(), parts[1].strip()
            if en and pt:
                pairs.append((pt, en))
    random.seed(42)
    random.shuffle(pairs)
    n = len(pairs)
    n_tr, n_val = int(0.8 * n), int(0.1 * n)
    return pairs[n_tr + n_val:]


if __name__ == "__main__":
    import sys

    demo = ["Eu gosto de futebol.", "Ela está estudando.", "Você fala inglês?",
            "Onde está meu carro?", "Eu não entendo.", "Você pode me ajudar?",
            "Nós vamos à escola.", "Ela gosta de música."]
    print("\n=== demo translations ===")
    for s in demo:
        print(f"PT: {s}\nEN: {translate(s)}\n" + "-" * 45)

    # ---- BLEU + chrF on held-out test sentences ----
    if os.path.exists(DATA_PATH):
        import sacrebleu
        N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
        test_pairs = load_test_pairs()
        sample = test_pairs[:N]
        print(f"\n=== BLEU on {len(sample)} held-out test sentences "
              f"(of {len(test_pairs)}) ===")
        hyps = [translate(pt) for pt, _ in sample]
        refs = [[en for _, en in sample]]
        bleu = sacrebleu.corpus_bleu(hyps, refs)
        chrf = sacrebleu.corpus_chrf(hyps, refs)
        print("BLEU:", round(bleu.score, 2), "| chrF:", round(chrf.score, 2))
        print("\nsamples:")
        for (pt, en), h in list(zip(sample, hyps))[:10]:
            print(f"PT : {pt}\nREF: {en}\nHYP: {h}\n" + "-" * 45)
    else:
        print(f"(skipping BLEU: {DATA_PATH} not found)")

    # ---- interactive mode only when attached to a terminal ----
    if sys.stdin.isatty():
        print("\nType Portuguese sentences (blank line to quit):")
        while True:
            try:
                s = input("PT> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not s:
                break
            print("EN>", translate(s))
