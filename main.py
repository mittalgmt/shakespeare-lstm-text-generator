"""
Shakespeare LSTM Text Generator
Interview Task: Generative AI with LSTM - Text Generation

Pipeline:
1. Download Shakespeare text from Project Gutenberg.
2. Clean/lowercase/tokenize the corpus.
3. Build a limited word vocabulary.
4. Create input -> next-word training pairs.
5. Train an LSTM language model with validation + callbacks.
6. Generate text from multiple seed inputs.
7. Save the model, vocabulary and sample outputs.
"""

from pathlib import Path
import json
import re
import urllib.request
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------- Configuration --------------------
SEED = 42
DATA_URL = "https://www.gutenberg.org/files/100/100-0.txt"
DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")
RAW_FILE = DATA_DIR / "shakespeare.txt"
MODEL_FILE = ARTIFACT_DIR / "best_model.keras"
VOCAB_FILE = ARTIFACT_DIR / "vocab.json"
OUTPUT_FILE = ARTIFACT_DIR / "sample_outputs.txt"

MAX_WORDS = 120_000       # Keep the 2-hour assignment practical.
SEQ_LEN = 20              # Number of context words.
VOCAB_SIZE = 8_000        # Most frequent words + <UNK>.
STEP = 2                  # Reduces overlapping training examples.
EMBED_DIM = 128
LSTM_UNITS = 128
BATCH_SIZE = 128
EPOCHS = 12

np.random.seed(SEED)
tf.random.set_seed(SEED)


# -------------------- Data --------------------
def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    if RAW_FILE.exists():
        print(f"[INFO] Dataset already exists: {RAW_FILE}")
        return

    print("[INFO] Downloading Shakespeare dataset...")
    try:
        urllib.request.urlretrieve(DATA_URL, RAW_FILE)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the dataset. Open Project Gutenberg and download "
            "the plain-text file as data/shakespeare.txt, then run again."
        ) from exc

    print(f"[INFO] Saved dataset to {RAW_FILE}")


def clean_and_tokenize(text):
    # Remove Project Gutenberg header/footer.
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start != -1:
        text = text[start:]
    if end != -1:
        text = text[: text.find("*** END OF")]

    # Requirement: lowercase + remove punctuation.
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    return tokens[:MAX_WORDS]


def build_vocabulary(tokens):
    counts = Counter(tokens)

    # ID 0 is reserved for unknown words.
    most_common = [word for word, _ in counts.most_common(VOCAB_SIZE - 1)]
    word_to_id = {word: idx + 1 for idx, word in enumerate(most_common)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    id_to_word[0] = "<UNK>"

    encoded = np.array([word_to_id.get(word, 0) for word in tokens], dtype=np.int32)
    return word_to_id, id_to_word, encoded


def make_sequences(encoded):
    # Each X is 20 words; y is the next word.
    starts = range(0, len(encoded) - SEQ_LEN, STEP)
    x = np.array([encoded[i:i + SEQ_LEN] for i in starts], dtype=np.int32)
    y = np.array([encoded[i + SEQ_LEN] for i in starts], dtype=np.int32)
    return x, y


# -------------------- Model --------------------
def build_model(vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBED_DIM),
        LSTM(LSTM_UNITS),
        Dropout(0.2),
        Dense(vocab_size, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


# -------------------- Text generation --------------------
def generate_text(model, seed_text, word_to_id, id_to_word, num_words=40, temperature=0.8):
    seed_tokens = re.sub(r"[^a-z\s]", " ", seed_text.lower()).split()
    if not seed_tokens:
        raise ValueError("Seed text must contain at least one word.")

    # Convert seed words to IDs; unseen words become <UNK>.
    context = [word_to_id.get(word, 0) for word in seed_tokens]
    generated = list(seed_tokens)

    for _ in range(num_words):
        context_ids = context[-SEQ_LEN:]
        if len(context_ids) < SEQ_LEN:
            context_ids = [0] * (SEQ_LEN - len(context_ids)) + context_ids

        x = np.array([context_ids], dtype=np.int32)
        probabilities = model.predict(x, verbose=0)[0]

        # Temperature controls randomness:
        # lower -> safer/more predictable, higher -> more diverse.
        probabilities = np.asarray(probabilities).astype("float64")
        probabilities = np.log(np.maximum(probabilities, 1e-9)) / temperature
        probabilities = np.exp(probabilities - np.max(probabilities))
        probabilities /= probabilities.sum()

        next_id = np.random.choice(len(probabilities), p=probabilities)
        next_word = id_to_word.get(int(next_id), "<UNK>")

        context.append(int(next_id))
        generated.append(next_word)

    return " ".join(generated)


# -------------------- Main pipeline --------------------
def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    download_dataset()

    raw_text = RAW_FILE.read_text(encoding="utf-8", errors="ignore")
    tokens = clean_and_tokenize(raw_text)

    if len(tokens) <= SEQ_LEN + 100:
        raise ValueError("Dataset is too small after preprocessing.")

    print(f"[INFO] Tokens used: {len(tokens):,}")

    word_to_id, id_to_word, encoded = build_vocabulary(tokens)
    print(f"[INFO] Vocabulary size: {len(word_to_id):,}")

    x, y = make_sequences(encoded)
    print(f"[INFO] Training examples: {len(x):,}")

    # Chronological split: avoids leaking later text into earlier training.
    split = int(len(x) * 0.90)
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"[INFO] Train examples: {len(x_train):,}")
    print(f"[INFO] Validation examples: {len(x_val):,}")

    model = build_model(len(word_to_id) + 1)
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_FILE,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("[INFO] Starting training...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Save vocabulary for reproducible generation.
    VOCAB_FILE.write_text(
        json.dumps(
            {"word_to_id": word_to_id, "id_to_word": {str(k): v for k, v in id_to_word.items()}},
            indent=2,
        ),
        encoding="utf-8",
    )

    # Save metrics.
    metrics = {
        "best_val_loss": float(min(history.history["val_loss"])),
        "final_train_loss": float(history.history["loss"][-1]),
        "epochs_completed": len(history.history["loss"]),
        "vocab_size": len(word_to_id) + 1,
        "train_examples": len(x_train),
        "validation_examples": len(x_val),
    }
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Required deliverable: sample outputs from different seeds.
    seeds = [
        "to be or not to be",
        "shall i compare thee",
        "love is",
    ]

    outputs = []
    print("\n===== GENERATED TEXT =====")
    for seed in seeds:
        generated = generate_text(
            model, seed, word_to_id, id_to_word, num_words=40, temperature=0.8
        )
        outputs.append(f"Seed: {seed}\n{generated}\n")
        print(f"\nSeed: {seed}\n{generated}")

    OUTPUT_FILE.write_text("\n".join(outputs), encoding="utf-8")

    print("\n[OK] Task completed.")
    print(f"[OK] Model: {MODEL_FILE}")
    print(f"[OK] Vocabulary: {VOCAB_FILE}")
    print(f"[OK] Outputs: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
