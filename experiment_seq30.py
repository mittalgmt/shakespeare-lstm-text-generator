from pathlib import Path
import re
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# -------------------- Configuration --------------------

SEED = 42

DATA_FILE = Path("data/shakespeare.txt")

MAX_WORDS = 120_000

# EXPERIMENT: changed from 20 to 30
SEQ_LEN = 30

VOCAB_SIZE = 8_000
STEP = 2

EMBED_DIM = 128
LSTM_UNITS = 128

BATCH_SIZE = 128
EPOCHS = 12

np.random.seed(SEED)
tf.random.set_seed(SEED)


# -------------------- Data --------------------

def clean_and_tokenize(text):

    start = text.find("*** START OF")
    end = text.find("*** END OF")

    if start != -1:
        text = text[start:]

    if end != -1:
        text = text[:text.find("*** END OF")]

    text = text.lower()

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    tokens = text.split()

    return tokens[:MAX_WORDS]


def build_vocabulary(tokens):

    counts = Counter(tokens)

    most_common = [
        word
        for word, _
        in counts.most_common(VOCAB_SIZE - 1)
    ]

    word_to_id = {
        word: idx + 1
        for idx, word in enumerate(most_common)
    }

    encoded = np.array(
        [
            word_to_id.get(word, 0)
            for word in tokens
        ],
        dtype=np.int32,
    )

    return word_to_id, encoded


def make_sequences(encoded):

    starts = range(
        0,
        len(encoded) - SEQ_LEN,
        STEP,
    )

    x = np.array(
        [
            encoded[i:i + SEQ_LEN]
            for i in starts
        ],
        dtype=np.int32,
    )

    y = np.array(
        [
            encoded[i + SEQ_LEN]
            for i in starts
        ],
        dtype=np.int32,
    )

    return x, y


# -------------------- Model --------------------

def build_model(vocab_size):

    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBED_DIM,
        ),

        LSTM(LSTM_UNITS),

        Dropout(0.2),

        Dense(
            vocab_size,
            activation="softmax",
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "sparse_categorical_accuracy"
        ],
    )

    return model


# -------------------- Main --------------------

def main():

    text = DATA_FILE.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tokens = clean_and_tokenize(text)

    print(
        f"[INFO] Tokens used: {len(tokens):,}"
    )

    word_to_id, encoded = build_vocabulary(
        tokens
    )

    print(
        f"[INFO] Vocabulary size: "
        f"{len(word_to_id) + 1:,}"
    )

    x, y = make_sequences(encoded)

    print(
        f"[INFO] Sequence length: {SEQ_LEN}"
    )

    print(
        f"[INFO] Training examples: "
        f"{len(x):,}"
    )

    split = int(len(x) * 0.90)

    x_train = x[:split]
    x_val = x[split:]

    y_train = y[:split]
    y_val = y[split:]

    print(
        f"[INFO] Train examples: "
        f"{len(x_train):,}"
    )

    print(
        f"[INFO] Validation examples: "
        f"{len(x_val):,}"
    )

    model = build_model(
        len(word_to_id) + 1
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    print(
        "\n[INFO] Training sequence-length "
        "30 experiment..."
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(
            x_val,
            y_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    best_val_loss = min(
        history.history["val_loss"]
    )

    best_epoch = (
        np.argmin(
            history.history["val_loss"]
        ) + 1
    )

    print("\n" + "=" * 50)
    print("SEQUENCE LENGTH EXPERIMENT RESULTS")
    print("=" * 50)

    print(
        f"Sequence length: {SEQ_LEN}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Epochs completed: "
        f"{len(history.history['loss'])}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()