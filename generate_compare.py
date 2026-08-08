from pathlib import Path
import json
import re

import numpy as np
import tensorflow as tf


MODEL_FILE = Path("artifacts/best_model.keras")
VOCAB_FILE = Path("artifacts/vocab.json")

SEQ_LEN = 20
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_artifacts():
    model = tf.keras.models.load_model(MODEL_FILE)

    vocab = json.loads(
        VOCAB_FILE.read_text(encoding="utf-8")
    )

    word_to_id = vocab["word_to_id"]
    id_to_word = {
        int(k): v
        for k, v in vocab["id_to_word"].items()
    }

    return model, word_to_id, id_to_word


def generate_text(
    model,
    seed_text,
    word_to_id,
    id_to_word,
    num_words=40,
    temperature=0.8,
    top_k=20,
):
    seed_tokens = re.sub(
        r"[^a-z\s]",
        " ",
        seed_text.lower()
    ).split()

    if not seed_tokens:
        raise ValueError(
            "Seed text must contain at least one word."
        )

    context = [
        word_to_id.get(word, 0)
        for word in seed_tokens
    ]

    generated = list(seed_tokens)

    for _ in range(num_words):

        context_ids = context[-SEQ_LEN:]

        if len(context_ids) < SEQ_LEN:
            context_ids = (
                [0] * (SEQ_LEN - len(context_ids))
                + context_ids
            )

        x = np.array(
            [context_ids],
            dtype=np.int32
        )

        probabilities = model.predict(
            x,
            verbose=0
        )[0]

        probabilities = np.asarray(
            probabilities
        ).astype("float64")

        # Temperature sampling
        probabilities = (
            np.log(
                np.maximum(
                    probabilities,
                    1e-9
                )
            ) / temperature
        )

        probabilities = np.exp(
            probabilities
            - np.max(probabilities)
        )

        probabilities /= probabilities.sum()

        # Top-K sampling
        if top_k is not None and top_k > 0:

            top_indices = np.argsort(
                probabilities
            )[-top_k:]

            top_probabilities = (
                probabilities[top_indices]
            )

            top_probabilities /= (
                top_probabilities.sum()
            )

            next_id = np.random.choice(
                top_indices,
                p=top_probabilities
            )

        else:

            next_id = np.random.choice(
                len(probabilities),
                p=probabilities
            )

        next_word = id_to_word.get(
            int(next_id),
            "<UNK>"
        )

        context.append(int(next_id))
        generated.append(next_word)

    return " ".join(generated)


def main():

    model, word_to_id, id_to_word = (
        load_artifacts()
    )

    seed = "to be or not to be"

    settings = [
        {
            "temperature": 0.5,
            "top_k": 10,
        },
        {
            "temperature": 0.8,
            "top_k": 20,
        },
        {
            "temperature": 1.0,
            "top_k": 40,
        },
    ]

    print("\n" + "=" * 70)
    print("TEMPERATURE + TOP-K GENERATION COMPARISON")
    print("=" * 70)

    for setting in settings:

        temperature = setting["temperature"]
        top_k = setting["top_k"]

        print(
            f"\nTemperature: {temperature} | "
            f"Top-K: {top_k}"
        )

        generated = generate_text(
            model,
            seed,
            word_to_id,
            id_to_word,
            num_words=40,
            temperature=temperature,
            top_k=top_k,
        )

        print(generated)


if __name__ == "__main__":
    main()