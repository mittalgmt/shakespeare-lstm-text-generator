# Shakespeare LSTM Text Generator

An interview-task implementation of **Generative AI with LSTM – Text Generation** using TensorFlow/Keras.

## 1. Task requirements covered

- Public-domain Shakespeare text dataset
- Lowercasing and punctuation removal
- Word tokenization
- Input/output next-token pairs
- Embedding + LSTM + Dense softmax architecture
- Categorical next-word prediction using sparse categorical cross-entropy
- Adam optimizer
- Training/validation split
- Early stopping
- Best-model checkpoint
- Seed-based iterative text generation
- Multiple sample outputs
- Reproducible vocabulary and metrics

## 2. Project structure

```text
shakespeare-lstm-text-generator/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── shakespeare.txt        # downloaded locally; not committed
└── artifacts/
    ├── best_model.keras
    ├── vocab.json
    ├── metrics.json
    └── sample_outputs.txt
```

## 3. Dataset

The dataset is **The Complete Works of William Shakespeare** from Project Gutenberg, eBook #100.

Dataset page:
https://www.gutenberg.org/ebooks/100

The script downloads the plain-text file automatically. The dataset is not committed to GitHub because it is unnecessary for the code repository.

## 4. Setup

Recommended: Python 3.10 or 3.11.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### If `python` is not recognized

Try:

```bash
py -m venv venv
venv\Scripts\activate
py -m pip install -r requirements.txt
py main.py
```

## 5. Model architecture

```text
Input: 20 word IDs
        ↓
Embedding(8,000 vocabulary, 128 dimensions)
        ↓
LSTM(128 units)
        ↓
Dropout(0.20)
        ↓
Dense(8,000, softmax)
        ↓
Probability of the next word
```

The model learns:

```text
input words → next word
```

Example:

```text
Input:  to be or not to
Target: be
```

During generation, the predicted word is appended to the context and becomes part of the next input.

## 6. Why these choices?

- **Word-level tokenization:** directly demonstrates next-word generation and is easy to explain in an interview.
- **20-word sequence:** gives the LSTM enough context without making training unnecessarily slow.
- **8,000-word vocabulary:** keeps the final Dense layer practical on a laptop.
- **128-dimensional embedding:** gives each word a learned vector representation.
- **128 LSTM units:** enough capacity for a small interview project.
- **Adam:** reliable default optimizer for this task.
- **Sparse categorical cross-entropy:** appropriate because the target is one integer token ID.
- **EarlyStopping:** stops when validation loss stops improving.
- **ModelCheckpoint:** keeps the best model instead of blindly using the final epoch.
- **Temperature:** controls generation randomness.

## 7. Results

After training, `artifacts/metrics.json` records the best validation loss, number of epochs completed, vocabulary size and training/validation example counts.

`artifacts/sample_outputs.txt` contains generated text from three different seed inputs.

## 8. Bonus experiment

If time remains, change:

```python
LSTM_UNITS = 128
```

to:

```python
LSTM_UNITS = 256
```

or add a second LSTM layer using `return_sequences=True`.

Compare validation loss and sample quality and document the result in the README.

## 9. Limitations

- The model was trained on a limited subset of the Shakespeare corpus
  to keep training time practical.
- Generated text may contain grammatically inconsistent sequences.
- `<UNK>` can appear when a word is outside the retained vocabulary.
- LSTM generation quality depends strongly on dataset size, sequence
  length, vocabulary size, and training time.




## Results

The model was trained on 120,000 Shakespeare tokens using a word-level
LSTM language model.

### Training Results

- Training examples: 53,991
- Validation examples: 5,999
- Best validation loss: 6.5899
- Best epoch: 5
- Early stopping: Yes

### Sample Generated Text

#### Seed: `to be or not to be`

> to be or not to be thy feasting steal with the own your tune is purse the reed murd most lafew of the dinner of a third love in i well farewell in my break the country and and convey frederick was for the story for

#### Seed: `shall i compare thee`

> shall i compare thee to this the conspire enter as thy power love and undertake so asked like his tiber that in the palace as my worship a hand is and shame and beloving thy art <UNK> ay no rather seen i are you

#### Seed: `love is`

> love is thank he have the room of thee well agrippa s my majesty look thou please it no lord the man i ll her let if you will mine s it is you rossillon say at make mine is menecrates be
