Shakespeare LSTM Text Generator

<p align="center">
  <b>Word-Level Language Modeling with LSTM + Temperature + Top-K Sampling</b>
</p>

<p align="center">
  An end-to-end Generative AI / NLP project that learns Shakespeare-style
  next-word patterns and generates text from user-provided seed phrases.
</p>

Overview

This project implements a word-level LSTM language model usingTensorFlow/Keras.

The complete pipeline is:

Shakespeare Corpus
        │
        ▼
Text Cleaning
        │
        ▼
Word Tokenization
        │
        ▼
Vocabulary Construction
        │
        ▼
Input → Next-Word Training Pairs
        │
        ▼
Embedding
        │
        ▼
LSTM
        │
        ▼
Dropout
        │
        ▼
Dense + Softmax
        │
        ▼
Temperature + Top-K Sampling
        │
        ▼
Generated Shakespeare-Style Text

The project also includes:

Early stopping

Best-model checkpointing

Training/validation monitoring

Multiple seed-based generations

Temperature comparison

Top-K sampling

A controlled sequence-length experiment

Saved vocabulary and training metrics

Reproducible inference artifacts

Table of Contents

Project Objective

Dataset

Features

Project Structure

Tech Stack

Setup

How to Run

Data Preprocessing

Vocabulary

Training Data

Model Architecture

Training Configuration

Training Strategy

Training Results

Training Curve

Text Generation

Temperature and Top-K

Generated Samples

Controlled Experiment

Artifacts

Limitations

Future Improvements

Reproducibility

Interview Takeaways

Conclusion

Project Objective

The goal is to build a neural language model that learns word patterns fromShakespeare's works and generates new text one word at a time.

For example:

Seed:
to be or not to be

The model predicts a probability distribution over the vocabulary, selectsthe next word using the configured sampling strategy, appends that word tothe context, and repeats the process.

Seed
  │
  ▼
Predict next word
  │
  ▼
Append prediction
  │
  ▼
Predict next word
  │
  ▼
Repeat

Dataset

The project uses The Complete Works of William Shakespeare fromProject Gutenberg, eBook #100.

Dataset:https://www.gutenberg.org/ebooks/100

The training script automatically downloads the plain-text dataset andstores it locally as:

data/shakespeare.txt

The downloaded dataset is excluded from Git because it can be recreatedautomatically by running the project.

Dataset used for training

Tokens used:          120,000
Vocabulary size:        8,000
Training examples:     53,991
Validation examples:    5,999

Features

Feature

Implementation

Text preprocessing

Lowercase + punctuation removal

Tokenization

Word-level

Vocabulary

Top 8,000 words

Context window

20 words

Embedding

128 dimensions

LSTM

128 units

Regularization

Dropout 0.20

Optimizer

Adam

Loss

Sparse categorical cross-entropy

Validation

Chronological 90/10 split

Training control

Early stopping

Checkpointing

Best validation loss

Generation

Temperature sampling

Sampling

Top-K

Experiments

Sequence length 20 vs 30

Project Structure

shakespeare-lstm-text-generator/
│
├── main.py
│   └── Main training and generation pipeline
│
├── generate_compare.py
│   └── Temperature + Top-K generation comparison
│
├── experiment_seq30.py
│   └── Controlled sequence-length experiment
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│
├── .gitignore
│
├── data/
│   └── shakespeare.txt
│       └── Downloaded locally; ignored by Git
│
└── artifacts/
    ├── best_model.keras
    │   └── Best checkpoint; ignored by Git
    │
    ├── vocab.json
    │   └── Saved word ↔ ID mappings
    │
    ├── metrics.json
    │   └── Training metrics
    │
    ├── sample_outputs.txt
    │   └── Generated samples
    │
    └── training_curve.png
        └── Training vs validation loss

venv/, the downloaded dataset, and the .keras model checkpoint areintentionally excluded from version control.

Tech Stack

Python

TensorFlow / Keras

NumPy

Matplotlib

LSTM

Word-level language modeling

Setup

Requirements

Recommended:

Python 3.10 or Python 3.11

1. Clone the repository

git clone <repository-url>
cd shakespeare-lstm-text-generator

2. Create a virtual environment

Windows:

python -m venv venv
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

How to Run

Train the model

python main.py

The main pipeline:

Downloads the dataset if it is not already present.

Cleans and tokenizes the text.

Builds the vocabulary.

Creates input/target sequences.

Splits the data into training and validation sets.

Builds the LSTM model.

Trains with early stopping and checkpointing.

Saves metrics and vocabulary.

Generates sample text.

Saves the training curve and generated outputs.

Compare generation settings

python generate_compare.py

This compares:

Temperature 0.5 + Top-K 10
Temperature 0.8 + Top-K 20
Temperature 1.0 + Top-K 40

Run the sequence-length experiment

python experiment_seq30.py

This trains a second model with:

Sequence length = 30

while keeping the major model and training settings unchanged.

Data Preprocessing

The raw Shakespeare text goes through three main preprocessing steps.

1. Lowercasing

To Be, Or Not To Be

becomes:

to be or not to be

2. Punctuation removal

Non-alphabetic characters are removed so the model works with normalizedword tokens.

3. Word tokenization

The cleaned text is split into individual words:

["to", "be", "or", "not", "to", "be"]

Vocabulary

The project keeps approximately 8,000 of the most frequent words.

Each retained word is assigned an integer ID.

An <UNK> token is reserved for words outside the retained vocabulary.

Example:

word → integer ID

This vocabulary limit keeps the final Dense + Softmax layer practical forCPU-based training.

Training Data

The baseline model uses a 20-word context window.

Each training sample follows:

Previous 20 words → Next word

Example:

Input:
to be or not to be ...

Target:
next word

The dataset is split chronologically:

90% → Training
10% → Validation

A chronological split is used instead of randomly shuffling the sequencesto avoid mixing later text into earlier training examples.

Dataset statistics

Metric

Value

Tokens used

120,000

Vocabulary

8,000

Training examples

53,991

Validation examples

5,999

Context length

20

Model Architecture

Input
20 word IDs
     │
     ▼
Embedding
128-dimensional representation
     │
     ▼
LSTM
128 units
     │
     ▼
Dropout
0.20
     │
     ▼
Dense
8,000 output classes
     │
     ▼
Softmax
     │
     ▼
Next-word probability distribution

Configuration

Parameter

Value

Tokenization

Word-level

Vocabulary

8,000

Sequence length

20

Embedding dimension

128

LSTM units

128

Dropout

0.20

Optimizer

Adam

Learning rate

0.001

Batch size

128

Maximum epochs

12

Loss

Sparse categorical cross-entropy

Why These Choices?

Word-level tokenization

Word-level modeling directly demonstrates next-word generation and is easyto inspect during an interview.

20-word context

A 20-word context provides useful surrounding information while keepingCPU training practical.

8,000-word vocabulary

A limited vocabulary reduces the size of the final Softmax layer and keepstraining manageable on a laptop.

128-dimensional embedding

The embedding layer learns a dense vector representation for each word.

128 LSTM units

This provides reasonable capacity for a small educational language modelwithout making CPU training unnecessarily expensive.

Dropout 0.20

Dropout provides regularization and helps reduce overfitting.

Adam

Adam provides adaptive parameter updates and is a practical optimizer forthis language-modeling task.

Sparse categorical cross-entropy

The target is represented by an integer token ID, making sparse categoricalcross-entropy appropriate.

Training Strategy

The model uses:

Early stopping

Validation loss is monitored with:

patience = 2

Training stops when validation loss fails to improve for two consecutiveepochs.

Model checkpointing

The model with the best validation loss is saved as:

artifacts/best_model.keras

This prevents the final model from being worse than the best checkpoint.

Training Results

Final baseline run

The baseline configuration produced:

Epoch

Training Loss

Validation Loss

1

6.8004

6.7109

2

6.4184

6.6706

3

6.2592

6.6278

4

6.0989

6.5912

5

5.9377

6.5629

6

5.7855

6.5842

7

5.6453

6.6260

Best result

Best validation loss: 6.5629
Best epoch: 5
Early stopping: Epoch 7

The best validation loss occurred at Epoch 5. Validation loss then increasedduring Epochs 6 and 7 while training loss continued decreasing. Earlystopping therefore restored the Epoch 5 weights.

This behavior is consistent with the model beginning to overfit the trainingdata.

Training Curve

The baseline training run also saves the loss curve:



The graph shows the relationship between training loss and validation lossacross the completed epochs.

Text Generation

Generation is autoregressive:

Seed text
    │
    ▼
Convert words to IDs
    │
    ▼
Predict next-word probabilities
    │
    ▼
Apply temperature
    │
    ▼
Apply Top-K filtering
    │
    ▼
Sample next word
    │
    ▼
Append word to context
    │
    ▼
Repeat

The predicted word becomes part of the next input sequence.

Temperature and Top-K

Temperature

Temperature controls how concentrated or diverse the sampling distributionis.

Temperature

Behavior

0.5

More conservative / predictable

0.8

Balanced

1.0

More diverse

Top-K

Top-K sampling restricts the next-word choice to the K most probablecandidates instead of sampling from the complete vocabulary.

The comparison script uses:

Temperature

Top-K

Purpose

0.5

10

Conservative generation

0.8

20

Balanced generation

1.0

40

More diverse generation

Generated Samples

Seed: to be or not to be

to be or not to be thy lord to be and your world is my love as the man
for thy son that you do a love so so s be a house and that was i shall
thee to the man i think i will

Seed: shall i compare thee

shall i compare thee thou am thee to be i be thee in her and i am the
heart of ephesus of i be so but i know me the good man you have not have
that be to you be all you must

Seed: love is

love is the room in you in thy man s the enter of the heart of my own
from your father s that be the gods shall the way the way to the man s
the man to the hand

Complete generated outputs are saved to:

artifacts/sample_outputs.txt

The generated text is intentionally shown as raw model output. Somegrammatical inconsistency and repetition are expected from a relativelysmall word-level LSTM trained on a limited corpus.

Controlled Experiment — Sequence Length

A controlled experiment was performed to test whether increasing the contextwindow from 20 to 30 words improves next-word prediction.

Baseline

Sequence length = 20
Best validation loss = 6.5629
Best epoch = 5

Experiment

Sequence length = 30
Best validation loss = 6.5827
Best epoch = 5

Comparison

Model

Sequence Length

Best Validation Loss

Best Epoch

Baseline

20

6.5629

5

Experiment

30

6.5827

5

Result

The 20-word baseline performed slightly better:

6.5629 < 6.5827

Difference:

0.0198 validation loss

The experiment therefore did not show an improvement from increasing thecontext length to 30 words under the same training configuration.

Only the sequence length was changed; the vocabulary, embedding dimension,LSTM units, optimizer, learning rate, batch size, dropout and early stoppingsettings remained the same.

This is a useful result because a larger context window does notautomatically improve an LSTM. The additional context can increase thelearning difficulty without providing enough useful information to improvevalidation performance within the same training budget.

Reproduce the experiment with:

python experiment_seq30.py

Artifacts

The training pipeline produces the following files:

File

Purpose

artifacts/best_model.keras

Best validation-loss model checkpoint

artifacts/vocab.json

Word-to-ID and ID-to-word mappings

artifacts/metrics.json

Training metrics and dataset statistics

artifacts/sample_outputs.txt

Generated samples

artifacts/training_curve.png

Training vs validation loss

The trained .keras model is excluded from Git to keep the repositorylightweight.

Limitations

This is an educational LSTM language model rather than a large-scalemodern language model.

Training uses a limited subset of the Shakespeare corpus to keep CPUtraining practical.

LSTMs have less effective long-range context handling than modernTransformer-based language models.

Generated text can contain grammatical inconsistencies.

<UNK> can appear for words outside the retained vocabulary.

The model may repeat words or produce semantically inconsistentsequences.

Generation quality depends on vocabulary size, context length, modelcapacity and training time.

The model is trained for next-word prediction rather than semanticunderstanding.

Future Improvements

Potential improvements include:

Train on the complete Shakespeare corpus.

Increase vocabulary size.

Compare additional context lengths.

Compare LSTM with GRU.

Experiment with multiple LSTM layers.

Add Top-P / nucleus sampling.

Compare different sampling strategies quantitatively.

Compare the LSTM approach with a small Transformer language model.

Use GPU acceleration for larger datasets and experiments.

Reproducibility

A fixed random seed is used:

SEED = 42

The project also saves:

Vocabulary mappings

Training metrics

Generated outputs

Best model checkpoint

This makes the training and inference workflow easier to reproduce.

Interview Takeaways

This project demonstrates practical understanding of:

Text preprocessing

Word-level tokenization

Vocabulary construction

Sequence modeling

Embeddings

LSTM networks

Next-word prediction

Softmax probability distributions

Sparse categorical cross-entropy

Adam optimization

Train/validation splitting

Early stopping

Model checkpointing

Temperature sampling

Top-K sampling

Autoregressive generation

Controlled model experimentation

Reproducible inference

Quick Start

git clone <repository-url>
cd shakespeare-lstm-text-generator

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python main.py

Generation comparison

python generate_compare.py

Sequence-length experiment

python experiment_seq30.py

Conclusion

This project implements an end-to-end word-level generative languagemodel using an LSTM.

The final baseline achieved a best validation loss of 6.5629 with a20-word context. A controlled experiment using a 30-word context achieved6.5827, showing that the longer context did not improve validationperformance under the same training configuration.

The project also demonstrates practical generation controls throughtemperature and Top-K sampling, together with early stopping,best-model checkpointing, saved vocabulary, training metrics, andgeneration artifacts.

The goal is to demonstrate the core principles of neural language modelingand autoregressive text generation using LSTMs, rather than reproduce thescale or capabilities of modern Transformer-based large language models.