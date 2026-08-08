Shakespeare LSTM Text Generator

A word-level language model built with Python and TensorFlow/Keras to generate Shakespeare-style text using an LSTM neural network.

This project demonstrates an end-to-end Generative AI / NLP text-generation pipeline:

Shakespeare Corpus
       ↓
Text Cleaning
       ↓
Word Tokenization
       ↓
Vocabulary Construction
       ↓
Input / Next-Word Sequences
       ↓
Embedding
       ↓
LSTM
       ↓
Dropout
       ↓
Dense + Softmax
       ↓
Temperature + Top-K Sampling
       ↓
Generated Text

1. Project Objective

The objective is to build a generative language model that learns patterns from Shakespeare's text and generates new text based on a user-provided seed phrase.

Example:

Seed:
to be or not to be

The model predicts one word at a time. Each predicted word is added to the context and used to predict the following word.

2. Dataset

The project uses The Complete Works of William Shakespeare from Project Gutenberg, eBook #100.

Dataset page:

https://www.gutenberg.org/ebooks/100

The dataset is downloaded automatically by main.py and stored locally as:

data/shakespeare.txt

The dataset file is intentionally not committed to GitHub because it can be downloaded automatically when the project is executed.

3. Project Structure

shakespeare-lstm-text-generator/
│
├── main.py
│       Main training pipeline
│
├── generate_compare.py
│       Inference script for comparing
│       temperature and Top-K sampling
│
├── experiment_seq30.py
│       Controlled sequence-length experiment
│
├── requirements.txt
│       Python dependencies
│
├── README.md
│
├── .gitignore
│
├── data/
│   └── shakespeare.txt
│       Downloaded locally; not committed
│
└── artifacts/
    ├── best_model.keras
    │       Best model checkpoint
    │
    ├── vocab.json
    │       Saved vocabulary mappings
    │
    ├── metrics.json
    │       Training metrics
    │
    └── sample_outputs.txt
            Generated text samples

4. Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

LSTM

Word-level language modeling

5. Setup

Recommended:

Python 3.10 or Python 3.11

Create virtual environment

Windows:

python -m venv venv
venv\Scripts\activate

Install dependencies

pip install -r requirements.txt

Run the complete training pipeline

python main.py

The script automatically:

Downloads the dataset if it does not already exist.

Cleans and tokenizes the text.

Builds the vocabulary.

Creates input/target sequences.

Splits the data into training and validation sets.

Builds and trains the LSTM model.

Uses early stopping and model checkpointing.

Saves the best model.

Generates text from multiple seed phrases.

Saves vocabulary, metrics and generated outputs.

6. Text Preprocessing

The raw Shakespeare text is processed using:

Lowercasing

To Be, Or Not To Be

becomes:

to be or not to be

Punctuation removal

Punctuation and non-alphabetic characters are removed.

Tokenization

The cleaned text is split into individual words:

["to", "be", "or", "not", "to", "be"]

7. Vocabulary Construction

The project uses approximately 8,000 tokens.

The most frequent words are assigned integer IDs.

An <UNK> token is reserved for words outside the retained vocabulary.

This keeps the final Dense + Softmax layer practical for CPU-based training.

8. Input / Target Sequence Creation

The baseline model uses a context length of:

20 words

The model performs next-word prediction:

previous 20 words → next word

During generation, the predicted word is appended to the sequence and used as part of the next prediction.

9. Model Architecture

Input
20 word IDs
     ↓
Embedding
128-dimensional word representation
     ↓
LSTM
128 units
     ↓
Dropout
0.20
     ↓
Dense
8,000 output classes
     ↓
Softmax
     ↓
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

10. Why These Choices?

Word-level tokenization: directly demonstrates next-word generation and is easy to explain.

Sequence length = 20: provides useful context while keeping CPU training practical.

8,000-word vocabulary: keeps the final Dense layer practical on a laptop.

128-dimensional embedding: gives each word a learned vector representation.

128 LSTM units: provides reasonable capacity for a small language model.

Dropout = 0.20: provides regularization.

Adam: suitable adaptive optimizer for this task.

Sparse categorical cross-entropy: appropriate because the target is an integer token ID.

11. Training and Validation

The dataset is divided chronologically:

90% → Training
10% → Validation

A chronological split is used rather than randomly shuffling the text.

Dataset statistics

Tokens used:       120,000
Training examples: 53,991
Validation examples: 5,999

12. Early Stopping and Model Checkpointing

The model monitors validation loss with:

patience = 2

The best model is saved to:

artifacts/best_model.keras

In the baseline run, validation loss improved through Epoch 5 and then worsened in Epochs 6 and 7. Training therefore stopped at Epoch 7 and restored the best weights from Epoch 5.

13. Training Results

Baseline model

Epoch

Training Loss

Validation Loss

1

6.7999

6.7233

2

6.4231

6.6727

3

6.2519

6.6217

4

6.0962

6.5982

5

5.9467

6.5899

6

5.8055

6.6084

7

5.6814

6.6365

Best validation loss: 6.5899Best epoch: 5Early stopping: Epoch 7

The increasing validation loss while training loss continued decreasing is consistent with the model beginning to overfit the training data.

14. Text Generation

Generation is performed iteratively:

Seed text
    ↓
Convert words to IDs
    ↓
Predict next-word probabilities
    ↓
Apply temperature
    ↓
Apply Top-K filtering
    ↓
Sample next word
    ↓
Append word to context
    ↓
Repeat

15. Temperature Sampling

Temperature controls randomness.

Temperature

Expected behavior

0.5

More conservative / predictable

0.8

Balanced

1.0

More diverse

16. Top-K Sampling

Top-K sampling restricts the next-word choice to the K most probable candidates.

The generation comparison uses:

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

Run:

python generate_compare.py

17. Sample Generated Text

Seed: to be or not to be

to be or not to be thy feasting steal with the own your tune is purse
the reed murd most lafew of the dinner of a third love in i well
farewell in my break the country and and convey frederick was for the
story for

Seed: shall i compare thee

shall i compare thee to this the conspire enter as thy power love and
undertake so asked like his tiber that in the palace as my worship a
hand is and shame and beloving thy art <UNK> ay no rather seen i are you

Seed: love is

love is thank he have the room of thee well agrippa s my majesty look
thou please it no lord the man i ll her let if you will mine s it is you
rossillon say at make mine is menecrates be

Complete outputs are saved in:

artifacts/sample_outputs.txt

18. Controlled Experiment — Sequence Length

A controlled experiment tested whether increasing the context window improves next-word prediction.

Baseline

Sequence length = 20
Best validation loss = 6.5899
Best epoch = 5

Experiment

Sequence length = 30
Best validation loss = 6.5823
Best epoch = 5

Comparison

Model

Sequence Length

Best Validation Loss

Best Epoch

Baseline

20

6.5899

5

Experiment

30

6.5823

5

The 30-word context produced a small improvement in validation loss:

6.5899 → 6.5823

Only the sequence length was changed; the major model and training parameters were kept the same.

Reproduce the experiment with:

python experiment_seq30.py

19. Model Artifacts

Best model

artifacts/best_model.keras

Best validation-loss checkpoint.

Vocabulary

artifacts/vocab.json

Word-to-ID and ID-to-word mappings.

Metrics

artifacts/metrics.json

Training and validation metrics.

Generated samples

artifacts/sample_outputs.txt

Generated text from multiple seeds.

20. Limitations

This is an educational LSTM language model rather than a large-scale modern language model.

Training uses a limited subset of the Shakespeare corpus to keep CPU training practical.

LSTMs have limited long-range context compared with Transformer-based language models.

Generated text can contain grammatical inconsistencies.

<UNK> may appear for words outside the retained vocabulary.

The model may repeat words or produce semantically inconsistent sequences.

Generation quality depends on vocabulary size, sequence length, model capacity and training time.

21. Future Improvements

Possible improvements include:

Train on the complete Shakespeare corpus.

Increase vocabulary size.

Increase context length.

Compare LSTM with GRU.

Experiment with multiple LSTM layers.

Add Top-P / nucleus sampling.

Experiment with beam search.

Compare the LSTM approach with a small Transformer language model.

Train using GPU acceleration for larger datasets.

22. Reproducibility

A fixed random seed is used:

SEED = 42

The vocabulary and model artifacts are also saved to make inference reproducible.

23. Interview Takeaways

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

Controlled model experimentation

Reproducible inference

24. Quick Start

git clone <repository-url>

cd shakespeare-lstm-text-generator

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python main.py

Generation comparison:

python generate_compare.py

Sequence-length experiment:

python experiment_seq30.py

25. Conclusion

This project implements an end-to-end word-level generative language model using an LSTM.

The baseline achieved a best validation loss of 6.5899, while the controlled 30-word context experiment achieved 6.5823.

The project also demonstrates controlled text generation using temperature and Top-K sampling, together with early stopping and best-model checkpointing.

The goal is to demonstrate the fundamental principles of neural language modeling and autoregressive text generation using LSTMs rather than reproduce the scale or capabilities of modern Transformer-based LLMs.