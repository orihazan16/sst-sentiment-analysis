# Sentiment Analysis on the Stanford Sentiment Treebank

A comparison of four approaches to binary sentiment classification, from a simple one-hot log-linear model to a fully fine-tuned Transformer: **one-hot log-linear**, **log-linear over frozen Transformer embeddings**, a **bidirectional LSTM** over frozen Transformer embeddings, and a **fine-tuned Transformer** (`distilroberta-base`). Built with PyTorch and Hugging Face `transformers` on the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/) (SST).

## Models

| # | Model | Input representation | Trainable parameters |
|---|-------|----------------------|------------------------|
| 1 | Log-linear, one-hot | Average of one-hot word vectors | Linear layer only |
| 2 | Log-linear, Transformer embeddings | Average of frozen `distilroberta-base` token embeddings | Linear layer only |
| 3 | Bi-LSTM | Sequence of frozen `distilroberta-base` token embeddings | Bi-LSTM + linear layer |
| 4 | Fine-tuned Transformer | `distilroberta-base`, end to end | Entire Transformer + linear layer |

All models are trained for binary sentiment classification (positive/negative; neutral phrases are excluded) with `BCEWithLogitsLoss` and the Adam optimizer.

## Project Structure

```
.
├── main.py              # Models, training loop, and experiments (Sections 6–9)
├── data_loader.py             # SentimentTreeBank parser and dataset utilities
└── stanfordSentimentTreebank/ # Raw SST data (see Dataset below)
    ├── datasetSentences.txt
    ├── datasetSplit.txt
    ├── dictionary.txt
    ├── sentiment_labels.txt
    ├── SOStr.txt
    ├── STree.txt
    └── original_rt_snippets.txt
```

## Dataset

The project uses the original **Stanford Sentiment Treebank** release:

- `SOStr.txt` / `STree.txt` — tokenized sentences and their constituency parse trees
- `dictionary.txt` / `sentiment_labels.txt` — all phrases (words, sub-phrases, full sentences) mapped to IDs and continuous sentiment scores in [0, 1]
- `datasetSentences.txt` / `datasetSplit.txt` — the original sentences and their train/val/test split
- `original_rt_snippets.txt` — the original Rotten Tomatoes review snippets the sentences were extracted from

`data_loader.py` reconstructs each sentence's parse tree as `SentimentTreeNode` objects, attaches the sentiment value and class (negative/neutral/positive, via thresholds at 0.4 and 0.6) to every node, and exposes:

- `get_train_set()` / `get_validation_set()` / `get_test_set()` — full sentences only
- `get_train_set_phrases()` — the train sentences **plus all their labeled sub-phrases**, used to augment training data
- `get_negated_polarity_examples()` — test sentences whose immediate sub-phrase sentiment disagrees with the sentence's own sentiment (a negation stress-test)
- `get_rare_words_examples()` — test sentences whose sentiment-bearing words are least frequent in the training set

Place the seven data files in a `stanfordSentimentTreebank/` directory next to the scripts (this is `data_loader.py`'s default path).

## Getting Started

```bash
git clone <repo-url>
cd <repo-name>
pip install torch transformers numpy matplotlib
```

Run any of the four experiments from `main.py`:

```python
from main import (
    train_log_linear_with_one_hot,
    train_log_linear_with_transformer,
    train_lstm_with_transformer,
    train_transformer,
)

train_log_linear_with_one_hot()        # Section 6
train_log_linear_with_transformer()    # Section 7
train_lstm_with_transformer()          # Section 8
train_transformer()                    # Section 9 (fine-tuning)
```

Each function trains the model, saves loss/accuracy curves as PNGs, and prints test-set loss/accuracy plus accuracy on the negated-polarity and rare-words subsets. The first run downloads `distilroberta-base` and caches its embedding matrix locally (`transformer_emb_cache.pkl`).

## Results

All numbers are test-set performance after training with the hyperparameters in `main.py` (10 epochs for the log-linear models, 4 for the Bi-LSTM, 2 for the fine-tuned Transformer on a 2,500-sentence sample).

| Model | Test accuracy | Test loss | Negated polarity acc. | Rare words acc. |
|-------|---------------:|----------:|-----------------------:|------------------:|
| Log-linear, one-hot | 80.47% | 0.423 | 64.52% | 56.00% |
| Log-linear, Transformer embeddings | 84.67% | 0.376 | 61.29% | **82.00%** |
| Bi-LSTM | 86.04% | 0.349 | 69.35% | 70.00% |
| Fine-tuned Transformer | **90.14%** | **0.251** | **72.58%** | 80.00% |

![One-hot log-linear: accuracy and loss](images/results_onehot.png)
![Log-linear with Transformer embeddings: accuracy and loss](images/results_transformer_loglinear.png)
![Bi-LSTM: accuracy and loss](images/results_lstm.png)
![Fine-tuned Transformer: accuracy and loss](images/results_finetuned.png)

**Takeaways**

- **One-hot overfits.** Its training accuracy (92.8%) is the highest of all four models, but its test accuracy is the lowest: with no notion of word similarity, every token is learned independently, so the model memorizes the training vocabulary instead of generalizing.
- **Frozen Transformer embeddings already help a linear model.** Swapping one-hot vectors for frozen `distilroberta-base` embeddings, with no other change, raises test accuracy from 80.5% to 84.7%, and more than doubles rare-word accuracy, since rare words still sit near their semantic neighbors in embedding space.
- **Word order matters for negation.** Both log-linear models average token embeddings, discarding order, so words like "not" get diluted into the sentence average. The Bi-LSTM, which processes the sequence directly, improves negated-polarity accuracy from ~61–65% to 69.4%.
- **Fine-tuning wins overall, on far less data.** Despite training on only 2,500 sentences (vs. the full train set used by the other models), the fine-tuned Transformer is best on every metric except rare words, where the frozen log-linear model edges it out slightly, likely because full fine-tuning on a small sample shifts the embedding space a bit away from the pretrained one.

## Key Design Points

- **Frozen vs. fine-tuned Transformer**: models 2 and 3 use `distilroberta-base`'s token embeddings purely as a **fixed lookup table** (no gradient flows into the Transformer); model 4 fine-tunes the whole model end to end on top of the `[CLS]` token.
- **Sub-phrase training data**: the log-linear and LSTM models train on the full set of labeled sub-phrases (not just full sentences), while the fine-tuned Transformer trains on full sentences only, sampled down to 2,500 examples.
- **Evaluation beyond accuracy**: every model is additionally evaluated on two targeted subsets of the test set — sentences with **negated polarity** (where a sub-phrase's sentiment flips the sentence's overall sentiment) and sentences with **rare sentiment words** — to probe generalization beyond overall test accuracy.

## Tech Stack

Python, PyTorch, Hugging Face Transformers, NumPy, Matplotlib




