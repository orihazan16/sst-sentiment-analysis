

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import random

from transformers import AutoTokenizer, AutoModel

import data_loader
from data_loader import get_negated_polarity_examples, get_rare_words_examples

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SEQ_LEN = 52

ONEHOT_AVERAGE       = "onehot_average"
TRANSFORMER_AVERAGE  = "transformer_average"
TRANSFORMER_SEQUENCE = "transformer_sequence"

TRAIN = "train"
VAL   = "val"
TEST  = "test"

TRANSFORMER_MODEL_NAME = "distilroberta-base"


# ──────────────────────────────────────────────────────────────────────────────
# Device helper  (provided)
# ──────────────────────────────────────────────────────────────────────────────

def get_available_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Pickle helpers  (provided)
# ──────────────────────────────────────────────────────────────────────────────

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Section 7 – Transformer embeddings loader
# ──────────────────────────────────────────────────────────────────────────────

def load_transformer_embeddings(cache_path="transformer_emb_cache.pkl"):
    """
    Loads distilroberta-base and extracts the token embedding matrix.
    Caches the matrix locally to avoid reloading.

    Returns
    -------
    tokenizer        : HuggingFace tokenizer
    embedding_matrix : np.ndarray of shape (vocab_size, hidden_dim)
    embedding_dim    : int
    """
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    if os.path.exists(cache_path):
        cached = load_pickle(cache_path)
        embedding_matrix = cached["embedding_matrix"]
        embedding_dim = cached["embedding_dim"]
    else:
        model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
        embedding_matrix = model.embeddings.word_embeddings.weight.detach().cpu().numpy()
        embedding_dim = embedding_matrix.shape[1]
        save_pickle({"embedding_matrix": embedding_matrix, "embedding_dim": embedding_dim}, cache_path)

    return tokenizer, embedding_matrix, embedding_dim


# ──────────────────────────────────────────────────────────────────────────────
# One-hot helpers  (Section 6)
# ──────────────────────────────────────────────────────────────────────────────

def get_word_to_ind(words_list):
    """
    Returns a dict mapping each word in words_list to a unique integer index.
    :param words_list: list of words
    :return: dict {word: index}
    """
    return {word: idx for idx, word in enumerate(sorted(set(words_list)))}


def get_one_hot(size, ind):
    """
    Returns a one-hot numpy vector of length `size` with 1 at position `ind`.
    :param size: length of the vector
    :param ind: index of the 1 entry
    :return: np.ndarray of shape (size,)
    """
    vec = np.zeros(size, dtype=np.float32)
    vec[ind] = 1.0
    return vec


def average_one_hots(sent, word_to_ind):
    """
    Returns the average one-hot embedding for the tokens in `sent`.
    Words not in word_to_ind are ignored (not added to the average).
    :param sent: Sentence object (sent.text is a list of strings)
    :param word_to_ind: dict {word: index}
    :return: np.ndarray of shape (vocab_size,)
    """
    size = len(word_to_ind)
    indices = [word_to_ind[w] for w in sent.text if w in word_to_ind]
    if len(indices) == 0:
        return np.zeros(size, dtype=np.float32)
    vecs = np.stack([get_one_hot(size, ind) for ind in indices], axis=0)
    return vecs.mean(axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Transformer-embedding helpers  (Sections 7 & 8)
# ──────────────────────────────────────────────────────────────────────────────

def get_transformer_average(sent, tokenizer, embedding_matrix, embedding_dim):
    """
    Tokenizes `sent` and returns the average of its Transformer token embeddings.
    Special tokens (e.g. <s>, </s>) should be excluded.
    Token IDs outside the embedding matrix range are treated as unknown (skip them).

    :param sent:             Sentence object
    :param tokenizer:        HuggingFace tokenizer
    :param embedding_matrix: np.ndarray (vocab_size, embedding_dim)
    :param embedding_dim:    int
    :return: np.ndarray of shape (embedding_dim,)
    """
    text = " ".join(sent.text)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    vocab_size = embedding_matrix.shape[0]
    vectors = [embedding_matrix[tid] for tid in token_ids if 0 <= tid < vocab_size]

    if len(vectors) == 0:
        return np.zeros(embedding_dim, dtype=np.float32)

    return np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)


def sentence_to_embedding(sent, tokenizer, embedding_matrix, seq_len, embedding_dim):
    """
    Maps `sent` to a fixed-size matrix of Transformer token embeddings.
    - Sentences longer than seq_len are truncated (keep the first seq_len tokens).
    - Sentences shorter than seq_len are padded with zero-vectors.
    Special tokens are excluded from the encoding.

    :param sent:             Sentence object
    :param tokenizer:        HuggingFace tokenizer
    :param embedding_matrix: np.ndarray (vocab_size, embedding_dim)
    :param seq_len:          int – target sequence length
    :param embedding_dim:    int
    :return: np.ndarray of shape (seq_len, embedding_dim)
    """
    text = " ".join(sent.text)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    vocab_size = embedding_matrix.shape[0]
    vectors = [embedding_matrix[tid] for tid in token_ids if 0 <= tid < vocab_size]

    result = np.zeros((seq_len, embedding_dim), dtype=np.float32)
    n = min(len(vectors), seq_len)
    for i in range(n):
        result[i] = vectors[i]
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Dataset & DataManager  (provided)
# ──────────────────────────────────────────────────────────────────────────────

class OnlineDataset(Dataset):
    """PyTorch Dataset that converts sentences on-the-fly."""

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent  = self.data[idx]
        emb   = self.sent_func(sent, **self.sent_func_kwargs)
        label = sent.sentiment_class
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class DataManager:
    """
    Handles all data management.
    data_type: one of ONEHOT_AVERAGE, TRANSFORMER_AVERAGE, TRANSFORMER_SEQUENCE
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True,
                 dataset_path="stanfordSentimentTreebank", batch_size=64,
                 embedding_dim=None, tokenizer=None, embedding_matrix=None):

        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)

        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()
        self.sentences[VAL]  = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        words_list = list(self.sentiment_dataset.get_word_counts().keys())

        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}

        elif data_type == TRANSFORMER_AVERAGE:
            assert tokenizer is not None and embedding_matrix is not None
            self.sent_func = get_transformer_average
            self.sent_func_kwargs = {
                "tokenizer":        tokenizer,
                "embedding_matrix": embedding_matrix,
                "embedding_dim":    embedding_dim,
            }

        elif data_type == TRANSFORMER_SEQUENCE:
            assert tokenizer is not None and embedding_matrix is not None
            self.sent_func = sentence_to_embedding
            self.sent_func_kwargs = {
                "tokenizer":        tokenizer,
                "embedding_matrix": embedding_matrix,
                "seq_len":          SEQ_LEN,
                "embedding_dim":    embedding_dim,
            }
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        self.torch_datasets  = {k: OnlineDataset(sents, self.sent_func, self.sent_func_kwargs)
                                 for k, sents in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(ds, batch_size=batch_size, shuffle=(k == TRAIN))
                                 for k, ds in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        return np.array([s.sentiment_class for s in self.sentences[data_subset]])

    def get_input_shape(self):
        return self.torch_datasets[TRAIN][0][0].shape


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

class LogLinear(nn.Module):
    """
    Log-linear model for sentiment analysis.
    Used for both the one-hot model (Section 6) and the Transformer-embedding model (Section 7).
    forward() must return LOGITS (no sigmoid).
    predict() applies sigmoid and returns probabilities.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))


class LSTM(nn.Module):
    """
    Bidirectional LSTM for sentiment analysis (Section 8).
    Input  : (batch, seq_len, embedding_dim)
    Output : logits of shape (batch,)
    forward() must return LOGITS (no sigmoid).
    predict() applies sigmoid and returns probabilities.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim * 2, 1)

    def forward(self, text):
        # text: (batch, seq_len, embedding_dim)
        _, (h_n, _) = self.lstm(text)
        # h_n: (num_layers * 2, batch, hidden_dim); last layer's fwd/bwd states are the last two entries
        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]
        hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        hidden = self.dropout(hidden)
        logits = self.linear(hidden)
        return logits.squeeze(-1)

    def predict(self, text):
        return torch.sigmoid(self.forward(text))


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────

def binary_accuracy(preds, y):
    """
    Returns the fraction of correctly classified examples.
    preds: probabilities in [0,1] (or logits – round at 0.5)
    y:     true labels {0, 1}
    :return: float
    """
    rounded = torch.round(preds)
    correct = (rounded == y).float()
    return correct.mean().item()


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    Runs one full pass over the training data.
    :param model:         a LogLinear or LSTM model
    :param data_iterator: PyTorch DataLoader for the training set
    :param optimizer:     torch optimizer
    :param criterion:     loss criterion (BCEWithLogitsLoss)
    :return: (avg_loss, avg_accuracy) over the epoch
    """
    model.train()
    device = next(model.parameters()).device

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    for x, y in data_iterator:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = torch.sigmoid(logits)
        total_loss += loss.item()
        total_acc += binary_accuracy(preds, y)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def evaluate(model, data_iterator, criterion):
    """
    Evaluates the model WITHOUT updating parameters.
    Use torch.no_grad() to avoid computing unnecessary gradients.
    :param model:         a LogLinear or LSTM model
    :param data_iterator: PyTorch DataLoader (val or test)
    :param criterion:     loss criterion
    :return: (avg_loss, avg_accuracy)
    """
    model.eval()
    device = next(model.parameters()).device

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in data_iterator:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            preds = torch.sigmoid(logits)
            total_loss += loss.item()
            total_acc += binary_accuracy(preds, y)
            n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def get_predictions_for_data(model, data_iter):
    """
    Collects all model predictions (probabilities) over a data iterator.
    :param model:     a LogLinear or LSTM model
    :param data_iter: PyTorch DataLoader
    :return: np.ndarray of shape (n_examples,)
    """
    model.eval()
    device = next(model.parameters()).device

    all_preds = []
    with torch.no_grad():
        for x, _ in data_iter:
            x = x.to(device)
            preds = model.predict(x)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.0):
    """
    Full training loop using Adam optimizer and BCEWithLogitsLoss.
    Records per-epoch train/validation loss and accuracy.

    :param model:        LogLinear or LSTM instance
    :param data_manager: DataManager instance
    :param n_epochs:     number of epochs
    :param lr:           learning rate
    :param weight_decay: L2 regularization coefficient
    :return: dict with keys train_loss, train_acc, val_loss, val_acc
             (each a list of length n_epochs)
    """
    device = get_available_device()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_iter = data_manager.get_torch_iterator(TRAIN)
    val_iter = data_manager.get_torch_iterator(VAL)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_iter, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_iter, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch + 1}/{n_epochs} | "
              f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    return history


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helper  (provided – feel free to modify)
# ──────────────────────────────────────────────────────────────────────────────

def plot_and_save(history, title_prefix, save_prefix):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"],   label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"{title_prefix} – Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"],   label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} – Accuracy")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{save_prefix}_acc.png")
    plt.close()


def evaluate_special_subsets(model, data_manager):
    """Prints accuracy on negated-polarity and rare-words subsets (provided)."""
    dataset    = data_manager.sentiment_dataset
    test_sents = data_manager.sentences[TEST]

    neg_idx  = get_negated_polarity_examples(test_sents)
    rare_idx = get_rare_words_examples(test_sents, dataset)

    all_preds  = get_predictions_for_data(model, data_manager.get_torch_iterator(TEST))
    all_labels = data_manager.get_labels(TEST)

    for name, indices in [("Negated polarity", neg_idx), ("Rare words", rare_idx)]:
        if len(indices) == 0:
            print(f"  {name}: no examples found")
            continue
        acc = float(np.mean(
            (all_preds[indices] >= 0.5).astype(float) == all_labels[indices]
        ))
        print(f"  {name} accuracy ({len(indices)} examples): {acc:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Section 6 – Log-linear with one-hot
# ──────────────────────────────────────────────────────────────────────────────

def train_log_linear_with_one_hot():
    """
    Creates all objects needed for training/evaluation of the one-hot log-linear model
    and runs the training process.
    Hyperparameters: lr=0.01, n_epochs=10, batch_size=64
    """
    data_manager = DataManager(data_type=ONEHOT_AVERAGE, use_sub_phrases=True, batch_size=64)
    embedding_dim = data_manager.get_input_shape()[0]

    model = LogLinear(embedding_dim)
    history = train_model(model, data_manager, n_epochs=10, lr=0.01, weight_decay=0.0)

    plot_and_save(history, "Log-Linear (One-Hot)", "log_linear_one_hot")

    device = get_available_device()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_acc = evaluate(model, data_manager.get_torch_iterator(TEST), criterion)
    print(f"[One-Hot Log-Linear] Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    evaluate_special_subsets(model, data_manager)

    return model, data_manager, history


# ──────────────────────────────────────────────────────────────────────────────
# Section 7 – Log-linear with Transformer embeddings
# ──────────────────────────────────────────────────────────────────────────────

def train_log_linear_with_transformer():
    """
    Creates all objects needed for training/evaluation of the Transformer-embedding
    log-linear model and runs the training process.
    Hyperparameters: lr=0.01, n_epochs=10, batch_size=64
    """
    tokenizer, embedding_matrix, embedding_dim = load_transformer_embeddings()

    data_manager = DataManager(
        data_type=TRANSFORMER_AVERAGE, use_sub_phrases=True, batch_size=64,
        embedding_dim=embedding_dim, tokenizer=tokenizer, embedding_matrix=embedding_matrix,
    )

    model = LogLinear(embedding_dim)
    history = train_model(model, data_manager, n_epochs=10, lr=0.01, weight_decay=0.0)

    plot_and_save(history, "Log-Linear (Transformer Embeddings)", "log_linear_transformer")

    device = get_available_device()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_acc = evaluate(model, data_manager.get_torch_iterator(TEST), criterion)
    print(f"[Transformer Log-Linear] Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    evaluate_special_subsets(model, data_manager)

    return model, data_manager, history


# ──────────────────────────────────────────────────────────────────────────────
# Section 8 – Bi-LSTM with Transformer embeddings
# ──────────────────────────────────────────────────────────────────────────────

def train_lstm_with_transformer():
    """
    Creates all objects needed for training/evaluation of the bi-LSTM model
    and runs the training process.
    Hyperparameters: lr=0.001, weight_decay=0.0001, dropout=0.5, n_epochs=4, batch_size=64
    """
    tokenizer, embedding_matrix, embedding_dim = load_transformer_embeddings()

    data_manager = DataManager(
        data_type=TRANSFORMER_SEQUENCE, use_sub_phrases=True, batch_size=64,
        embedding_dim=embedding_dim, tokenizer=tokenizer, embedding_matrix=embedding_matrix,
    )

    model = LSTM(embedding_dim=embedding_dim, hidden_dim=100, n_layers=1, dropout=0.5)
    history = train_model(model, data_manager, n_epochs=4, lr=0.001, weight_decay=0.0001)

    plot_and_save(history, "Bi-LSTM (Transformer Embeddings)", "lstm_transformer")

    device = get_available_device()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_acc = evaluate(model, data_manager.get_torch_iterator(TEST), criterion)
    print(f"[Bi-LSTM] Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    evaluate_special_subsets(model, data_manager)

    return model, data_manager, history


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Exercise 5 – Sentiment Analysis")
    # print("train_log_linear_with_one_hot()")
    # train_log_linear_with_one_hot()
    print("train_log_linear_with_transformer()")
    train_log_linear_with_transformer()
#     # train_lstm_with_transformer()



# ══════════════════════════════════════════════════════════════════════════════
# Section 9 – Fine-Tuning a Transformer Model
# ══════════════════════════════════════════════════════════════════════════════

class TransformerSentimentDataset(Dataset):
    """
    Encodes each sentence with the Transformer tokenizer on the fly.
    __getitem__ should return (input_ids, attention_mask, label).
    """

    MAX_LENGTH = 64

    def __init__(self, sentences, tokenizer, max_length=None):
        self.sentences  = sentences
        self.tokenizer  = tokenizer
        self.max_length = max_length or self.MAX_LENGTH

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        text = " ".join(sent.text)

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        label = torch.tensor(sent.sentiment_class, dtype=torch.float32)

        return input_ids, attention_mask, label


class TransformerSentimentClassifier(nn.Module):
    """
    distilroberta-base base model (no HF classification head) +
    a single Linear layer on top of the [CLS] token's final hidden state.
    forward() returns logits (no sigmoid).
    predict() returns probabilities (after sigmoid).
    """

    def __init__(self, model_name=TRANSFORMER_MODEL_NAME):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_dim = self.transformer.config.hidden_size
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        logits = self.linear(cls_hidden)
        return logits.squeeze(-1)

    def predict(self, input_ids, attention_mask):
        return torch.sigmoid(self.forward(input_ids, attention_mask))


def train_transformer(dataset_path="stanfordSentimentTreebank",
                      batch_size=64, n_epochs=2, lr=1e-5, weight_decay=0.0):
    """
    Full training procedure for the fine-tuned Transformer.
    Use BCEWithLogitsLoss and Adam. No HuggingFace Trainer.
    Report train/val loss and accuracy per epoch, test loss/accuracy,
    and accuracy on the two special subsets.
    """
    # NOTE: the provided DataManager (used in Sections 6-8) is built around the
    # average/sequence embedding functions and has no max_train_samples argument.
    # For this section we therefore build the sentence lists and DataLoaders
    # directly from SentimentTreeBank, sampling 2500 *full sentences* (no
    # sub-phrases) for the training set, as required by the exercise.
    device = get_available_device()
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
    train_sents_full = sentiment_dataset.get_train_set()
    val_sents = sentiment_dataset.get_validation_set()
    test_sents = sentiment_dataset.get_test_set()

    rng = random.Random(0)
    train_sents = (rng.sample(train_sents_full, 2500)
                   if len(train_sents_full) > 2500 else train_sents_full)

    train_ds = TransformerSentimentDataset(train_sents, tokenizer)
    val_ds = TransformerSentimentDataset(val_sents, tokenizer)
    test_ds = TransformerSentimentDataset(test_sents, tokenizer)

    train_iter = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_iter = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_iter = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = TransformerSentimentClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    def run_epoch(data_iter, train_mode):
        model.train() if train_mode else model.eval()
        total_loss, total_acc, n_batches = 0.0, 0.0, 0
        context = torch.enable_grad() if train_mode else torch.no_grad()
        with context:
            for input_ids, attention_mask, y in data_iter:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                y = y.to(device)

                if train_mode:
                    optimizer.zero_grad()

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, y)

                if train_mode:
                    loss.backward()
                    optimizer.step()

                preds = torch.sigmoid(logits)
                total_loss += loss.item()
                total_acc += binary_accuracy(preds, y)
                n_batches += 1

        return total_loss / n_batches, total_acc / n_batches

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(n_epochs):
        train_loss, train_acc = run_epoch(train_iter, train_mode=True)
        val_loss, val_acc = run_epoch(val_iter, train_mode=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch + 1}/{n_epochs} | "
              f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    plot_and_save(history, "Fine-Tuned Transformer", "transformer_finetuned")

    test_loss, test_acc = run_epoch(test_iter, train_mode=False)
    print(f"[Fine-Tuned Transformer] Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # Special subsets (negated polarity / rare words)
    neg_idx = get_negated_polarity_examples(test_sents)
    rare_idx = get_rare_words_examples(test_sents, sentiment_dataset)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for input_ids, attention_mask, _ in test_iter:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            preds = model.predict(input_ids, attention_mask)
            all_preds.append(preds.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.array([s.sentiment_class for s in test_sents])

    for name, indices in [("Negated polarity", neg_idx), ("Rare words", rare_idx)]:
        if len(indices) == 0:
            print(f"  {name}: no examples found")
            continue
        acc = float(np.mean(
            (all_preds[indices] >= 0.5).astype(float) == all_labels[indices]
        ))
        print(f"  {name} accuracy ({len(indices)} examples): {acc:.4f}")

    return model, history

