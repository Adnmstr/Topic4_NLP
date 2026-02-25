"""
============================================================================
AIT-204 Deep Learning | Topic 4: Natural Language Processing
ACTIVITY 1: Text Preprocessing Module
============================================================================

PURPOSE:
    Build a reusable text preprocessing module that converts raw text into
    PyTorch tensors. This module is imported by Activities 2-4 and by the
    final deployed web app.

DURATION: ~75 minutes in-class

BRIDGING THE GAP (Concept -> Algorithm -> Code -> Deployed App):
    ---------------------------------------------------------------
    In Topics 1-3, your neural network inputs were already numbers:
      - Topic 1: x-values for linear regression
      - Topic 2: numerical inputs for perceptrons
      - Topic 3: pixel values (0-255) for CNNs

    In NLP, input is RAW TEXT — "This movie was great!"
    Neural networks cannot process strings. They need tensors of numbers.

    THE PIPELINE (each function below implements one step):

        Raw Text
          |  clean_text()      — normalize: lowercase, remove punctuation
          v
        "this movie was great"
          |  tokenize()        — split into word units
          v
        ["this", "movie", "was", "great"]
          |  vocab.encode()    — map each word to an integer ID
          v
        [4, 12, 7, 23]
          |  pad_sequence()    — make all sequences the same length
          v
        [4, 12, 7, 23, 0, 0, 0, 0, 0, 0]    <- 0 is <PAD>
          |  torch.tensor()   — convert to PyTorch tensor
          v
        tensor([4, 12, 7, 23, 0, 0, 0, 0, 0, 0])   <- READY for neural network!

    WHY THIS MATTERS FOR YOUR APP:
        When a user types a review into your web app (Activity 4), THIS module
        is what processes their text before the model can make a prediction.
        Without it, the model has no input.

WHAT YOU'LL IMPLEMENT:
    - clean_text(): Text normalization
    - tokenize(): Word splitting
    - Vocabulary class: Word-to-integer mapping
    - pad_sequence(): Fixed-length sequences
    - preprocess_for_model(): Full pipeline convenience function

RUN THIS FILE: python activity1_preprocessing.py
    The __main__ block at the bottom demonstrates and tests every function.
============================================================================
"""

import re
import json
import os
import tempfile
import torch
from collections import Counter


# =========================================================================
# STEP 1: TEXT CLEANING
# =========================================================================
# CONCEPT: "Great", "great", and "GREAT" must all be the same word.
#          Punctuation like "!!!" adds no value for our model.
# ALGORITHM: lowercase -> remove non-letters -> collapse whitespace
# =========================================================================

def clean_text(text):
    """
    Normalize raw text for NLP processing.

    Args:
        text (str): Raw input, e.g. "This Movie was GREAT!!!"
    Returns:
        str: Cleaned text, e.g. "this movie was great"

    Example:
        >>> clean_text("I'd rate it 10/10 -- a MUST-SEE!!!")
        'id rate it  a mustsee'
    """
    # TODO 1: Convert to lowercase
    # HINT: Use the .lower() string method
    text = text.lower()

    # TODO 2: Remove everything that is NOT a lowercase letter or space
    # HINT: re.sub(r'[^a-z ]', '', text) replaces non-matching chars with ''
    text = re.sub(r'[^a-z ]', '', text)

    # TODO 3: Replace multiple spaces with a single space, then strip edges
    # HINT: re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# =========================================================================
# STEP 2: TOKENIZATION
# =========================================================================
# CONCEPT: Split text into individual word tokens.
#          "this movie was great" -> ["this", "movie", "was", "great"]
# WHY: Each token becomes one position in the input sequence.
# =========================================================================

def tokenize(text):
    """
    Split cleaned text into word tokens.

    Args:
        text (str): Cleaned text
    Returns:
        list[str]: List of word tokens

    Example:
        >>> tokenize("this movie was great")
        ['this', 'movie', 'was', 'great']
    """
    # TODO 4: Split the text on whitespace
    # HINT: .split() with no arguments splits on any whitespace
    tokens = text.split()

    return tokens


# =========================================================================
# STEP 3: VOCABULARY
# =========================================================================
# CONCEPT: Map every unique word to a unique integer ID.
#   Special tokens:
#     <PAD> = 0  : Fills shorter sequences to fixed length
#     <UNK> = 1  : Replaces words not seen during training
#
# WHY: The embedding layer (Activity 2) is indexed by these IDs.
#      vocab_size determines the embedding table size and parameter count.
#
# IMPORTANT FOR DEPLOYMENT: The vocabulary built during training must be
#   SAVED and loaded by the app. We use JSON for this (save/load methods).
# =========================================================================

class Vocabulary:
    """
    Maps words to integer indices and back. Supports save/load for deployment.

    Usage:
        vocab = Vocabulary(min_freq=2)
        vocab.build([ ["this", "is", "good"], ["this", "is", "bad"] ])
        ids = vocab.encode(["this", "is", "unknown"])   # -> [2, 3, 1]
        words = vocab.decode([2, 3, 1])                 # -> ["this", "is", "<UNK>"]
        vocab.save("vocab.json")
        vocab2 = Vocabulary.load("vocab.json")
    """

    def __init__(self, min_freq=1):
        """
        Args:
            min_freq: Minimum word frequency to include in vocabulary.
                      Words appearing fewer times map to <UNK>.
        """
        self.min_freq = min_freq
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()

    def build(self, token_lists):
        """
        Build vocabulary from a list of tokenized documents.

        Args:
            token_lists: list of lists of tokens, e.g.
                         [["this", "is", "good"], ["this", "is", "bad"]]
        """
        # Count every word across all documents
        # TODO 5: Update self.word_counts with each token list
        # HINT: self.word_counts.update(tokens) adds counts from one list
        for tokens in token_lists:
            self.word_counts.update(tokens)

        # Start with special tokens
        self.word2idx = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}

        # TODO 6: Add words meeting min_freq to word2idx
        # For each (word, count) in self.word_counts.items():
        #   if count >= self.min_freq:
        #       assign the word the next available index: len(self.word2idx)
        for word, count in self.word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = len(self.word2idx)

        # Build reverse mapping
        # TODO 7: Create idx2word by inverting word2idx
        # HINT: dict comprehension {idx: word for word, idx in ...}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, tokens):
        """Convert tokens to integer IDs. Unknown words become <UNK>."""
        # TODO 8: Return list of IDs using word2idx.get(token, self.unk_idx)
        return [self.word2idx.get(token, self.unk_idx) for token in tokens]

    def decode(self, indices):
        """Convert integer IDs back to words."""
        return [self.idx2word.get(idx, self.unk_token) for idx in indices]

    def __len__(self):
        return len(self.word2idx)

    # --- SAVE/LOAD for deployment ---
    # These are provided. Your web app (Activity 4) will call Vocabulary.load()
    # to recover the exact same vocabulary used during training.

    def save(self, filepath):
        """Save vocabulary to JSON for use by the deployed app."""
        data = {
            "min_freq": self.min_freq,
            "word2idx": self.word2idx,
            "word_counts": dict(self.word_counts),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath):
        """Load a saved vocabulary from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        vocab = cls(min_freq=data["min_freq"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(i): w for w, i in vocab.word2idx.items()}
        vocab.word_counts = Counter(data["word_counts"])
        return vocab


# =========================================================================
# STEP 4: SEQUENCE PADDING
# =========================================================================
# CONCEPT: Neural networks process batches. Batches require uniform shape.
#          Sentences have different lengths. Padding fixes this.
#
#   Before: [[5, 3],       [8, 2, 7, 4, 9]]
#   After:  [[5, 3, 0, 0, 0], [8, 2, 7, 4, 9]]
#                  ^^^^^ <PAD>
# =========================================================================

def pad_sequence(encoded, max_length, pad_value=0):
    """
    Pad or truncate a single encoded sequence to max_length.

    Args:
        encoded (list[int]): Integer-encoded token sequence
        max_length (int): Target length
        pad_value (int): Padding token ID (default 0 = <PAD>)
    Returns:
        list[int]: Sequence of exactly max_length integers
    """
    # TODO 9: Truncate if too long, pad if too short
    # If len(encoded) >= max_length: return encoded[:max_length]
    # Else: return encoded + [pad_value] * (max_length - len(encoded))
    if len(encoded) >= max_length:
        return encoded[:max_length]
    else:
        return encoded + [pad_value] * (max_length - len(encoded))


# =========================================================================
# STEP 5: FULL PIPELINE (convenience function)
# =========================================================================
# This chains all steps. Used by train.py AND by app.py at inference time.
# =========================================================================

def preprocess_for_model(text, vocab, max_length):
    """
    Full pipeline: raw text -> padded integer tensor ready for the model.

    Args:
        text (str): Raw review text
        vocab (Vocabulary): Built vocabulary
        max_length (int): Fixed sequence length
    Returns:
        torch.Tensor: Shape (1, max_length), dtype long — batched for model input
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    encoded = vocab.encode(tokens)
    padded = pad_sequence(encoded, max_length)
    return torch.tensor([padded], dtype=torch.long)


def preprocess_dataset(reviews, vocab, max_length, fit_vocab=False):
    """
    Preprocess an entire dataset of (text, label) pairs.

    Args:
        reviews: list of (text, label) tuples
        vocab: Vocabulary instance
        max_length: Fixed sequence length
        fit_vocab: If True, build vocab from this data (only for training data!)
    Returns:
        (text_tensor, label_tensor): Both torch.long tensors
    """
    all_tokens = [tokenize(clean_text(text)) for text, _ in reviews]
    labels = [label for _, label in reviews]

    if fit_vocab:
        vocab.build(all_tokens)

    encoded = [vocab.encode(tokens) for tokens in all_tokens]
    padded = [pad_sequence(seq, max_length) for seq in encoded]

    return torch.tensor(padded, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# =========================================================================
# DEMO: Run this file to test your implementations
# =========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  ACTIVITY 1: Text Preprocessing Module — Demo & Tests")
    print("=" * 65)

    # --- Test clean_text ---
    print("\n[Step 1] Testing clean_text():")
    test_cases = [
        ("This Movie was GREAT!!!", "this movie was great"),
        ("   Terrible...  absolutely   terrible.  ", "terrible absolutely terrible"),
        ("I'd rate it 10/10!!!", "id rate it"),
    ]
    for raw, expected in test_cases:
        result = clean_text(raw)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{raw}' -> '{result}'")

    # --- Test tokenize ---
    print("\n[Step 2] Testing tokenize():")
    tokens = tokenize("this movie was great")
    print(f"  'this movie was great' -> {tokens}")
    assert tokens == ["this", "movie", "was", "great"], "tokenize() failed!"
    print("  PASS")

    # --- Test Vocabulary ---
    print("\n[Step 3] Testing Vocabulary:")
    sample_docs = [
        ["this", "movie", "was", "great", "and", "wonderful"],
        ["this", "movie", "was", "terrible", "and", "awful"],
        ["a", "great", "movie", "with", "great", "acting"],
    ]
    vocab = Vocabulary(min_freq=1)
    vocab.build(sample_docs)
    print(f"  Vocabulary size: {len(vocab)} (including <PAD>, <UNK>)")
    print(f"  'movie' -> ID {vocab.word2idx.get('movie', '?')}")

    encoded = vocab.encode(["this", "movie", "is", "unknown_word"])
    print(f"  encode(['this', 'movie', 'is', 'unknown_word']) -> {encoded}")
    print(f"  decode({encoded}) -> {vocab.decode(encoded)}")

    # --- Test pad_sequence ---
    print("\n[Step 4] Testing pad_sequence():")
    short = [5, 3]
    long = [1, 2, 3, 4, 5, 6, 7, 8]
    print(f"  pad_sequence({short}, max_length=5) -> {pad_sequence(short, 5)}")
    print(f"  pad_sequence({long}, max_length=5)  -> {pad_sequence(long, 5)}")

    # --- Test full pipeline ---
    print("\n[Step 5] Testing full pipeline:")
    tensor = preprocess_for_model("This Movie was GREAT!!!", vocab, max_length=8)
    print(f"  Input:  'This Movie was GREAT!!!'")
    print(f"  Output: {tensor} (shape: {tensor.shape})")
    print(f"  Decoded: {vocab.decode(tensor[0].tolist())}")

    # --- Test save/load ---
    print("\n[Deployment] Testing save/load:")
    temp_vocab_path = os.path.join(tempfile.gettempdir(), "test_vocab.json")
    vocab.save(temp_vocab_path)
    vocab_loaded = Vocabulary.load(temp_vocab_path)
    assert len(vocab_loaded) == len(vocab), "Save/load failed!"
    print(f"  Saved and loaded vocabulary: {len(vocab_loaded)} words. PASS")

    print("\n" + "=" * 65)
    print("  ACTIVITY 1 COMPLETE")
    print("  You built the preprocessing module that your app will use.")
    print("  Next: Activity 2 — Model architecture with embeddings.")
    print("=" * 65)
