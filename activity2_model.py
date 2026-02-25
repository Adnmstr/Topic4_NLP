"""
============================================================================
AIT-204 Deep Learning | Topic 4: Natural Language Processing
ACTIVITY 2: Model Architecture — Embeddings + Sentiment Classifier
============================================================================

PURPOSE:
    Define the neural network architecture for sentiment classification.
    This module is imported by Activity 3 (training) and Activity 4 (app).

DURATION: ~75 minutes in-class

BRIDGING THE GAP (Concept -> Algorithm -> Code -> Deployed App):
    ---------------------------------------------------------------
    PROBLEM: After Activity 1, each word is an integer ID.
        "great" = 7,  "wonderful" = 15,  "terrible" = 23

    These IDs have NO meaning. Is word 7 closer to word 8 or word 23?
    The integer ordering is arbitrary.

    SOLUTION — EMBEDDINGS:
        Each word gets a dense vector of real numbers (learned during training):
            "great"     -> [ 0.82, -0.15,  0.43, ...]   (64 dimensions)
            "wonderful" -> [ 0.79, -0.12,  0.47, ...]   (similar to "great"!)
            "terrible"  -> [-0.71,  0.23, -0.38, ...]   (far from "great")

    ANALOGY FROM TOPIC 3:
        CNN filters LEARN to detect visual features from pixel values.
        Embedding layers LEARN to detect semantic features from word IDs.

    YOUR MODEL ARCHITECTURE (compare to Topic 2's neural network):

        Topic 2:  [Numerical Features] -> Linear -> ReLU -> Linear -> Output
        Topic 4:  [Word IDs] -> Embedding -> AvgPool -> Linear -> ReLU -> Linear -> Output
                               ^^^^^^^^    ^^^^^^^
                              NEW for NLP   NEW for NLP (rest is same as Topic 2!)

    HOW nn.Embedding WORKS:
        It's a lookup table. A matrix E of shape (vocab_size, embed_dim).
        Given word ID w, it returns row E[w]. That's it.
        During training, backpropagation updates E so similar words
        get similar rows.

    FOR YOUR APP:
        When Activity 4's web app receives a review from the user,
        this model processes it:
            User types review -> Activity 1 preprocesses it -> THIS model predicts sentiment

WHAT YOU'LL IMPLEMENT:
    - SentimentClassifier.__init__(): Define layers
    - SentimentClassifier.forward(): Wire layers together
    - Understand parameter counts and architecture trade-offs

RUN THIS FILE: python activity2_model.py
============================================================================
"""

import torch
import torch.nn as nn
import os
import tempfile


class SentimentClassifier(nn.Module):
    """
    Feed-forward neural network for binary sentiment classification.

    Architecture:
        Word IDs -> Embedding -> Average Pooling -> FC1 -> ReLU -> Dropout -> FC2 -> Sigmoid

    This is the same structure as a Topic 2 neural network, with an
    Embedding layer added at the front to handle text input.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx=0, dropout=0.3):
        """
        Args:
            vocab_size (int): Number of words in vocabulary (from Activity 1)
            embed_dim (int): Dimension of word embedding vectors
            hidden_dim (int): Neurons in hidden layer
            pad_idx (int): Index of <PAD> token (embeddings stay zero for padding)
            dropout (float): Dropout rate for regularization
        """
        super(SentimentClassifier, self).__init__()

        self.pad_idx = pad_idx

        # Store hyperparameters for save/load
        self.config = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "pad_idx": pad_idx,
            "dropout": dropout,
        }

        # ---- LAYER DEFINITIONS ----

        # TODO 1: Create the embedding layer
        # This is a lookup table: given word ID i, return row i of a
        # weight matrix of shape (vocab_size, embed_dim).
        # padding_idx=pad_idx ensures <PAD> tokens always embed to zeros.
        #
        # HINT: nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # TODO 2: Create the first fully-connected (linear) layer
        # Input: embed_dim (the averaged embedding for the whole sentence)
        # Output: hidden_dim (number of hidden neurons)
        # This is EXACTLY the same type of layer from Topic 2.
        #
        # HINT: nn.Linear(embed_dim, hidden_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)

        # Activation and regularization (provided)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # TODO 3: Create the output layer
        # Input: hidden_dim
        # Output: 1 (single probability: positive vs. negative)
        #
        # HINT: nn.Linear(hidden_dim, 1)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, text_ids):
        """
        Forward pass: word IDs in, sentiment probability out.

        Args:
            text_ids: tensor of shape (batch_size, seq_len) with integer word IDs

        Returns:
            tensor of shape (batch_size, 1) with probability of positive sentiment.
            Values near 1.0 = positive, near 0.0 = negative.
        """
        # STEP 1: Embed — convert integer IDs to dense vectors
        # Input:  (batch, seq_len)         e.g. (32, 15)
        # Output: (batch, seq_len, embed)  e.g. (32, 15, 64)
        #
        # TODO 4: Pass text_ids through self.embedding
        # HINT: embedded = self.embedding(text_ids)
        embedded = self.embedding(text_ids)  # (batch, seq_len, embed)

        # STEP 2: Average Pooling — collapse sequence into single vector
        # We average all NON-PADDING embeddings in each sequence.
        # Input:  (batch, seq_len, embed)  e.g. (32, 15, 64)
        # Output: (batch, embed)           e.g. (32, 64)
        #
        # This is given — study how it works:
        mask = (text_ids != self.pad_idx).float()          # (batch, seq_len)
        mask_expanded = mask.unsqueeze(2)                   # (batch, seq_len, 1)
        masked_embeddings = embedded * mask_expanded         # zero out padding
        summed = masked_embeddings.sum(dim=1)               # (batch, embed)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = summed / lengths                           # (batch, embed)

        # STEP 3: Classification head — same as Topic 2!
        # pooled -> fc1 -> relu -> dropout -> fc2 -> sigmoid
        #
        # TODO 5: Pass `pooled` through the classification layers
        # HINT: Chain them one by one:
        #   x = self.fc1(pooled)
        #   x = self.relu(x)
        #   x = self.dropout(x)
        #   x = self.fc2(x)
        #   output = self.sigmoid(x)
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.sigmoid(x)

        return output

    def get_embeddings(self):
        """Return the embedding weight matrix (for visualization in the app)."""
        return self.embedding.weight.detach()


# =========================================================================
# HELPER: Model factory with save/load support
# =========================================================================
# These functions are used by train.py (save) and app.py (load).

def save_model(model, filepath):
    """Save model weights and config to a file."""
    torch.save({
        "config": model.config,
        "state_dict": model.state_dict(),
    }, filepath)


def load_model(filepath, device="cpu"):
    """Load a saved model from file."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = SentimentClassifier(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


# =========================================================================
# DEMO: Run this file to inspect the architecture
# =========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  ACTIVITY 2: Model Architecture — Demo & Inspection")
    print("=" * 65)

    # Create a model with example hyperparameters
    VOCAB_SIZE = 150   # Typical for our small dataset
    EMBED_DIM = 64     # Each word becomes a 64-dimensional vector
    HIDDEN_DIM = 32    # Hidden layer size

    model = SentimentClassifier(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
    )

    # --- Print architecture ---
    print("\n[Architecture]")
    print(model)

    # --- Parameter count breakdown ---
    print("\n[Parameter Counts]")
    total = 0
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        print(f"  {name:<25} shape={str(list(param.shape)):<16} params={count:>8,}")
    print(f"  {'TOTAL':<25} {'':16} params={total:>8,}")

    # --- Show how embedding works ---
    print("\n[Embedding Demonstration]")
    # Simulate a batch of 2 reviews, each 5 words long
    fake_input = torch.tensor([
        [2, 5, 8, 3, 0],   # "this movie was great <PAD>"
        [4, 7, 0, 0, 0],   # "terrible film <PAD> <PAD> <PAD>"
    ], dtype=torch.long)

    print(f"  Input shape:  {fake_input.shape}  (batch=2, seq_len=5)")

    with torch.no_grad():
        output = model(fake_input)
    print(f"  Output shape: {output.shape}  (batch=2, prediction=1)")
    print(f"  Predictions:  {output.squeeze().tolist()}")
    print(f"  (Random predictions — model is untrained)")

    # --- Embedding dimension comparison ---
    print("\n[Embedding Dimension Trade-offs]")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  {'Embed Dim':<12} {'Embed Params':>14} {'Total Params':>14}")
    print(f"  {'-'*42}")
    for dim in [16, 32, 64, 128, 256]:
        m = SentimentClassifier(VOCAB_SIZE, dim, HIDDEN_DIM)
        embed_params = VOCAB_SIZE * dim
        total_params = sum(p.numel() for p in m.parameters())
        print(f"  {dim:<12} {embed_params:>14,} {total_params:>14,}")

    # --- One-hot vs Embedding comparison ---
    print("\n[Why Embeddings Beat One-Hot Encoding]")
    print(f"  One-hot vector for 1 word:  {VOCAB_SIZE} floats ({VOCAB_SIZE * 4} bytes)")
    print(f"  Embedding vector for 1 word: {EMBED_DIM} floats ({EMBED_DIM * 4} bytes)")
    print(f"  Compression ratio: {VOCAB_SIZE / EMBED_DIM:.1f}x smaller")
    print(f"  AND embeddings capture meaning — one-hot does not!")

    # --- Test save/load ---
    print("\n[Save/Load Test]")
    temp_model_path = os.path.join(tempfile.gettempdir(), "test_model.pt")
    save_model(model, temp_model_path)
    loaded = load_model(temp_model_path)
    with torch.no_grad():
        out1 = model(fake_input)
        out2 = loaded(fake_input)
    match = torch.allclose(out1, out2)
    print(f"  Saved and loaded model. Outputs match: {match}")

    print("\n" + "=" * 65)
    print("  ACTIVITY 2 COMPLETE")
    print("  You defined the model architecture that your app will use.")
    print("  Next: Activity 3 — Train the model and save it to disk.")
    print("=" * 65)

    # ---- REFLECTION QUESTIONS (answer in your report) ----
    #
    # 1. The embedding layer has vocab_size * embed_dim parameters.
    #    For a vocabulary of 50,000 words and embed_dim=256, how many
    #    parameters is that? Is it more or fewer than the classification
    #    layers? What does this tell you about where the "knowledge" lives?
    #
    # 2. We use Average Pooling to go from per-word embeddings to one
    #    sentence vector. "not great" and "great not" produce the SAME
    #    averaged vector. Why is this a limitation? What architectures
    #    solve this? (Hint: Topic 5 covers RNNs)
    #
    # 3. The sigmoid output gives a probability between 0 and 1. We
    #    treat >= 0.5 as positive, < 0.5 as negative. What might it
    #    mean when the model outputs exactly 0.50? Should you trust
    #    a prediction of 0.51 as much as 0.99?
    #
    # 4. Compare this model's architecture to the CNN from Topic 3.
    #    Both take raw data and classify it. What role does the
    #    Embedding layer play that's analogous to Conv layers in a CNN?
