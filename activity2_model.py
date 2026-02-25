"""
============================================================================
AIT-204 Deep Learning | Topic 4: Natural Language Processing
ACTIVITY 2: Model Architecture — Embeddings + Sentiment Classifier
============================================================================
"""

import torch
import torch.nn as nn
import os
import tempfile


class SentimentClassifier(nn.Module):
    """
    Supports:
      - binary sentiment: output shape (batch, 1) with sigmoid probability
      - multiclass sentiment (optional): output shape (batch, C) logits
        (softmax handled outside)
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        pad_idx=0,
        dropout=0.3,
        num_classes=1,
        task="binary",
        **kwargs,
    ):
        super().__init__()

        self.pad_idx = int(pad_idx)

        # If a checkpoint says num_classes > 1, it must be multiclass
        try:
            nc = int(num_classes)
        except Exception:
            nc = 1

        if task not in ("binary", "multiclass"):
            task = "multiclass" if nc > 1 else "binary"

        # Critical safety:
        # if task says binary but num_classes is > 1, override to multiclass
        if task == "binary" and nc > 1:
            task = "multiclass"

        self.task = task
        self.num_classes = nc if self.task == "multiclass" else 1

        self.config = {
            "vocab_size": int(vocab_size),
            "embed_dim": int(embed_dim),
            "hidden_dim": int(hidden_dim),
            "pad_idx": int(pad_idx),
            "dropout": float(dropout),
            "num_classes": int(self.num_classes),
            "task": str(self.task),
        }

        # Layers
        self.embedding = nn.Embedding(int(vocab_size), int(embed_dim), padding_idx=self.pad_idx)
        self.fc1 = nn.Linear(int(embed_dim), int(hidden_dim))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))

        out_dim = 1 if self.task == "binary" else self.num_classes
        self.fc2 = nn.Linear(int(hidden_dim), int(out_dim))

        self.sigmoid = nn.Sigmoid()

    def forward(self, text_ids):
        embedded = self.embedding(text_ids)

        mask = (text_ids != self.pad_idx).float()
        mask_expanded = mask.unsqueeze(2)
        masked_embeddings = embedded * mask_expanded
        summed = masked_embeddings.sum(dim=1)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = summed / lengths

        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        if self.task == "binary":
            return self.sigmoid(x)

        return x  # logits

    def get_embeddings(self):
        return self.embedding.weight.detach()


def save_model(model, filepath):
    torch.save({"config": model.config, "state_dict": model.state_dict()}, filepath)


def load_model(filepath, device="cpu"):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    if not (isinstance(checkpoint, dict) and "config" in checkpoint and "state_dict" in checkpoint):
        raise ValueError("Unsupported checkpoint format. Expected {'config', 'state_dict'}.")

    config = dict(checkpoint["config"])
    state_dict = checkpoint["state_dict"]

    # Make older checkpoints safe
    if "task" not in config:
        # infer from num_classes if present
        try:
            nc = int(config.get("num_classes", 1))
        except Exception:
            nc = 1
        config["task"] = "multiclass" if nc > 1 else "binary"
    if "num_classes" not in config:
        config["num_classes"] = 1

    model = SentimentClassifier(**config)
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    print("=" * 65)
    print("  ACTIVITY 2: Model Architecture — Demo & Inspection")
    print("=" * 65)

    VOCAB_SIZE = 150
    EMBED_DIM = 64
    HIDDEN_DIM = 32

    model = SentimentClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
    print("\n[Architecture]")
    print(model)

    print("\n[Save/Load Test]")
    temp_model_path = os.path.join(tempfile.gettempdir(), "test_model.pt")
    save_model(model, temp_model_path)
    loaded = load_model(temp_model_path)

    fake_input = torch.tensor([[2, 5, 8, 3, 0]], dtype=torch.long)
    with torch.no_grad():
        out1 = model(fake_input)
        out2 = loaded(fake_input)

    print("Outputs match:", torch.allclose(out1, out2))
    print("=" * 65)