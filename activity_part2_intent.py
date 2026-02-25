
"""
============================================================================
AIT-204 Deep Learning | Topic 5: Advanced NLP
ACTIVITY ��� PART 2: Intent Extraction with Multi-Task Learning
============================================================================

PURPOSE:
    Extend the multi-level sentiment model into a DUAL-HEAD classifier
    that simultaneously predicts:
        (1) Fine-grained sentiment   (7 classes from Part 1)
        (2) Communicative intent     (8 classes: complaint, recommendation, ...)

    This introduces MULTI-TASK LEARNING ��� a single shared encoder feeds two
    independent classification heads. Both tasks are trained jointly.

DURATION: ~75 minutes  (see time targets per section below)

PREREQUISITES:
    Part 1 of this activity (multi-level sentiment) should be understood first.
    This activity imports preprocessing from Topic 4 Activity 1.

BRIDGING THE GAP (Concept -> Algorithm -> Code):
    ---------------------------------------------------------------
    PROBLEM: Sentiment alone doesn't tell you what action to take.
        "Don't waste your money on this film" ��� NEGATIVE (sentiment)
        But is it a COMPLAINT? A WARNING to others? A RECOMMENDATION?
        These are different communicative intentions.

    WHAT IS INTENT?
        Intent = the communicative PURPOSE behind an utterance.
        Examples:
          "The service was terrible, I want a refund." ��� COMPLAINT
          "You absolutely have to see this film."      ��� RECOMMENDATION
          "When does the movie start?"                 ��� INQUIRY
          "The director did an outstanding job."       ��� PRAISE
          "Avoid this at all costs."                   ��� WARNING
          "Oh sure, because everyone loves a 3-hr snooze." ��� SARCASM
          "Unlike the original, this is far better."  ��� COMPARISON
          "The film was released in 2023."             ��� NEUTRAL REPORT

    SOLUTION ��� MULTI-TASK LEARNING:
        ���������������������������������������������������������������������������������������������������������������������������������������������������������������������������������
        ���  Word IDs                                               ���
        ���      ���                                                  ���
        ���  [Embedding]    ��� shared lookup table                   ���
        ���      ���                                                  ���
        ���  [Avg Pooling]  ��� shared sequence encoder               ���
        ���      ���                                                  ���
        ���  [FC ��� ReLU ��� Dropout]  ��� shared representation        ���
        ���      ���               ���                                  ���
        ���  [Sentiment Head]   [Intent Head]   ��� task-specific     ���
        ���  FC(hidden, 7)      FC(hidden, 8)                       ���
        ���      ���               ���                                  ���
        ���  sentiment logits   intent logits                       ���
        ���������������������������������������������������������������������������������������������������������������������������������������������������������������������������������

        Combined Loss:
            total_loss = �� �� sentiment_loss + (1-��) �� intent_loss
            where �� ��� (0, 1) controls task priority

    WHY MULTI-TASK LEARNING?
        Sharing the encoder forces the model to learn features that are
        useful for BOTH tasks simultaneously. This can improve generalization,
        especially when one task has limited training data.

WHAT YOU'LL IMPLEMENT:  (8 TODOs, ~9 min each)
    TODO 1: Explore dataset ��� distributions for both tasks (10 min)
    TODO 2: Add intent head to DualHeadClassifier (10 min)
    TODO 3: Complete forward() to return (sentiment_logits, intent_logits) (5 min)
    TODO 4: Compute combined loss using alpha weighting (10 min)
    TODO 5: Track BOTH accuracies in the training loop (10 min)
    TODO 6: Evaluate both tasks with confusion matrix + F1 (15 min)
    TODO 7: Build joint inference ��� return sentiment AND intent (10 min)
    TODO 8: Experiment with alpha ��� which task benefits from higher weight? (5 min)

RUN THIS FILE:  python activity_part2_intent.py
============================================================================
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ������ Import preprocessing from Topic 4 ���������������������������������������������������������������������������������������������������������������
TOPIC4_PATH = os.path.join(os.path.dirname(__file__), "..", "Topic4_NLP")
sys.path.insert(0, TOPIC4_PATH)

try:
    from activity1_preprocessing import (
        Vocabulary, clean_text, tokenize,
        pad_sequence, preprocess_dataset, preprocess_for_model
    )
    print("[OK] Imported preprocessing from Topic 4 Activity 1")
except ImportError as e:
    print(f"[ERROR] Could not import from Topic 4: {e}")
    sys.exit(1)


# =========================================================================
# LABEL DEFINITIONS
# =========================================================================

SENTIMENT_LABELS = {
    0: "Extremely Negative",
    1: "Very Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Very Positive",
    6: "Extremely Positive",
}
NUM_SENTIMENTS = len(SENTIMENT_LABELS)

INTENT_LABELS = {
    0: "Complaint",
    1: "Recommendation",
    2: "Inquiry",
    3: "Praise",
    4: "Warning",
    5: "Sarcasm",
    6: "Comparison",
    7: "Neutral Report",
}
NUM_INTENTS = len(INTENT_LABELS)


# =========================================================================
# DATASET: PAIRED (text, sentiment_class, intent_class)
# =========================================================================
# Each example is a (text, sentiment_id, intent_id) triple.
# This is the key structure for multi-task learning.
#
# Notice: negative sentiment + complaint intent often co-occur,
#         but not always (a neutral report can describe a negative event).
# =========================================================================

MULTI_TASK_DATA = [
    # ������ COMPLAINT (intent=0) ������������������������������������������������������������������������������������������������������������������������������������������������
    # Complaints are typically negative in sentiment but not always extreme
    ("The customer service was absolutely terrible I want my money back immediately", 0, 0),
    ("This product failed after two days and no one will help fix it", 1, 0),
    ("I paid full price for a broken experience this is unacceptable", 1, 0),
    ("The staff were rude dismissive and completely unhelpful throughout", 0, 0),
    ("I have been waiting three weeks for a refund this is outrageous", 0, 0),
    ("The service was poor and my complaint has been ignored for two months", 1, 0),
    ("I cannot believe how badly this company has treated its customers", 0, 0),
    ("My experience was terrible and I demand a full refund right away", 0, 0),
    ("The product quality is far below what was advertised a total disappointment", 2, 0),
    ("Shocking lack of professionalism from the entire team I am disgusted", 0, 0),
    ("I am deeply unhappy with this service and will be escalating my complaint", 1, 0),
    ("This is the worst customer experience I have ever had completely unacceptable", 0, 0),

    # ������ RECOMMENDATION (intent=1) ������������������������������������������������������������������������������������������������������������������������������������
    # Recommendations are typically positive and directed at others
    ("You absolutely have to see this film it is completely unmissable", 6, 1),
    ("I would strongly recommend this product to anyone looking for quality", 5, 1),
    ("If you enjoy this genre then this film is essential viewing for you", 5, 1),
    ("Everyone should try this service it has made my daily routine so much easier", 5, 1),
    ("I cannot recommend this book highly enough it is genuinely life-changing", 6, 1),
    ("This is a must-see for any fan of great storytelling and beautiful cinema", 6, 1),
    ("Tell all your friends about this restaurant they will absolutely love it", 5, 1),
    ("I would recommend this to anyone who wants a solid dependable product", 4, 1),
    ("Worth every penny I recommend buying this without any hesitation at all", 5, 1),
    ("Highly recommended this course will transform how you think about the subject", 5, 1),
    ("If you get the chance to see this film please do not miss it", 5, 1),
    ("I recommend giving this a try you will be pleasantly surprised by it", 4, 1),

    # ������ INQUIRY (intent=2) ������������������������������������������������������������������������������������������������������������������������������������������������������������
    # Inquiries are typically neutral ��� questions seeking information
    ("What time does the evening screening start at your cinema please", 3, 2),
    ("Can you tell me how long the film runs before I book my tickets", 3, 2),
    ("Is there a student discount available for weekend performances", 3, 2),
    ("Does this product come with a warranty and if so how long is it", 3, 2),
    ("How many episodes are in the first season of this new series", 3, 2),
    ("Where can I find more information about the director of this film", 3, 2),
    ("What are the main differences between the two available versions", 3, 2),
    ("Is this product available in a larger size or just the standard", 3, 2),
    ("Could you explain what the return policy is for online purchases", 3, 2),
    ("Are there any subtitles available for the international release", 3, 2),
    ("How many stars did the critics give this film at its premiere", 3, 2),
    ("What is the age rating for this film and is it suitable for children", 3, 2),

    # ������ PRAISE (intent=3) ������������������������������������������������������������������������������������������������������������������������������������������������������������
    # Praise is typically very positive and directed at a specific person or work
    ("The director did an absolutely outstanding job bringing this story to life", 6, 3),
    ("The lead actor delivered one of the most powerful performances I have seen", 6, 3),
    ("The writing was exceptional every scene felt purposeful and brilliantly crafted", 6, 3),
    ("Incredible cinematography that elevated every single scene beyond expectation", 5, 3),
    ("The team behind this product deserves huge credit for such thoughtful design", 5, 3),
    ("A masterclass in storytelling from a filmmaker working at the very top", 6, 3),
    ("The composer created a breathtaking score that perfectly complemented the visuals", 5, 3),
    ("What an extraordinary achievement by everyone involved in this production", 6, 3),
    ("The editing was superb creating a perfect rhythm and pace throughout", 5, 3),
    ("The writing team crafted some of the sharpest dialogue I have ever heard", 5, 3),
    ("Outstanding work from the supporting cast who added real depth to the story", 5, 3),
    ("The special effects team did remarkable work creating a fully believable world", 5, 3),

    # ������ WARNING (intent=4) ���������������������������������������������������������������������������������������������������������������������������������������������������������
    # Warnings are negative but directed outward ��� cautioning others
    ("Do not waste your money on this awful sequel trust me and avoid it", 1, 4),
    ("Stay away from this product the quality is terrible and it breaks quickly", 1, 4),
    ("I am warning everyone this film contains very disturbing content throughout", 2, 4),
    ("Please do not make the same mistake I made by buying this product", 2, 4),
    ("Avoid this at all costs it is a complete waste of your time and money", 1, 4),
    ("I am begging you do not see this film you will deeply regret it", 1, 4),
    ("Be warned this service is extremely unreliable and support is non-existent", 0, 4),
    ("Do not trust the positive reviews this product is genuinely awful", 1, 4),
    ("Run from this film it is a catastrophically bad experience from start", 0, 4),
    ("Seriously don't bother with this the negative reviews are all accurate", 1, 4),
    ("A word of warning the hidden fees are completely outrageous and unexpected", 2, 4),
    ("Before you buy please read the one-star reviews they are telling the truth", 2, 4),

    # ������ SARCASM (intent=5) ������������������������������������������������������������������������������������������������������������������������������������������������������������
    # Sarcasm is negative in tone but uses positive surface language
    ("Oh sure because everyone loves paying fifty dollars for a two-hour nap", 2, 5),
    ("Wow what a totally original idea I definitely haven't seen this exact plot", 1, 5),
    ("Absolutely riveting three hours of my life I didn't need anyway really", 0, 5),
    ("Oh how innovative another generic sequel with exactly zero new ideas", 1, 5),
    ("Yes this is exactly what audiences were desperately clamouring for another reboot", 1, 5),
    ("Truly groundbreaking work there completely unchanged from every other film", 2, 5),
    ("A masterpiece for sure if your standards are incredibly extremely low", 1, 5),
    ("Oh great another predictable ending no surprises whatsoever thanks so much", 1, 5),
    ("Such refreshing originality I have never seen anything like this before today", 2, 5),
    ("Brilliant idea spending that much on something that broke after one week", 0, 5),
    ("Oh yes very subtle messaging there I definitely did not see that coming", 2, 5),
    ("Shocking only took three sequels to completely ruin a perfectly good concept", 1, 5),

    # ������ COMPARISON (intent=6) ������������������������������������������������������������������������������������������������������������������������������������������������
    # Comparisons are mixed in sentiment and evaluate relative to something else
    ("Unlike the dreadful original this sequel is actually a significant improvement", 4, 6),
    ("This version is far superior to the remake they released two years ago", 5, 6),
    ("The new model is so much better than its predecessor in almost every way", 5, 6),
    ("Compared to their last album this is a disappointing and lazy effort", 2, 6),
    ("The book is much richer and more nuanced than the film adaptation", 4, 6),
    ("The director's earlier work was far stronger than this underwhelming effort", 1, 6),
    ("While the original was a masterpiece this sequel falls far short of it", 2, 6),
    ("This is a massive improvement over the buggy disaster of the first version", 4, 6),
    ("The acting here is considerably better than in the previous installment", 4, 6),
    ("Not as strong as the debut album but still a worthwhile and enjoyable effort", 4, 6),
    ("The service has improved greatly compared to how it was two years ago", 4, 6),
    ("This product costs twice as much but delivers at least four times the value", 5, 6),

    # ������ NEUTRAL REPORT (intent=7) ���������������������������������������������������������������������������������������������������������������������������������������
    # Neutral reports state facts or observations without strong evaluative language
    ("The film was released in 2023 and stars three academy award winners", 3, 7),
    ("This is the third installment in the franchise which began in 2015", 3, 7),
    ("The running time is approximately two hours and twenty minutes long", 3, 7),
    ("The director previously worked on several acclaimed independent productions", 3, 7),
    ("The film was shot on location across five different countries worldwide", 3, 7),
    ("This product was launched last year and has sold over a million units", 3, 7),
    ("The lead actor has appeared in more than twenty films since their debut", 3, 7),
    ("The series is based on a bestselling novel published in the early nineties", 3, 7),
    ("This film won three awards at the festival including best picture", 4, 7),
    ("The studio invested heavily in the production which ran for two years", 3, 7),
    ("The composer wrote the original score over a period of six months", 3, 7),
    ("This is the director's first major commercial release after a decade away", 3, 7),
]

# =========================================================================
# RECOMMENDED DATASETS FOR INTENT CLASSIFICATION
# =========================================================================
# Replace MULTI_TASK_DATA with one of these for a real-world model:
#
#   pip install datasets
#
#   1. CLINC150 (150 intents, 10 domains, includes out-of-scope)
#      from datasets import load_dataset
#      ds = load_dataset("clinc_oos", "plus")  # 22,500 train, 3,100 test
#
#   2. Banking77 (77 banking-domain intents, 13,083 examples)
#      ds = load_dataset("banking77")
#
#   3. HWU64 (64 intents, 21 domains ��� smart home & personal assistant)
#      Available at: https://github.com/xliuhw/NLU-Evaluation-Data
#
#   4. SNIPS NLU (7 intents: AddToPlaylist, BookRestaurant, etc.)
#      Available at: https://github.com/snipsco/nlu-benchmark
#
#   5. ATIS (Airline Travel Information System, ~18 flight-related intents)
#      Available at: https://github.com/yvchen/JointSLU/tree/master/data
#
# For joint sentiment+intent (multi-task like this activity):
#   Use two separate datasets and align by topic, or annotate your own.
#   GoEmotions (emotions) + SNIPS (intents) is a common pairing.
# =========================================================================


# =========================================================================
# HYPERPARAMETERS
# =========================================================================
EMBED_DIM   = 64
HIDDEN_DIM  = 64
DROPOUT     = 0.4
MAX_LENGTH  = 20
BATCH_SIZE  = 16
EPOCHS      = 100
LR          = 0.001
ALPHA       = 0.5    # Weight for sentiment loss; (1-ALPHA) for intent loss
                     # ��=0.7 prioritizes sentiment; ��=0.3 prioritizes intent
TRAIN_SPLIT = 0.80


# =========================================================================
# STEP 1: EXPLORE BOTH TASK DISTRIBUTIONS                  [~10 minutes]
# =========================================================================

def explore_dataset(data):
    """
    Analyze label distributions for BOTH tasks in the multi-task dataset.

    Args:
        data: list of (text, sentiment_id, intent_id) triples
    """
    print("\n" + "=" * 65)
    print("  STEP 1: Multi-Task Dataset Exploration")
    print("=" * 65)

    # TODO 1a: Count examples per sentiment class
    # HINT: sentiment_counts = Counter(s for _, s, _ in data)
    sentiment_counts = Counter(s for _, s, _ in data)

    # TODO 1b: Count examples per intent class
    # HINT: intent_counts = Counter(i for _, _, i in data)
    intent_counts    = Counter(i for _, _, i in data)

    print(f"\n  Total examples: {len(data)}\n")

    print(f"  Sentiment distribution:")
    for class_id, name in SENTIMENT_LABELS.items():
        count = sentiment_counts.get(class_id, 0)
        print(f"    {class_id} {name:<25} {count:>3} | {'���' * count}")

    print(f"\n  Intent distribution:")
    for class_id, name in INTENT_LABELS.items():
        count = intent_counts.get(class_id, 0)
        print(f"    {class_id} {name:<20} {count:>3} | {'���' * count}")

    # TODO 1c: Identify co-occurring patterns
    # Which sentiment is most common with each intent?
    # HINT: Build a (NUM_INTENTS �� NUM_SENTIMENTS) count matrix
    #   cooccurrence[intent_id][sentiment_id] += 1
    #   Then print the dominant sentiment per intent
    cooccurrence = np.zeros((NUM_INTENTS, NUM_SENTIMENTS))
    for _, s, i in data:
        cooccurrence[i][s] += 1
    print(f"\n  Dominant sentiment per intent:")
    for i in range(NUM_INTENTS):
        dominant_sentiment = np.argmax(cooccurrence[i])
        count = cooccurrence[i][dominant_sentiment]
        print(f"    Intent {i} ({INTENT_LABELS[i]}) -> Sentiment {dominant_sentiment} ({SENTIMENT_LABELS[dominant_sentiment]}) [{count} examples]")

    # TODO 1d: Compute class weights for both tasks
    # Use the same formula as Part 1:
    #   weight_i = total / (num_classes * count_i)
    total = len(data)
    sent_weights   = torch.tensor([total / (NUM_SENTIMENTS * sentiment_counts.get(i, 1)) for i in range(NUM_SENTIMENTS)], dtype=torch.float)
    intent_weights = torch.tensor([total / (NUM_INTENTS * intent_counts.get(i, 1)) for i in range(NUM_INTENTS)], dtype=torch.float)
    return sent_weights, intent_weights


# =========================================================================
# STEP 2: DUAL-HEAD CLASSIFIER                             [~10 minutes]
# =========================================================================
# Architecture:
#   Shared trunk:  Embedding ��� AvgPool ��� FC_shared ��� ReLU ��� Dropout
#   Sentiment head: FC(hidden, NUM_SENTIMENTS)  [Task A]
#   Intent head:    FC(hidden, NUM_INTENTS)     [Task B]
#
# KEY INSIGHT: The shared trunk learns a representation useful for BOTH.
# If sentiment depends on adjectives and intent on verb phrases, the shared
# layer is encouraged to capture both types of features simultaneously.
# =========================================================================

class DualHeadClassifier(nn.Module):
    """
    Multi-task sentiment + intent classifier with a shared encoder.

    The shared trunk processes text into a context vector.
    Two task-specific linear heads map this vector to their respective classes.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_sentiments, num_intents, pad_idx=0, dropout=0.3):
        """
        Args:
            vocab_size    (int): Vocabulary size
            embed_dim     (int): Embedding dimension
            hidden_dim    (int): Hidden layer neurons (shared)
            num_sentiments(int): Number of sentiment classes
            num_intents   (int): Number of intent classes
            pad_idx       (int): <PAD> token index
            dropout     (float): Dropout rate
        """
        super(DualHeadClassifier, self).__init__()

        self.pad_idx = pad_idx
        self.config  = {
            "vocab_size":     vocab_size,
            "embed_dim":      embed_dim,
            "hidden_dim":     hidden_dim,
            "num_sentiments": num_sentiments,
            "num_intents":    num_intents,
            "pad_idx":        pad_idx,
            "dropout":        dropout,
        }

        # ������ SHARED TRUNK (processes text for both tasks) ���������������������������������������������������������
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fc_shared  = nn.Linear(embed_dim, hidden_dim)
        self.relu       = nn.ReLU()
        self.dropout    = nn.Dropout(dropout)

        # ������ TASK-SPECIFIC HEADS ���������������������������������������������������������������������������������������������������������������������������������
        # Sentiment head: maps shared representation ��� sentiment logits
        self.sentiment_head = nn.Linear(hidden_dim, num_sentiments)

        # TODO 2: Add the intent classification head
        # It maps the shared representation ��� intent logits
        # Same shape pattern as sentiment_head but for NUM_INTENTS outputs
        #
        # HINT: self.intent_head = nn.Linear(hidden_dim, num_intents)
        self.intent_head = nn.Linear(hidden_dim, num_intents)

    def forward(self, text_ids):
        """
        Forward pass through shared trunk then both task heads.

        Args:
            text_ids: tensor of shape (batch_size, seq_len)
        Returns:
            tuple: (sentiment_logits, intent_logits)
                   Both are shape (batch_size, respective_num_classes)
        """
        # ������ Shared Trunk ���������������������������������������������������������������������������������������������������������������������������������������������������������
        embedded = self.embedding(text_ids)          # (batch, seq_len, embed)

        # Masked average pooling (same as Part 1)
        mask             = (text_ids != self.pad_idx).float()
        mask_expanded    = mask.unsqueeze(2)
        masked_embeddings = embedded * mask_expanded
        summed           = masked_embeddings.sum(dim=1)
        lengths          = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled           = summed / lengths                   # (batch, embed)

        # Shared FC layer ��� produces a representation useful for both tasks
        shared = self.dropout(self.relu(self.fc_shared(pooled)))  # (batch, hidden)

        # ������ Task-Specific Heads ������������������������������������������������������������������������������������������������������������������������������������
        # Sentiment: shared ��� K_sentiment logits
        sentiment_logits = self.sentiment_head(shared)       # (batch, num_sentiments)

        # TODO 3: Compute intent logits
        # Pass `shared` through self.intent_head (no activation ��� raw logits)
        # HINT: intent_logits = self.intent_head(shared)
        intent_logits = self.intent_head(shared)

        # TODO 3b: Return BOTH outputs as a tuple
        # The training loop unpacks this as:
        #   sentiment_logits, intent_logits = model(batch_x)
        # HINT: return sentiment_logits, intent_logits
        return sentiment_logits, intent_logits


def save_dual_model(model, filepath):
    """Save the dual-head model to disk."""
    torch.save({"config": model.config, "state_dict": model.state_dict()}, filepath)
    print(f"  Model saved ��� {filepath}")


def load_dual_model(filepath, device="cpu"):
    """Load a saved dual-head model."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model = DualHeadClassifier(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


# =========================================================================
# STEP 3: DATA PREPARATION
# =========================================================================

def prepare_multitask_data(data, vocab, max_length, fit_vocab=False):
    """
    Prepare data for multi-task training.

    Args:
        data:      list of (text, sentiment_id, intent_id) triples
        vocab:     Vocabulary instance
        max_length (int): Fixed sequence length
        fit_vocab (bool): If True, build vocab from this data

    Returns:
        (X, y_sent, y_intent): Three tensors, all aligned by example index
    """
    texts      = [text for text, _, _ in data]
    y_sent     = [s   for _,    s, _ in data]
    y_intent   = [i   for _,    _, i in data]

    all_tokens = [tokenize(clean_text(text)) for text in texts]

    if fit_vocab:
        vocab.build(all_tokens)

    encoded = [vocab.encode(tokens) for tokens in all_tokens]
    padded  = [pad_sequence(seq, max_length) for seq in encoded]

    X        = torch.tensor(padded,    dtype=torch.long)
    y_sent   = torch.tensor(y_sent,    dtype=torch.long)
    y_intent = torch.tensor(y_intent,  dtype=torch.long)

    return X, y_sent, y_intent


# =========================================================================
# STEP 4: MULTI-TASK TRAINING LOOP                         [~15 minutes]
# =========================================================================
# The combined loss balances both tasks:
#   total_loss = �� �� sentiment_loss + (1-��) �� intent_loss
#
# When �� = 0.5: tasks weighted equally
# When �� = 0.7: model prioritizes getting sentiment right
# When �� = 0.3: model prioritizes getting intent right
#
# After training, compare the val accuracy for each task.
# =========================================================================

def train_multitask(model, train_X, train_y_sent, train_y_intent,
                    val_X, val_y_sent, val_y_intent,
                    sent_weights=None, intent_weights=None,
                    alpha=ALPHA, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    """
    Multi-task training loop producing TWO accuracies per epoch.

    Args:
        model:                 DualHeadClassifier instance
        train_X, val_X:        Encoded text tensors (torch.long)
        train_y_sent, val_y_sent:   Sentiment label tensors (torch.long)
        train_y_intent, val_y_intent: Intent label tensors (torch.long)
        sent_weights:          Class weights for sentiment loss
        intent_weights:        Class weights for intent loss
        alpha (float):         Weight for sentiment loss [0, 1]
        epochs, lr, batch_size: Training hyperparameters

    Returns:
        dict with per-epoch history for both tasks
    """
    # TODO 4a: Define TWO separate CrossEntropyLoss functions
    # One for sentiment (using sent_weights) and one for intent (intent_weights)
    # HINT:
    #   sent_criterion   = nn.CrossEntropyLoss(weight=sent_weights)
    #   intent_criterion = nn.CrossEntropyLoss(weight=intent_weights)
    sent_criterion   = nn.CrossEntropyLoss(weight=sent_weights)
    intent_criterion = nn.CrossEntropyLoss(weight=intent_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_sent_loss": [],  "train_intent_loss": [],
        "train_sent_acc":  [],  "train_intent_acc":  [],
        "val_sent_acc":    [],  "val_intent_acc":    [],
    }
    n_train = len(train_X)

    print(f"\n  Multi-task training for {epochs} epochs | ��={alpha:.1f}")
    print(f"  {'Ep':>4}  {'SentLoss':>9}  {'SentAcc':>8}  "
          f"{'IntLoss':>8}  {'IntAcc':>8}  {'VSentAcc':>9}  {'VIntAcc':>8}")
    print(f"  {'-'*63}")

    for epoch in range(1, epochs + 1):
        model.train()
        ep_sent_loss = ep_intent_loss = 0.0
        ep_sent_correct = ep_intent_correct = 0

        indices = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx    = indices[start : start + batch_size]
            bx     = train_X[idx]
            by_s   = train_y_sent[idx]
            by_i   = train_y_intent[idx]

            optimizer.zero_grad()

            # TODO 4b: Run the forward pass
            # The model returns a TUPLE: (sentiment_logits, intent_logits)
            # Unpack both outputs.
            # HINT: sent_logits, intent_logits = model(bx)
            sent_logits, intent_logits = model(bx)

            # TODO 4c: Compute both task losses
            # HINT:
            #   sent_loss   = sent_criterion(sent_logits, by_s)
            #   intent_loss = intent_criterion(intent_logits, by_i)
            sent_loss   = sent_criterion(sent_logits, by_s)
            intent_loss = intent_criterion(intent_logits, by_i)

            # TODO 4d: Combine losses with alpha weighting
            # total_loss = �� �� sent_loss + (1-��) �� intent_loss
            # This single value drives the backward pass for BOTH heads
            # HINT: total_loss = alpha * sent_loss + (1 - alpha) * intent_loss
            total_loss  = alpha * sent_loss + (1 - alpha) * intent_loss

            total_loss.backward()
            optimizer.step()

            # TODO 5a: Track sentiment accuracy for this batch
            # HINT: sent_preds = sent_logits.argmax(dim=1)
            #        ep_sent_correct += (sent_preds == by_s).sum().item()
            sent_preds = sent_logits.argmax(dim=1)
            ep_sent_correct   += (sent_preds == by_s).sum().item()

            # TODO 5b: Track intent accuracy for this batch
            # HINT: Same pattern as sentiment above but for intent
            intent_preds = intent_logits.argmax(dim=1)
            ep_intent_correct += (intent_preds == by_i).sum().item()

            ep_sent_loss   += sent_loss.item()   * len(idx)
            ep_intent_loss += intent_loss.item() * len(idx)

        train_sent_acc   = ep_sent_correct   / n_train
        train_intent_acc = ep_intent_correct / n_train
        train_sent_loss  = ep_sent_loss   / n_train
        train_intent_loss= ep_intent_loss / n_train

        # ������ Validation ���������������������������������������������������������������������������������������������������������������������������������������������������������������
        model.eval()
        with torch.no_grad():
            val_sent_logits, val_intent_logits = model(val_X)

            # TODO 5c: Compute validation accuracy for BOTH tasks
            # HINT:
            #   val_sent_acc   = (val_sent_logits.argmax(1)   == val_y_sent).float().mean().item()
            #   val_intent_acc = (val_intent_logits.argmax(1) == val_y_intent).float().mean().item()
            val_sent_acc   = (val_sent_logits.argmax(1)   == val_y_sent).float().mean().item()
            val_intent_acc = (val_intent_logits.argmax(1) == val_y_intent).float().mean().item()

        history["train_sent_loss"].append(train_sent_loss)
        history["train_intent_loss"].append(train_intent_loss)
        history["train_sent_acc"].append(train_sent_acc)
        history["train_intent_acc"].append(train_intent_acc)
        history["val_sent_acc"].append(val_sent_acc)
        history["val_intent_acc"].append(val_intent_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:>4}  {train_sent_loss:>9.4f}  {train_sent_acc:>7.1%}  "
                  f"{train_intent_loss:>8.4f}  {train_intent_acc:>7.1%}  "
                  f"{val_sent_acc:>8.1%}  {val_intent_acc:>8.1%}")

    return history


def plot_multitask_training(history):
    """Plot training curves for both tasks side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot([a * 100 for a in history["train_sent_acc"]],   label="Train Sentiment")
    axes[0].plot([a * 100 for a in history["val_sent_acc"]],     label="Val Sentiment",   linestyle="--")
    axes[0].plot([a * 100 for a in history["train_intent_acc"]], label="Train Intent",    linestyle="-.")
    axes[0].plot([a * 100 for a in history["val_intent_acc"]],   label="Val Intent",      linestyle=":")
    axes[0].axhline(y=100/NUM_SENTIMENTS, color="red",   linestyle=":", alpha=0.5, label="Sent baseline")
    axes[0].axhline(y=100/NUM_INTENTS,    color="orange",linestyle=":", alpha=0.5, label="Intent baseline")
    axes[0].set_title("Accuracy ��� Both Tasks"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_sent_loss"],   label="Sentiment Loss")
    axes[1].plot(history["train_intent_loss"], label="Intent Loss", linestyle="--")
    axes[1].set_title("Loss ��� Both Tasks"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves_multitask.png", dpi=120)
    plt.close()
    print("  Saved: training_curves_multitask.png")


# =========================================================================
# STEP 5: EVALUATION                                       [~15 minutes]
# =========================================================================

def evaluate_task(true_labels, pred_labels, label_dict, task_name):
    """
    Print confusion matrix and per-class F1 for one task.

    Args:
        true_labels (np.array): Ground-truth class indices
        pred_labels (np.array): Predicted class indices
        label_dict (dict):      {class_id: class_name}
        task_name  (str):       e.g. "Sentiment" or "Intent"
    """
    num_classes = len(label_dict)
    accuracy = (true_labels == pred_labels).mean()
    print(f"\n  [{task_name}] Accuracy: {accuracy:.2%}")

    # TODO 6a: Build the confusion matrix
    # Same approach as Part 1, now applied to both tasks
    # HINT: confusion = np.zeros((num_classes, num_classes), dtype=int)
    #        for t, p in zip(true_labels, pred_labels): confusion[t][p] += 1
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        confusion[t][p] += 1

    # Print compact confusion matrix
    names = [label_dict[i][:6] for i in range(num_classes)]
    print(f"\n  Confusion Matrix [{task_name}]  (rows=actual, cols=predicted):")
    header = " " * 10 + "  ".join(f"{n:>6}" for n in names)
    print("  " + header)
    for i, row in enumerate(confusion):
        row_str = "  ".join(f"{v:>6}" for v in row)
        print(f"    {names[i]:>6} | {row_str}")

    # TODO 6b: Compute and print per-class F1
    # (Same formula as Part 1)
    print(f"\n  Per-class F1 [{task_name}]:")
    print(f"    {'Class':<22}  {'F1':>6}  {'Support':>7}")
    print(f"    {'-'*38}")
    for i in range(num_classes):
        tp        = confusion[i, i]
        col_sum   = confusion[:, i].sum()
        row_sum   = confusion[i, :].sum()
        precision = tp / col_sum if col_sum > 0 else 0.0
        recall    = tp / row_sum if row_sum > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
        support   = row_sum
        print(f"    {label_dict[i]:<22}  {f1:>6.2%}  {support:>7}")


def evaluate_both_tasks(model, X, y_sent, y_intent, split_name="Validation"):
    """Evaluate the dual-head model on both tasks simultaneously."""
    model.eval()
    with torch.no_grad():
        sent_logits, intent_logits = model(X)
        sent_preds   = sent_logits.argmax(dim=1).numpy()
        intent_preds = intent_logits.argmax(dim=1).numpy()

    print(f"\n{'='*65}")
    print(f"  EVALUATION: {split_name}")
    print(f"{'='*65}")

    evaluate_task(y_sent.numpy(),   sent_preds,   SENTIMENT_LABELS, "Sentiment")
    evaluate_task(y_intent.numpy(), intent_preds, INTENT_LABELS,    "Intent")


# =========================================================================
# STEP 6: JOINT INFERENCE                                  [~10 minutes]
# =========================================================================

def predict_joint(text, model, vocab, max_length=MAX_LENGTH):
    """
    Run a single text through the dual-head model and return both predictions.

    Args:
        text       (str):   Raw input text
        model:              DualHeadClassifier (trained)
        vocab:              Vocabulary used at training time
        max_length (int):   Sequence length (must match training)

    Returns:
        dict with keys: sentiment_class, sentiment_name, sentiment_conf,
                        intent_class, intent_name, intent_conf,
                        sentiment_probs, intent_probs
    """
    model.eval()
    with torch.no_grad():
        tensor = preprocess_for_model(text, vocab, max_length)  # (1, max_length)

        # TODO 7a: Run the forward pass and unpack both logits
        # HINT: sent_logits, intent_logits = model(tensor)
        sent_logits, intent_logits = model(tensor)

        # TODO 7b: Convert logits ��� probabilities for both tasks
        # HINT: sent_probs   = torch.softmax(sent_logits,   dim=1).squeeze()
        #        intent_probs = torch.softmax(intent_logits, dim=1).squeeze()
        sent_probs   = torch.softmax(sent_logits,   dim=1).squeeze()
        intent_probs = torch.softmax(intent_logits, dim=1).squeeze()

        # TODO 7c: Find the top predicted class for each task
        # HINT: sent_class   = sent_probs.argmax().item()
        #        intent_class = intent_probs.argmax().item()
        sent_class   = sent_probs.argmax().item()
        intent_class = intent_probs.argmax().item()

    return {
        "sentiment_class": sent_class,
        "sentiment_name":  SENTIMENT_LABELS[sent_class],
        "sentiment_conf":  sent_probs[sent_class].item(),
        "intent_class":    intent_class,
        "intent_name":     INTENT_LABELS[intent_class],
        "intent_conf":     intent_probs[intent_class].item(),
        "sentiment_probs": {SENTIMENT_LABELS[i]: round(p, 3)
                            for i, p in enumerate(sent_probs.tolist())},
        "intent_probs":    {INTENT_LABELS[i]:    round(p, 3)
                            for i, p in enumerate(intent_probs.tolist())},
    }


# =========================================================================
# MAIN: Full Multi-Task Pipeline
# =========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  TOPIC 5 ��� PART 2: Intent Extraction (Multi-Task Learning)")
    print("=" * 65)

    # ������ 1. Explore dataset ���������������������������������������������������������������������������������������������������������������������������������������������������
    sent_weights, intent_weights = explore_dataset(MULTI_TASK_DATA)

    # ������ 2. Prepare data ������������������������������������������������������������������������������������������������������������������������������������������������������������
    print("\n" + "=" * 65)
    print("  STEP 2: Preparing Data")
    print("=" * 65)

    import random
    random.shuffle(MULTI_TASK_DATA)
    split       = int(len(MULTI_TASK_DATA) * TRAIN_SPLIT)
    train_data  = MULTI_TASK_DATA[:split]
    val_data    = MULTI_TASK_DATA[split:]

    vocab = Vocabulary(min_freq=1)
    train_X, train_y_sent, train_y_intent = prepare_multitask_data(
        train_data, vocab, MAX_LENGTH, fit_vocab=True
    )
    val_X, val_y_sent, val_y_intent = prepare_multitask_data(
        val_data, vocab, MAX_LENGTH
    )
    print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Vocab: {len(vocab)}")
    print(f"  train_X shape:       {train_X.shape}")
    print(f"  train_y_sent shape:  {train_y_sent.shape}  (sentiment labels, dtype={train_y_sent.dtype})")
    print(f"  train_y_intent shape:{train_y_intent.shape}  (intent labels,   dtype={train_y_intent.dtype})")

    # ������ 3. Build model ���������������������������������������������������������������������������������������������������������������������������������������������������������������
    print("\n" + "=" * 65)
    print("  STEP 3: Model Architecture")
    print("=" * 65)

    model = DualHeadClassifier(
        vocab_size     = len(vocab),
        embed_dim      = EMBED_DIM,
        hidden_dim     = HIDDEN_DIM,
        num_sentiments = NUM_SENTIMENTS,
        num_intents    = NUM_INTENTS,
        dropout        = DROPOUT,
    )
    print(f"\n  Model:\n{model}")
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total:,}")

    # Shape check
    dummy = torch.zeros(2, MAX_LENGTH, dtype=torch.long)
    with torch.no_grad():
        out = model(dummy)
    print(f"\n  Shape check: input {dummy.shape}")
    print(f"    sentiment logits: {out[0].shape}  (expected: (2, {NUM_SENTIMENTS}))")
    print(f"    intent logits:    {out[1].shape}  (expected: (2, {NUM_INTENTS}))")

    # ������ 4. Train ���������������������������������������������������������������������������������������������������������������������������������������������������������������������������������
    print("\n" + "=" * 65)
    print(f"  STEP 4: Multi-Task Training  (��={ALPHA})")
    print("=" * 65)
    print(f"  Sentiment baseline: {100/NUM_SENTIMENTS:.1f}%  ({NUM_SENTIMENTS} classes)")
    print(f"  Intent baseline:    {100/NUM_INTENTS:.1f}%  ({NUM_INTENTS} classes)")

    history = train_multitask(
        model,
        train_X, train_y_sent, train_y_intent,
        val_X,   val_y_sent,   val_y_intent,
        sent_weights=sent_weights,
        intent_weights=intent_weights,
        alpha=ALPHA,
    )
    plot_multitask_training(history)

    # ������ 5. Evaluate ������������������������������������������������������������������������������������������������������������������������������������������������������������������������
    evaluate_both_tasks(model, train_X, train_y_sent, train_y_intent, "Training")
    evaluate_both_tasks(model, val_X,   val_y_sent,   val_y_intent,   "Validation")

    # ������ 6. Save ������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������
    os.makedirs("saved_model_multitask", exist_ok=True)
    save_dual_model(model, "saved_model_multitask/model.pt")
    vocab.save("saved_model_multitask/vocab.json")
    with open("saved_model_multitask/config.json", "w") as f:
        json.dump({
            "max_length": MAX_LENGTH,
            "num_sentiments": NUM_SENTIMENTS,
            "num_intents": NUM_INTENTS,
            "sentiment_labels": SENTIMENT_LABELS,
            "intent_labels": INTENT_LABELS,
        }, f, indent=2)

    # ������ 7. Inference demo ������������������������������������������������������������������������������������������������������������������������������������������������������
    print("\n" + "=" * 65)
    print("  STEP 5: Joint Inference Demo")
    print("=" * 65)

    test_examples = [
        "I absolutely demand a full refund this product is completely broken",
        "You have to see this film it is a true masterpiece of cinema",
        "What time does the film start at the downtown location please",
        "Unlike the terrible original this sequel is surprisingly excellent",
        "Oh sure because everyone loves spending twenty dollars to fall asleep",
        "The film premiered in Cannes in 2023 to a standing ovation",
    ]

    for text in test_examples:
        result = predict_joint(text, model, vocab)
        print(f"\n  '{text[:60]}{'...' if len(text) > 60 else ''}'")
        print(f"  Sentiment: {result['sentiment_name']:<25} ({result['sentiment_conf']:.0%} confidence)")
        print(f"  Intent:    {result['intent_name']:<25} ({result['intent_conf']:.0%} confidence)")

    # ������ 8. Alpha experiment ������������������������������������������������������������������������������������������������������������������������������������������������
    print("\n" + "=" * 65)
    print("  TODO 8: Alpha Experiment (if time permits)")
    print("=" * 65)
    print("""
  Run this file again with different ALPHA values at the top:
      ALPHA = 0.2  ��� Strongly prioritize intent
      ALPHA = 0.5  ��� Equal weight (default)
      ALPHA = 0.8  ��� Strongly prioritize sentiment

  For each alpha, record val_sent_acc and val_intent_acc at the end.
  Fill in the table:

    Alpha | Val Sent Acc | Val Intent Acc | Observation
    ------|--------------|----------------|------------------
    0.2   |              |                |
    0.5   |              |                |
    0.8   |              |                |

  QUESTION: Does prioritizing one task always hurt the other?
  Why or why not? (Think about the shared trunk architecture)
  """)

    print("=" * 65)
    print("  PART 2 COMPLETE ��� Both activities finished!")
    print("=" * 65)


# ������ REFLECTION QUESTIONS ������������������������������������������������������������������������������������������������������������������������������������������������������
#
# 1. MULTI-TASK LEARNING: The shared encoder must learn features useful
#    for BOTH sentiment AND intent. Give an example of a word or phrase
#    that is informative for sentiment but not intent, and vice versa.
#    What does this suggest about the representation the encoder learns?
#
# 2. SARCASM: Look at the sarcasm examples in the dataset.
#    Sarcasm uses positive words to convey negative sentiment.
#    How does this challenge the embedding+pooling architecture?
#    What architectural change might help? (Hint: word order matters)
#
# 3. DATA EFFICIENCY: If you only had 100 labelled examples for intent,
#    but 10,000 labelled examples for sentiment, how could multi-task
#    learning help the intent task? (Hint: shared representations)
#
# 4. LABEL CORRELATION: Complaints tend to be negative.
#    Recommendations tend to be positive. Does this make the model's
#    job easier or harder? Could the model "cheat" by learning to
#    predict intent from sentiment alone without reading the text?
#
# 5. DEPLOYMENT: How would you integrate this dual-head model into
#    the Streamlit app from Topic 4? What UI elements would you add
#    to display intent alongside sentiment? Sketch a design.
#
# 6. BEYOND BINARY SENTIMENT IN PRODUCTION:
#    Amazon uses 1-5 star ratings. Netflix uses thumbs up/down.
#    Google uses 5 emoji reactions. What are the trade-offs between
#    a 2-class, 5-class, and 7-class sentiment system in production?
#    Consider: annotation cost, user experience, and model accuracy.