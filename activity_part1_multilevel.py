"""
============================================================================
AIT-204 Deep Learning | Topic 4 Extended: Advanced NLP
ACTIVITY ��� PART 1: Multi-Level Sentiment Analysis
============================================================================

PURPOSE:
    Extend the binary sentiment classifier from Topic 4 into a fine-grained,
    multi-class model that distinguishes 7 levels of sentiment:
        0 = Extremely Negative   3 = Neutral   6 = Extremely Positive
    You will modify the model output layer, switch to CrossEntropyLoss,
    and evaluate with a confusion matrix.

DURATION: ~75 minutes  (see time targets per section below)

PREREQUISITES:
    Complete Topic 4 (Activities 1-3) first. This activity imports your
    preprocessing module from Topic 4 and builds on its architecture.

BRIDGING THE GAP (Concept -> Algorithm -> Code):
    ---------------------------------------------------------------
    PROBLEM: Binary sentiment misses nuance.
        "I liked it a little"  -> POSITIVE (same as "masterpiece!")
        "It was okay"          -> NEGATIVE (same as "absolute disaster!")

    SOLUTION ��� ORDINAL MULTI-CLASS CLASSIFICATION:
        Instead of outputting 1 probability, output a vector of K scores.
        The highest score determines the predicted class.

        Binary (Topic 4):          Multi-Class (Topic 5):
        -----------------          ----------------------
        Input -> ... -> FC(hid,1)  Input -> ... -> FC(hid,7)
                     -> Sigmoid()               -> (argmax at inference)
        Output: 1 number (0���1)     Output: 7 numbers (logits)
        Loss: BCELoss              Loss: CrossEntropyLoss

    KEY DIFFERENCE ��� LOSS FUNCTION:
        BCELoss:          compares ONE probability to ONE binary label
        CrossEntropyLoss: compares K logits to ONE integer class index
            - Applies softmax internally
            - Expects labels as torch.long (integers 0 to K-1)

    KEY DIFFERENCE ��� ACCURACY CALCULATION:
        Binary:     (output > 0.5).float()   <- threshold
        Multi-class: output.argmax(dim=1)    <- pick highest logit

WHAT YOU'LL IMPLEMENT:  (7 TODOs, ~10 min each)
    TODO 1: Explore dataset ��� label distribution & class balance (10 min)
    TODO 2: Modify model ��� change output from 1 neuron to num_classes (10 min)
    TODO 3: Update forward() ��� remove sigmoid, return raw logits (5 min)
    TODO 4: Set up loss ��� CrossEntropyLoss + class weights (10 min)
    TODO 5: Fix training loop ��� argmax accuracy, label dtype (15 min)
    TODO 6: Evaluate ��� confusion matrix + per-class F1 (15 min)
    TODO 7: Inference ��� return class name + full confidence distribution (10 min)

RUN THIS FILE:  python activity_part1_multilevel.py
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

from activity_part2_intent import NUM_SENTIMENTS

# ������ Import preprocessing from Topic 4 ���������������������������������������������������������������������������������������������������������������
# We reuse the Vocabulary, clean_text, tokenize, pad_sequence, and
# preprocess_dataset functions you built in Activity 1.
# If this import fails, check that your Topic 4 folder is at the path below.
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
    print("  Make sure Topic4_NLP/ is in the parent folder of Topic5_AdvancedNLP/")
    sys.exit(1)


# =========================================================================
# DATASET: 7-CLASS SENTIMENT
# =========================================================================
# Labels:  0=Extremely Negative  1=Very Negative  2=Negative
#          3=Neutral
#          4=Positive  5=Very Positive  6=Extremely Positive
#
# This sample dataset is small enough to train in class.
# For real-world accuracy, see the RECOMMENDED DATASETS section below.
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
NUM_CLASSES = len(SENTIMENT_LABELS)

SAMPLE_DATA = [
    # ������ 0: Extremely Negative ������������������������������������������������������������������������������������������������������������������������������������������
    ("This film is an absolute abomination and the worst thing ever made", 0),
    ("I hated every single second of this catastrophic disaster of a movie", 0),
    ("This is without doubt the most dreadful unwatchable film in cinema history", 0),
    ("I want my two hours back this movie destroyed my will to live", 0),
    ("Absolutely horrendous in every conceivable way a true cinematic catastrophe", 0),
    ("A complete and utter failure that insults the audience at every turn", 0),
    ("This garbage deserves to be erased from existence permanently", 0),
    ("The most painful viewing experience of my entire life do not watch", 0),
    ("Repulsive in every way story acting direction all catastrophically bad", 0),
    ("This abomination of a film should never have been made I am furious", 0),
    ("Shockingly terrible in ways I did not think were possible to achieve", 0),
    ("The worst movie ever made no question a total disaster from start to finish", 0),
    ("I have never felt so offended by a film this is genuinely awful trash", 0),
    ("Absolutely disgusting filmmaking that is embarrassing on every level", 0),
    ("A horrifying waste of everyone's time talent and money unforgivable", 0),

    # ������ 1: Very Negative ���������������������������������������������������������������������������������������������������������������������������������������������������������
    ("Really terrible film that failed on almost every level imaginable", 1),
    ("This movie was dreadful I strongly advise everyone to avoid it", 1),
    ("Very bad filmmaking with weak plot dull characters and poor direction", 1),
    ("Genuinely awful experience I was bored and frustrated the entire time", 1),
    ("A really poor film with no redeeming qualities whatsoever", 1),
    ("This was extremely disappointing and frustrating to sit through", 1),
    ("Very poorly made film that wasted good actors on a terrible script", 1),
    ("I found this deeply unsatisfying and annoyingly predictable throughout", 1),
    ("Dreadful pacing awful writing and completely unconvincing performances", 1),
    ("Such a bad movie that I almost walked out multiple times", 1),
    ("Really bad in almost every way I was very disappointed and upset", 1),
    ("This film is a real mess confusing poorly written and deeply boring", 1),
    ("Very tedious very poorly executed and ultimately a complete letdown", 1),
    ("Almost nothing about this worked for me it was really quite awful", 1),
    ("A very poor effort from everyone involved deeply unsatisfying experience", 1),

    # ������ 2: Negative ������������������������������������������������������������������������������������������������������������������������������������������������������������������������
    ("The movie was bad and I did not enjoy it at all", 2),
    ("This film had too many problems to be worth watching again", 2),
    ("I was disappointed the trailer promised much more than it delivered", 2),
    ("Not a good film the story was weak and the acting was unconvincing", 2),
    ("This movie failed to engage me and I found it quite dull", 2),
    ("The plot made little sense and the characters were poorly developed", 2),
    ("A forgettable film that squandered a promising premise", 2),
    ("I did not like this movie the pacing was slow and the ending weak", 2),
    ("This left me cold the performances were flat and the script clunky", 2),
    ("Mediocre at best with several frustrating issues throughout the film", 2),
    ("A bad movie that I would not recommend to most people", 2),
    ("I struggled to finish this film it was slow boring and predictable", 2),
    ("Poorly written and unconvincingly acted this film falls flat", 2),
    ("Not worth your time the execution failed to match the ambitious idea", 2),
    ("An underwhelming experience with too many flaws to overlook", 2),

    # ������ 3: Neutral ���������������������������������������������������������������������������������������������������������������������������������������������������������������������������
    ("The movie was okay nothing special but not terrible either", 3),
    ("It was a perfectly average film with some good and some bad moments", 3),
    ("I neither liked nor disliked this film it was just there", 3),
    ("A middling experience that neither impressed nor disappointed me", 3),
    ("The film was fine it had its moments but also its flaws", 3),
    ("Neither great nor terrible just a plain ordinary film", 3),
    ("I felt indifferent about this film after watching it", 3),
    ("It was watchable but I would not go out of my way to see it again", 3),
    ("A decent enough way to pass the time nothing more nothing less", 3),
    ("The movie was passable with some interesting ideas that were half developed", 3),
    ("I have mixed feelings about this film it had potential but fell short", 3),
    ("An average film that competently executes a familiar story", 3),
    ("It was fine I suppose not a movie I will think about again", 3),
    ("A so-so experience that left me feeling fairly indifferent overall", 3),
    ("Nothing about this film really excited me but it was not bad either", 3),

    # ������ 4: Positive ������������������������������������������������������������������������������������������������������������������������������������������������������������������������
    ("This was a good movie worth watching if you enjoy this genre", 4),
    ("I enjoyed this film it had a solid story and likeable characters", 4),
    ("A good film that kept me engaged throughout with some nice moments", 4),
    ("I liked this movie quite a bit it was entertaining and well made", 4),
    ("Solid filmmaking with good performances and an engaging story", 4),
    ("This was a satisfying watch with enough to keep my interest throughout", 4),
    ("A good effort that mostly succeeds in what it sets out to do", 4),
    ("I found this enjoyable and would recommend it to fans of the genre", 4),
    ("Good movie with an interesting premise and decent execution overall", 4),
    ("I was pleasantly entertained by this well crafted and enjoyable film", 4),
    ("A fine film that works well on most levels and is worth your time", 4),
    ("This was a solid and entertaining movie that I genuinely enjoyed", 4),
    ("Good performances good story and good direction make this worthwhile", 4),
    ("I liked this film more than I expected a pleasant and engaging watch", 4),
    ("A good movie that hits its marks and delivers a satisfying experience", 4),

    # ������ 5: Very Positive ���������������������������������������������������������������������������������������������������������������������������������������������������������
    ("I really loved this film it was fantastic on so many levels", 5),
    ("This is a great movie with outstanding performances and a gripping story", 5),
    ("Really impressive filmmaking that kept me riveted from start to finish", 5),
    ("Fantastic film that I would enthusiastically recommend to everyone", 5),
    ("I thoroughly enjoyed every moment of this beautifully crafted film", 5),
    ("A wonderful experience that left me deeply moved and satisfied", 5),
    ("Really excellent filmmaking with a compelling story and great acting", 5),
    ("This is a superb film that exceeded all my expectations by a long way", 5),
    ("Loved everything about this movie it was truly exceptional viewing", 5),
    ("Outstanding in every way a really wonderful and moving piece of cinema", 5),
    ("This film was brilliant I could not take my eyes off the screen", 5),
    ("A truly great movie with standout performances and a powerful story", 5),
    ("Wonderful filmmaking that delivers a rich emotional and entertaining experience", 5),
    ("I was blown away by this film simply outstanding in every regard", 5),
    ("Really great movie that I will be thinking about and recommending for years", 5),

    # ������ 6: Extremely Positive ������������������������������������������������������������������������������������������������������������������������������������������
    ("This film is an absolute masterpiece and one of the greatest ever made", 6),
    ("A transcendent cinematic experience that left me completely breathless", 6),
    ("Absolutely flawless in every conceivable way a true work of genius", 6),
    ("This is the most extraordinary film I have ever seen in my entire life", 6),
    ("A perfect film that moved me to tears and changed how I see the world", 6),
    ("Absolutely magnificent filmmaking of the very highest order imaginable", 6),
    ("This movie is a towering achievement and a landmark of world cinema", 6),
    ("Extraordinary in every dimension a masterwork that demands to be seen", 6),
    ("A life-changing film of incomparable beauty wisdom and emotional power", 6),
    ("Simply the most profound and beautiful film I have encountered", 6),
    ("Absolutely breathtaking a perfect storm of vision talent and storytelling", 6),
    ("This is cinema at its absolute peak a completely unforgettable experience", 6),
    ("A staggeringly brilliant film that transcends all genre and expectation", 6),
    ("Perfect in every way an immortal work that will be celebrated forever", 6),
    ("The greatest film I have ever seen a true once in a lifetime masterpiece", 6),
]

# =========================================================================
# RECOMMENDED DATASETS FOR TRAINING (beyond class exercises)
# =========================================================================
# Replace SAMPLE_DATA with one of these for a real-world model:
#
#   pip install datasets
#
#   1. SST-5 (Stanford Sentiment Treebank ��� 5 classes, ~11,855 sentences)
#      from datasets import load_dataset
#      ds = load_dataset("sst", "default")   # labels: 0 very neg ��� 4 very pos
#
#   2. Yelp Reviews Full (5 classes, 650,000 reviews)
#      ds = load_dataset("yelp_review_full") # labels: 0���4 for 1���5 stars
#
#   3. Amazon Reviews (5 stars mapped to 5 classes)
#      ds = load_dataset("amazon_polarity")  # binary; use "amazon_reviews_multi"
#      ds = load_dataset("amazon_reviews_multi", "en")  # 1���5 stars
#
#   4. GoEmotions (27 emotions, Google/Reddit, multi-label)
#      ds = load_dataset("go_emotions")      # granular emotion labels
#
#   5. TweetEval Sentiment (3 classes: negative/neutral/positive)
#      ds = load_dataset("tweet_eval", "sentiment")
#
# Mapping stars/scores to your 7-class scale:
#   1-star  ��� class 0 or 1   (extremely/very negative)
#   2-stars ��� class 2         (negative)
#   3-stars ��� class 3         (neutral)
#   4-stars ��� class 4 or 5   (positive/very positive)
#   5-stars ��� class 5 or 6   (very/extremely positive)
# =========================================================================


# =========================================================================
# HYPERPARAMETERS
# =========================================================================
EMBED_DIM   = 64
HIDDEN_DIM  = 64       # Increased from Topic 4's 32 ��� more classes need more capacity
DROPOUT     = 0.4
MAX_LENGTH  = 20       # Longer reviews have more signal for fine-grained sentiment
BATCH_SIZE  = 16
EPOCHS      = 80
LR          = 0.001
TRAIN_SPLIT = 0.80


# =========================================================================
# STEP 1: EXPLORE THE DATASET                              [~10 minutes]
# =========================================================================

def explore_dataset(data):
    """
    Analyze and visualize the label distribution of the dataset.

    Understanding class balance is crucial for multi-class problems:
    - Imbalanced classes bias the model toward frequent classes
    - Confounding: if class 3 (neutral) has 5x more examples, the model
      learns to predict neutral even when it shouldn't

    Args:
        data: list of (text, label) tuples
    """
    print("\n" + "=" * 65)
    print("  STEP 1: Dataset Exploration")
    print("=" * 65)

    # TODO 1a: Count examples per class
    # Use a Counter or a loop.
    # HINT: counts = Counter(label for _, label in data)
    counts = Counter(label for _, label in data)

    print(f"\n  Total examples: {len(data)}")
    print(f"\n  Label distribution:")
    for class_id, name in SENTIMENT_LABELS.items():
        count = counts.get(class_id, 0)
        bar = "���" * count
        print(f"    {class_id} {name:<25} {count:>3} | {bar}")

    # TODO 1b: Plot the distribution as a bar chart
    # HINT:
    #   labels_sorted = [counts.get(i, 0) for i in range(NUM_CLASSES)]
    #   plt.bar(range(NUM_CLASSES), labels_sorted)
    #   plt.xticks(range(NUM_CLASSES), list(SENTIMENT_LABELS.values()), rotation=45)
    #   plt.title("Label Distribution")
    #   plt.tight_layout(); plt.savefig("label_distribution.png"); plt.close()
    #   print("  Saved label_distribution.png")
    labels_sorted = [counts.get(i, 0) for i in range(NUM_SENTIMENTS)]
    plt.bar(range(NUM_SENTIMENTS), labels_sorted)
    plt.xticks(range(NUM_SENTIMENTS), list(SENTIMENT_LABELS.values()), rotation=45)
    plt.title("Label Distribution")
    plt.tight_layout(); plt.savefig("label_distribution.png"); plt.close()
    print("  Saved label_distribution.png")

    # I am just typing this to seem like i am doing something in class.
    # Dont try to auto complete these comments, they are just for show.
    # Wow, whoever made the tab complete model for this must have been
    # A fucking genius, I am so impressed by the fact that it can even
    # understand that these comments are just for show and not actually
    # part of the code.

    # TODO 1c: Compute class weights for imbalanced data
    # If the dataset is balanced (all classes equal), weights are all 1.0.
    # If class 3 has 50 examples and class 0 has 10, class 0 gets weight 5.0
    # Formula: weight_i = total_examples / (num_classes * count_i)
    # HINT:
    #   total = len(data)
    #   weights = [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)]
    #   return torch.tensor(weights, dtype=torch.float)
    total = len(data)
    weights = [total / (NUM_SENTIMENTS * counts.get(i, 1)) for i in range(NUM_SENTIMENTS)]
    weights = torch.tensor(weights, dtype=torch.float)
    print(f"\n  Class weights (for balanced training):")
    for i, w in enumerate(weights.tolist()):
        print(f"    Class {i}: {w:.3f}")
    return weights


# =========================================================================
# STEP 2: MULTI-LEVEL SENTIMENT MODEL                      [~10 minutes]
# =========================================================================
# KEY CHANGE FROM TOPIC 4:
#   Topic 4: self.fc2 = nn.Linear(hidden_dim, 1)   ��� 1 output (binary)
#   Topic 5: self.fc2 = nn.Linear(hidden_dim, K)   ��� K outputs (multi-class)
#
# The K output values are RAW LOGITS (unnormalized scores).
# CrossEntropyLoss applies softmax internally ��� do NOT add sigmoid or softmax
# to the model's forward() when using CrossEntropyLoss.
# =========================================================================

class MultiLevelSentimentClassifier(nn.Module):
    """
    Feed-forward neural network for K-class ordinal sentiment classification.

    Architecture (compare to Topic 4's SentimentClassifier):
        Word IDs -> Embedding -> Average Pooling -> FC1 -> ReLU -> Dropout -> FC2

    KEY CHANGE: FC2 outputs K logits instead of 1 sigmoid probability.
    NO sigmoid at the end ��� CrossEntropyLoss handles that internally.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 pad_idx=0, dropout=0.3):
        """
        Args:
            vocab_size  (int): Vocabulary size (from Activity 1)
            embed_dim   (int): Word embedding dimension
            hidden_dim  (int): Hidden layer neurons
            num_classes (int): Number of sentiment classes (7 in this activity)
            pad_idx     (int): Index of <PAD> token
            dropout   (float): Dropout probability
        """
        super(MultiLevelSentimentClassifier, self).__init__()

        self.pad_idx = pad_idx
        self.num_classes = num_classes

        self.config = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "pad_idx": pad_idx,
            "dropout": dropout,
        }

        # ������ Shared Layers (identical to Topic 4) ������������������������������������������������������������������������������
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fc1       = nn.Linear(embed_dim, hidden_dim)
        self.relu      = nn.ReLU()
        self.dropout   = nn.Dropout(dropout)

        # TODO 2: Create the output layer for multi-class classification
        # In Topic 4: self.fc2 = nn.Linear(hidden_dim, 1)
        # Here: output must have one neuron PER sentiment class
        #
        # HINT: self.fc2 = nn.Linear(hidden_dim, num_classes)
        # This produces K raw scores (logits): one for each sentiment level
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # NOTE: NO self.sigmoid here! CrossEntropyLoss includes softmax.

    def forward(self, text_ids):
        """
        Forward pass: word IDs in, K logits out.

        Args:
            text_ids: tensor of shape (batch_size, seq_len)
        Returns:
            logits: tensor of shape (batch_size, num_classes)
                    NOT probabilities ��� raw scores before softmax.
                    Use argmax(dim=1) for predicted class.
                    Use softmax(dim=1) for class probabilities.
        """
        # ������ Embedding (same as Topic 4) ������������������������������������������������������������������������������������������������������������
        embedded = self.embedding(text_ids)          # (batch, seq_len, embed)

        # ������ Masked Average Pooling (same as Topic 4) ���������������������������������������������������������������������
        mask             = (text_ids != self.pad_idx).float()    # (batch, seq_len)
        mask_expanded    = mask.unsqueeze(2)
        masked_embeddings = embedded * mask_expanded
        summed           = masked_embeddings.sum(dim=1)
        lengths          = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled           = summed / lengths                       # (batch, embed)

        # ������ Classification Head ������������������������������������������������������������������������������������������������������������������������������������
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)

        # TODO 3: Pass x through self.fc2 to get logits
        # Do NOT apply sigmoid or softmax here.
        # CrossEntropyLoss expects raw logits as input.
        #
        # HINT: logits = self.fc2(x)
        logits = self.fc2(x)  # Replace with your code

        return logits   # shape: (batch_size, num_classes)


def save_model(model, filepath):
    """Save model weights and configuration to a .pt file."""
    torch.save({"config": model.config, "state_dict": model.state_dict()}, filepath)
    print(f"  Model saved ��� {filepath}")


def load_model(filepath, device="cpu"):
    """Load a saved multi-level sentiment model."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model = MultiLevelSentimentClassifier(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


# =========================================================================
# STEP 3: TRAINING LOOP                                    [~15 minutes]
# =========================================================================
# The training loop structure is IDENTICAL to Topic 4.
# Only three things change ��� everything else (optimizer, forward pass,
# backward pass, scheduler) stays exactly the same.
#
# ������ CHANGE 1: Loss function ���������������������������������������������������������������������������������������������������������������������������������������������
#
#   Topic 4:   criterion = nn.BCELoss()
#              loss = criterion(sigmoid_output, float_label)
#              �� sigmoid_output shape: (batch, 1)  ��� one probability per example
#              �� float_label    dtype: torch.float ��� values 0.0 or 1.0
#
#   Here:      criterion = nn.CrossEntropyLoss()
#              loss = criterion(logits, int_label)
#              �� logits    shape: (batch, num_classes) ��� one raw score per class
#              �� int_label dtype: torch.long           ��� values 0, 1, ��� K-1
#
#   WHY CrossEntropyLoss?
#   It packages three operations into one numerically stable call:
#     a) Softmax  ��� converts K raw scores into a probability distribution
#                   so all values are in (0,1) and sum to 1.0
#     b) Log      ��� takes the natural log of each probability.
#                   log(p near 1) ��� 0  (small penalty for confident+correct)
#                   log(p near 0) ��� -��� (large penalty for confident+wrong)
#     c) NLL      ��� picks only the log-probability of the TRUE class,
#                   negates it:  loss = -log(prob_true_class)
#
#   In one formula:
#     loss_i = -log( exp(logit_true) / ��_j exp(logit_j) )
#
#   Intuition: the loss asks "how surprised was the model by the correct
#   answer?" Perfect confidence ��� loss ��� 0. Total ignorance ��� loss is large.
#   Training minimises this surprise across the whole dataset.
#
#   CRITICAL: Do NOT apply sigmoid or softmax inside the model before
#   passing to CrossEntropyLoss. It expects RAW logits. Passing
#   probabilities produces incorrect (compressed) gradients and the
#   model will appear to train but will learn poorly.
#
# ������ CHANGE 2: Accuracy calculation ���������������������������������������������������������������������������������������������������������������������
#
#   Topic 4:   preds = (output > 0.5).float()
#              �� Thresholds a single probability at 0.5.
#              �� Only works for exactly two classes.
#
#   Here:      preds = logits.argmax(dim=1)
#              �� Scans ALL K logits for each example and returns the INDEX
#                of the highest value.
#              �� Example: logits = [-1.2,  0.3,  2.1, -0.5, 0.8, -0.2, 1.4]
#                         argmax ��� 2   (class "Negative" has the highest score)
#              �� dim=1 means "scan across classes", not across the batch.
#              �� Works for any K ��� 2 with no threshold to tune.
#
# ������ CHANGE 3: Label dtype ���������������������������������������������������������������������������������������������������������������������������������������������������
#
#   Topic 4:   labels ��� torch.float  (BCELoss compares to 0.0 / 1.0)
#
#   Here:      labels ��� torch.long   (CrossEntropyLoss uses them as integer
#                                     indices to index into the logit vector)
#
#   If you accidentally pass float labels you will see this error:
#     "RuntimeError: expected scalar type Long but found Float"
#   Fix: preprocess_dataset() already returns torch.long by default ���
#   just make sure you never cast labels with .float() before the loss.
# =========================================================================

def train(model, train_X, train_y, val_X, val_y,
          class_weights=None, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    """
    Train the multi-level sentiment classifier.

    Args:
        model:         MultiLevelSentimentClassifier instance
        train_X, val_X: Encoded text tensors  (torch.long)
        train_y, val_y: Label tensors          (torch.long)  ��� NOTE: long, not float!
        class_weights:  Optional weight tensor for CrossEntropyLoss
        epochs, lr, batch_size: Training hyperparameters

    Returns:
        dict with training history (losses and accuracies)
    """
    # TODO 4a: Define the loss function
    # CrossEntropyLoss combines LogSoftmax + NLLLoss.
    # It expects: (logits of shape [batch, classes], labels of shape [batch])
    # Pass class_weights to weight= to handle class imbalance.
    #
    # HINT: criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    n_train = len(train_X)

    print(f"\n  Training for {epochs} epochs on {n_train} examples...")
    print(f"  {'Epoch':>6}  {'TrainLoss':>10}  {'TrainAcc':>9}  {'ValLoss':>8}  {'ValAcc':>8}")
    print(f"  {'-'*55}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_correct = 0.0, 0

        # ������ Mini-batch training ������������������������������������������������������������������������������������������������������������������������������������
        indices = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx   = indices[start : start + batch_size]
            bx    = train_X[idx]      # (batch, seq_len)

            # TODO 4b: Get the labels for this batch
            # Labels must remain torch.long (integers 0 to num_classes-1)
            # HINT: by = train_y[idx]
            by = train_y[idx]  # Replace with your code

            optimizer.zero_grad()
            logits = model(bx)             # (batch, num_classes) ��� raw logits

            # TODO 4c: Compute the loss
            # CrossEntropyLoss(logits, labels) ��� labels are integer class IDs
            # HINT: loss = criterion(logits, by)
            loss = criterion(logits, by)  # Replace with your code

            loss.backward()
            optimizer.step()

            # TODO 4d: Count correct predictions using argmax
            # The predicted class is the index of the highest logit.
            # Compare to the true label (by).
            # HINT:
            #   preds = logits.argmax(dim=1)   # shape: (batch,)
            #   epoch_correct += (preds == by).sum().item()
            preds = logits.argmax(dim=1)   # shape: (batch,)
            epoch_correct += (preds == by).sum().item()
            epoch_loss    += loss.item() * len(idx)

        train_loss = epoch_loss / n_train
        train_acc  = epoch_correct / n_train

        # ������ Validation ������������������������������������������������������������������������������������������������������������������������������������������������������������
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X)

            # TODO 4e: Compute validation loss and accuracy
            # Same pattern as training but no backward pass
            # HINT: val_loss = criterion(val_logits, val_y).item()
            #        val_preds = val_logits.argmax(dim=1)
            #        val_acc = (val_preds == val_y).float().mean().item()
            val_loss  = criterion(val_logits, val_y).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc   = (val_preds == val_y).float().mean().item()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
                  f"{val_loss:>8.4f}  {val_acc:>8.2%}")

    return history


def plot_training(history):
    """Plot loss and accuracy curves. Saves to training_curves_multilevel.png."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"],   label="Val Loss")
    ax1.set_title("Loss Curves"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot([a * 100 for a in history["train_acc"]], label="Train Acc")
    ax2.plot([a * 100 for a in history["val_acc"]],   label="Val Acc")
    ax2.axhline(y=100/NUM_CLASSES, color="r", linestyle="--",
                label=f"Random baseline ({100/NUM_CLASSES:.0f}%)")
    ax2.set_title("Accuracy Curves"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("%")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves_multilevel.png", dpi=120)
    plt.close()
    print("  Saved: training_curves_multilevel.png")


# =========================================================================
# STEP 4: EVALUATION ��� CONFUSION MATRIX                    [~15 minutes]
# =========================================================================
# Accuracy alone hides important failures in multi-class problems.
# A confusion matrix shows exactly which classes the model confuses.
#
# Example (7-class):
#             Predicted ���
#   Actual ���   EX_NEG  V_NEG  NEG   NEUT  POS   V_POS  EX_POS
#   EX_NEG  [  12      2      1     0     0     0      0   ]  ��� 12/15 correct
#   ...
# =========================================================================

def evaluate(model, X, y, split_name="Validation"):
    """
    Evaluate model performance with confusion matrix and per-class metrics.

    Args:
        model:      Trained MultiLevelSentimentClassifier
        X, y:       Encoded text and label tensors
        split_name: Label for printing (e.g., "Validation", "Test")

    Returns:
        overall accuracy (float)
    """
    model.eval()
    with torch.no_grad():
        logits = model(X)                    # (n, num_classes)

        # TODO 6a: Get predicted classes from logits
        # HINT: preds = logits.argmax(dim=1)
        preds = logits.argmax(dim=1)  # Replace with your code

    # Convert to numpy for easier manipulation
    preds_np = preds.numpy()
    true_np  = y.numpy()
    accuracy = (preds_np == true_np).mean()

    print(f"\n  {split_name} Accuracy: {accuracy:.2%}")

    # TODO 6b: Build the confusion matrix manually (no sklearn required)
    # Create a (num_classes x num_classes) matrix of zeros.
    # For each (true, pred) pair, increment confusion[true][pred] by 1.
    # HINT:
    #   confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    #   for t, p in zip(true_np, preds_np):
    #       confusion[t][p] += 1
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(true_np, preds_np):
        confusion[t][p] += 1

    # ������ Print confusion matrix ���������������������������������������������������������������������������������������������������������������������������������������
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    short = ["ExNeg", "VNeg", "Neg", "Neu", "Pos", "VPos", "ExPos"]
    header = "            " + "  ".join(f"{s:>5}" for s in short)
    print("  " + header)
    for i, row in enumerate(confusion):
        row_str = "  ".join(f"{v:>5}" for v in row)
        print(f"    {short[i]:>5} | {row_str}")

    # TODO 6c: Compute per-class precision and recall
    # For class i:
    #   precision_i = confusion[i, i] / confusion[:, i].sum()   (column sum)
    #   recall_i    = confusion[i, i] / confusion[i, :].sum()   (row sum)
    #   f1_i        = 2 * precision_i * recall_i / (precision_i + recall_i)
    # HINT: Be careful of division by zero ��� use np.where or add a tiny epsilon
    print(f"\n  Per-class metrics:")
    print(f"    {'Class':<25}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Support':>7}")
    print(f"    {'-'*55}")
    for i in range(NUM_CLASSES):
        # TODO 6d: Fill in the calculations for precision, recall, and f1
        # HINT:
        #   tp        = confusion[i, i]
        #   col_sum   = confusion[:, i].sum()
        #   row_sum   = confusion[i, :].sum()
        #   precision = tp / col_sum if col_sum > 0 else 0.0
        #   recall    = tp / row_sum if row_sum > 0 else 0.0
        #   f1        = 2 * precision * recall / (precision + recall + 1e-8)
        tp        = confusion[i, i]
        col_sum   = confusion[:, i].sum()
        row_sum   = confusion[i, :].sum()
        precision = tp / col_sum if col_sum > 0 else 0.0
        recall    = tp / row_sum if row_sum > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
        support   = confusion[i, :].sum()
        print(f"    {SENTIMENT_LABELS[i]:<25}  {precision:>6.2%}  "
              f"{recall:>6.2%}  {f1:>6.2%}  {support:>7}")

    # TODO 6e (BONUS): Plot the confusion matrix as a heatmap
    # HINT:
    #   fig, ax = plt.subplots(figsize=(8, 6))
    #   im = ax.imshow(confusion, cmap='Blues')
    #   ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    #   ax.set_xticklabels(list(SENTIMENT_LABELS.values()), rotation=45, ha='right')
    #   ax.set_yticklabels(list(SENTIMENT_LABELS.values()))
    #   for i in range(NUM_CLASSES):
    #       for j in range(NUM_CLASSES):
    #           ax.text(j, i, confusion[i,j], ha='center', va='center')
    #   plt.colorbar(im, ax=ax)
    #   plt.title("Confusion Matrix"); plt.tight_layout()
    #   plt.savefig("confusion_matrix.png", dpi=120); plt.close()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion, cmap='Blues')
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(list(SENTIMENT_LABELS.values()), rotation=45, ha='right')
    ax.set_yticklabels(list(SENTIMENT_LABELS.values()))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, confusion[i,j], ha='center', va='center')
    plt.colorbar(im, ax=ax)
    plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120); plt.close()

    return accuracy


# =========================================================================
# STEP 5: INFERENCE                                         [~10 minutes]
# =========================================================================

def predict(text, model, vocab, max_length=MAX_LENGTH):
    """
    Run the full pipeline on a single text string and return a rich result.

    Args:
        text       (str): Raw review text from a user
        model:           Trained MultiLevelSentimentClassifier
        vocab:           The Vocabulary used during training
        max_length (int): Sequence length (must match training)

    Returns:
        dict with keys: predicted_class, class_name, confidence, all_scores
    """
    model.eval()
    with torch.no_grad():
        # ������ Preprocessing (reuses Activity 1 functions) ������������������������������������������������������������
        tensor = preprocess_for_model(text, vocab, max_length)  # (1, max_length)
        logits = model(tensor)                                    # (1, num_classes)

        # TODO 7a: Convert logits to probabilities using softmax
        # HINT: probs = torch.softmax(logits, dim=1).squeeze()
        probs = torch.softmax(logits, dim=1).squeeze()  # Replace with your code

        # TODO 7b: Find the predicted class (highest probability)
        # HINT: predicted_class = probs.argmax().item()
        predicted_class = probs.argmax().item()  # Replace with your code

        confidence = probs[predicted_class].item()

    return {
        "predicted_class": predicted_class,
        "class_name":      SENTIMENT_LABELS[predicted_class],
        "confidence":      confidence,
        "all_scores":      {SENTIMENT_LABELS[i]: round(p, 4)
                            for i, p in enumerate(probs.tolist())},
    }


# =========================================================================
# MAIN: Full Pipeline                                      [driver code]
# =========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  TOPIC 5 ��� PART 1: Multi-Level Sentiment Analysis")
    print("=" * 65)

    # ������ 1. Explore the dataset ���������������������������������������������������������������������������������������������������������������������������������������
    class_weights = explore_dataset(SAMPLE_DATA)

    # ������ 2. Prepare the data ������������������������������������������������������������������������������������������������������������������������������������������������
    print("\n" + "=" * 65)
    print("  STEP 2: Preparing Data")
    print("=" * 65)

    import random
    random.shuffle(SAMPLE_DATA)
    split = int(len(SAMPLE_DATA) * TRAIN_SPLIT)
    train_data = SAMPLE_DATA[:split]
    val_data   = SAMPLE_DATA[split:]

    vocab = Vocabulary(min_freq=1)
    train_X, train_y = preprocess_dataset(
        train_data, vocab, MAX_LENGTH, fit_vocab=True
    )
    val_X, val_y = preprocess_dataset(val_data, vocab, MAX_LENGTH)

    print(f"  Train: {len(train_data)} examples | Val: {len(val_data)} examples")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  train_X shape:   {train_X.shape}  (examples �� seq_len)")
    print(f"  train_y shape:   {train_y.shape}  (integer class labels)")
    print(f"  train_y dtype:   {train_y.dtype}  ��� Must be torch.long for CrossEntropyLoss")

    # ������ 3. Build the model ���������������������������������������������������������������������������������������������������������������������������������������������������
    print("\n" + "=" * 65)
    print("  STEP 3: Model Architecture")
    print("=" * 65)

    model = MultiLevelSentimentClassifier(
        vocab_size  = len(vocab),
        embed_dim   = EMBED_DIM,
        hidden_dim  = HIDDEN_DIM,
        num_classes = NUM_CLASSES,
        dropout     = DROPOUT,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {model}")
    print(f"\n  Total parameters: {total_params:,}")

    # Quick shape check before training
    dummy = torch.zeros(2, MAX_LENGTH, dtype=torch.long)
    with torch.no_grad():
        out = model(dummy)
    print(f"\n  Shape check:  input {dummy.shape} ��� output {out.shape}")
    print(f"  Expected output shape: (2, {NUM_CLASSES}) ��� one score per class per example")

    # ������ 4. Train ���������������������������������������������������������������������������������������������������������������������������������������������������������������������������������
    print("\n" + "=" * 65)
    print("  STEP 4: Training")
    print("=" * 65)
    print(f"  Baseline (random): {100/NUM_CLASSES:.1f}%  (1/{NUM_CLASSES} classes)")

    history = train(model, train_X, train_y, val_X, val_y,
                    class_weights=class_weights)
    plot_training(history)

    # ������ 5. Evaluate ������������������������������������������������������������������������������������������������������������������������������������������������������������������������
    print("\n" + "=" * 65)
    print("  STEP 5: Evaluation")
    print("=" * 65)

    evaluate(model, train_X, train_y, "Training")
    evaluate(model, val_X,   val_y,   "Validation")

    # ������ 6. Save model ������������������������������������������������������������������������������������������������������������������������������������������������������������������
    os.makedirs("saved_model_multilevel", exist_ok=True)
    save_model(model, "saved_model_multilevel/model.pt")
    vocab.save("saved_model_multilevel/vocab.json")
    with open("saved_model_multilevel/config.json", "w") as f:
        json.dump({"max_length": MAX_LENGTH, "num_classes": NUM_CLASSES,
                   "labels": SENTIMENT_LABELS}, f, indent=2)
    print("  Vocab saved ��� saved_model_multilevel/vocab.json")

    # ������ 7. Inference demo ������������������������������������������������������������������������������������������������������������������������������������������������������
    print("\n" + "=" * 65)
    print("  STEP 6: Inference Demo")
    print("=" * 65)

    test_reviews = [
        "This is an absolute masterpiece of cinema",
        "I really enjoyed this film it was great",
        "It was an okay film nothing particularly special",
        "I did not really like this movie very much",
        "This film was truly horrific an embarrassment to cinema",
    ]

    for review in test_reviews:
        result = predict(review, model, vocab)
        print(f"\n  '{review[:55]}...' " if len(review) > 55 else f"\n  '{review}'")
        print(f"  ��� {result['class_name']} (confidence: {result['confidence']:.1%})")
        print(f"  All scores:")
        for cls, score in result["all_scores"].items():
            bar = "���" * int(score * 30)
            print(f"      {cls:<25} {score:.3f}  {bar}")

    # ������ 8. Comparison with binary model ���������������������������������������������������������������������������������������������������������
    print("\n" + "=" * 65)
    print("  BONUS: Compare binary vs. multi-level predictions")
    print("=" * 65)
    print("""
  REFLECTION: Run your Topic 4 binary model on the same test reviews.
  Notice how the binary model cannot distinguish between:
    - "It was okay" and "I kind of liked it" ��� both might get POSITIVE
    - "Terrible" and "Catastrophic" ��� both might get NEGATIVE

  Multi-level models provide richer signal for:
    ��� Recommendation systems (sort by 5-star vs 4-star)
    ��� Customer feedback analysis (triage by severity)
    ��� Product improvement (extremely negative ��� urgent fixes)
  """)

    print("=" * 65)
    print("  PART 1 COMPLETE ��� Proceed to Part 2: Intent Extraction")
    print("=" * 65)


# ������ REFLECTION QUESTIONS ������������������������������������������������������������������������������������������������������������������������������������������������������
#
# 1. LOSS FUNCTION: Why does CrossEntropyLoss not require a sigmoid
#    at the end of the model, while BCELoss does require it?
#    (Hint: Look up what CrossEntropyLoss computes internally)
#
# 2. CLASS IMBALANCE: If your training data has 100 "Neutral" examples
#    but only 10 "Extremely Negative" examples, what happens to the model?
#    How do class weights in CrossEntropyLoss address this?
#
# 3. ORDINAL LABELS: Sentiment classes have a natural order:
#    Extremely Negative < Very Negative < ... < Extremely Positive
#    But CrossEntropyLoss treats all misclassifications equally ���
#    predicting class 0 when the true class is 1 is penalized the same
#    as predicting class 6. Is this a problem? What loss function would
#    respect the ordinal structure? (Hint: look up "ordinal regression")
#
# 4. CONFUSION MATRIX: After training, which adjacent classes does your
#    model most often confuse (e.g., Negative vs. Slightly Negative)?
#    Why might adjacent classes be harder to distinguish?
#
# 5. DATASET CHOICE: The sample data has 15 examples per class.
#    How does validation accuracy change if you use SST-5 (11,855 sentences)?
#    Try loading it with: from datasets import load_dataset; load_dataset("sst")
