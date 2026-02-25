"""
============================================================================
AIT-204 Deep Learning | Topic 4: Natural Language Processing
ACTIVITY 3: Training Pipeline — Train, Evaluate, Save
============================================================================

PURPOSE:
    Train the sentiment classifier on movie review data, evaluate it,
    plot training curves, and save the trained model + vocabulary to disk.
    These saved artifacts are loaded by Activity 4's web app.

DURATION: ~90 minutes in-class

BRIDGING THE GAP (Concept -> Algorithm -> Code -> Deployed App):
    ---------------------------------------------------------------
    The training loop here is STRUCTURALLY IDENTICAL to Topics 1-3:

        for each epoch:
            for each batch:
                1. Forward pass:    predictions = model(inputs)
                2. Compute loss:    loss = criterion(predictions, targets)
                3. Backward pass:   loss.backward()
                4. Update weights:  optimizer.step()
                5. Zero gradients:  optimizer.zero_grad()

    The ONLY difference from Topics 1-2: the input is text (preprocessed
    by Activity 1) instead of raw numbers, and the model has an embedding
    layer (Activity 2) instead of taking numerical features directly.

    FOR YOUR APP:
        This script produces two files:
            saved_model/model.pt   — trained neural network weights
            saved_model/vocab.json — vocabulary mapping words to IDs
        Activity 4's web app loads these to make predictions in real-time.

WHAT YOU'LL IMPLEMENT:
    - The training step (forward, loss, backward, step)
    - The validation evaluation loop
    - Hyperparameter choices (learning rate, epochs, etc.)

RUN THIS FILE: python activity3_train.py
    Produces: saved_model/model.pt, saved_model/vocab.json, plots/
============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import YOUR modules from Activities 1 and 2
from activity1_preprocessing import Vocabulary, preprocess_dataset
from activity2_model import SentimentClassifier, save_model

torch.manual_seed(42)
np.random.seed(42)


# =========================================================================
# TRAINING DATA
# =========================================================================
# 120 movie reviews (60 positive, 60 negative).
# In the full assignment, replace this with the IMDB dataset (~25,000).
# =========================================================================

REVIEWS = [
    # ===== POSITIVE (label=1) =====
    ("This movie was absolutely wonderful and I loved every minute", 1),
    ("A brilliant masterpiece with outstanding performances", 1),
    ("The story was engaging and deeply compelling throughout", 1),
    ("Thoroughly entertained from start to finish great film", 1),
    ("Beautiful cinematography and a heartwarming emotional story", 1),
    ("An excellent adventure that keeps you on the edge", 1),
    ("The acting was superb and the plot cleverly written", 1),
    ("One of the best films I have seen highly recommended", 1),
    ("A delightful comedy that had everyone laughing", 1),
    ("Incredible effects paired with a solid compelling storyline", 1),
    ("The director did an amazing job with this story", 1),
    ("A perfect blend of humor and drama on every level", 1),
    ("This film exceeded all my expectations great experience", 1),
    ("Loved the soundtrack and the emotional depth of characters", 1),
    ("A refreshing original film that stands out beautifully", 1),
    ("Simply outstanding this is cinema at its finest", 1),
    ("Every scene was crafted with care and attention", 1),
    ("The performances were moving and deeply heartfelt", 1),
    ("An uplifting story that restores faith in filmmaking", 1),
    ("Captivated from the opening scene to the end credits", 1),
    ("Masterful direction and writing make this a must see", 1),
    ("The chemistry between the actors was wonderful to watch", 1),
    ("A triumphant return to form for this talented director", 1),
    ("The script was sharp witty and emotionally resonant", 1),
    ("This is the kind of film that stays with you", 1),
    ("Absolutely loved it one of the years best offerings", 1),
    ("The visuals were stunning and the story was gripping", 1),
    ("A beautiful film with a powerful and moving message", 1),
    ("I laughed I cried and left the theater feeling inspired", 1),
    ("Exceptional storytelling with unforgettable characters", 1),
    ("This movie is a joy from beginning to end", 1),
    ("The cast delivered stellar performances across the board", 1),
    ("A genuinely touching film that earns every emotion", 1),
    ("Smart funny and surprisingly deep a real treat", 1),
    ("The best movie experience I have had in a long time", 1),
    ("Perfectly paced with great dialogue and wonderful acting", 1),
    ("This film will make you laugh cry and think", 1),
    ("An extraordinary achievement in modern cinema", 1),
    ("Blown away by the quality of this production", 1),
    ("The story had real heart and genuine emotional impact", 1),
    ("A wonderful adventure filled with warmth and excitement", 1),
    ("This deserves all the praise it has received", 1),
    ("Both entertaining and intellectually stimulating throughout", 1),
    ("A captivating tale told with skill and passion", 1),
    ("Highly recommend this film to everyone who loves movies", 1),
    ("The ending was perfect and deeply satisfying", 1),
    ("A remarkable film that sets a new standard for quality", 1),
    ("Every moment felt authentic and emotionally true", 1),
    ("This movie proves that great stories never get old", 1),
    ("Flawless direction and incredible performances throughout", 1),
    ("A magical experience that transported me completely", 1),
    ("Funny heartwarming and brilliantly executed from start", 1),
    ("The storytelling was masterful and deeply engaging", 1),
    ("Loved every single moment of this incredible film", 1),
    ("A fantastic movie that exceeded every expectation possible", 1),
    ("The best film of the year without any question", 1),
    ("Outstanding performances and a truly compelling narrative", 1),
    ("This movie is pure cinematic gold in every way", 1),
    ("A breathtaking film with amazing and powerful performances", 1),
    ("Moved to tears by this beautiful and touching story", 1),

    # ===== NEGATIVE (label=0) =====
    ("This movie was terrible and a complete waste of time", 0),
    ("The acting was awful and the plot made no sense", 0),
    ("So bored I almost fell asleep during this film", 0),
    ("A dull lifeless movie with no redeeming qualities", 0),
    ("The worst movie I have seen this year avoid it", 0),
    ("Terrible dialogue and unconvincing performances throughout", 0),
    ("The story was predictable and the characters flat", 0),
    ("A disappointing sequel that fails on every level", 0),
    ("Poorly directed with awkward pacing and confusion", 0),
    ("Regret spending money on this truly awful movie", 0),
    ("The special effects were cheap and the script laughable", 0),
    ("An absolute disaster from beginning to end entirely", 0),
    ("This film is a mess with horrible writing throughout", 0),
    ("Boring predictable and uninspired nothing to offer", 0),
    ("A painful experience I would not wish on anyone", 0),
    ("Completely forgettable and a waste of talent entirely", 0),
    ("The pacing was terrible and the story went nowhere", 0),
    ("Not a single performance in this film was convincing", 0),
    ("A lazy effort that insults the intelligence of viewers", 0),
    ("Never been so disappointed by a movie in my life", 0),
    ("The script was full of cliches and bad dialogue", 0),
    ("A hollow empty film with no substance or meaning", 0),
    ("The worst acting I have seen in a major production", 0),
    ("This movie failed on every conceivable level possible", 0),
    ("A tedious bore that dragged on far too long", 0),
    ("The direction was confused and the editing terrible", 0),
    ("Wanted to leave the theater halfway through this", 0),
    ("An embarrassment for everyone involved in making it", 0),
    ("The plot holes were enormous and impossible to ignore", 0),
    ("A joyless experience from start to finish entirely", 0),
    ("This movie has no heart no soul and no point", 0),
    ("The characters were unlikable and poorly developed", 0),
    ("A cheap knockoff that brings nothing new at all", 0),
    ("Wasted potential with terrible execution throughout it", 0),
    ("Cannot believe this movie received any positive reviews", 0),
    ("The dialogue was cringe worthy and completely unnatural", 0),
    ("A total misfire that misses the mark on everything", 0),
    ("The film was so bad it was almost unintentionally funny", 0),
    ("Nothing about this movie worked for me at all", 0),
    ("A frustrating mess of a film that goes nowhere fast", 0),
    ("Easily one of the worst films ever made period", 0),
    ("The story made absolutely no sense at all whatsoever", 0),
    ("A mindless bore with no creativity or originality here", 0),
    ("Felt nothing watching this empty shell of a movie", 0),
    ("The movie was a complete and total disappointment", 0),
    ("Terrible terrible terrible in every possible way imaginable", 0),
    ("A forgettable disaster that should never have been made", 0),
    ("The plot was nonsensical and the acting was wooden", 0),
    ("Would give this zero stars if I possibly could", 0),
    ("An insult to audiences everywhere avoid this film completely", 0),
    ("The worst two hours I have ever spent anywhere", 0),
    ("Lazy writing bad acting and awful direction throughout", 0),
    ("This film has absolutely nothing going for it at all", 0),
    ("A trainwreck of epic proportions from start to finish", 0),
    ("Hated every minute of this dreadful and awful movie", 0),
    ("The movie was painfully slow and incredibly dull throughout", 0),
    ("A catastrophic failure in every single department possible", 0),
    ("This movie is an offense to good taste everywhere", 0),
    ("Unwatchable garbage from start to finish entirely horrible", 0),
    ("The absolute worst movie of the entire decade easily", 0),
]


# =========================================================================
# STEP 1: PREPARE DATA
# =========================================================================

# Shuffle and split: 80% train, 20% validation
indices = list(range(len(REVIEWS)))
np.random.shuffle(indices)
shuffled = [REVIEWS[i] for i in indices]

split = int(0.8 * len(shuffled))
train_reviews = shuffled[:split]
val_reviews = shuffled[split:]

# Hyperparameters
MAX_LENGTH = 15
EMBED_DIM = 64
HIDDEN_DIM = 32
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 60
DROPOUT = 0.3

# Preprocess: build vocab on training data ONLY
vocab = Vocabulary(min_freq=1)
train_X, train_y = preprocess_dataset(train_reviews, vocab, MAX_LENGTH, fit_vocab=True)
val_X, val_y = preprocess_dataset(val_reviews, vocab, MAX_LENGTH, fit_vocab=False)

print(f"Training:   {len(train_reviews)} reviews -> tensor {train_X.shape}")
print(f"Validation: {len(val_reviews)} reviews -> tensor {val_X.shape}")
print(f"Vocabulary: {len(vocab)} words")

# DataLoaders for batching
train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=BATCH_SIZE)


# =========================================================================
# STEP 2: CREATE MODEL, LOSS, OPTIMIZER
# =========================================================================

model = SentimentClassifier(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    pad_idx=vocab.pad_idx,
    dropout=DROPOUT,
)

print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

# TODO 1: Create the loss function
# For binary classification (positive/negative), use Binary Cross-Entropy.
# HINT: criterion = nn.BCELoss()
criterion = nn.BCELoss()

# TODO 2: Create the optimizer
# Adam is an improved version of gradient descent from Topic 1.
# HINT: optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# =========================================================================
# STEP 3: TRAINING LOOP
# =========================================================================
# This is the SAME loop from Topics 1-3. The only difference is that
# the input is text (processed by Activity 1) rather than raw numbers.
# =========================================================================

train_losses, val_losses = [], []
train_accs, val_accs = [], []

print(f"\nTraining for {NUM_EPOCHS} epochs...")
print(f"{'Epoch':>6} | {'Train Loss':>11} {'Train Acc':>10} | {'Val Loss':>11} {'Val Acc':>10}")
print("-" * 62)

for epoch in range(NUM_EPOCHS):

    # ---- TRAINING PHASE ----
    model.train()  # Enable dropout
    epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

    for batch_X, batch_y in train_loader:

        # TODO 3: Implement the 5-step training process
        # This is IDENTICAL to Topics 1-3.

        # Step A: Zero the gradients from the previous iteration
        # HINT: optimizer.zero_grad()
        optimizer.zero_grad()

        # Step B: Forward pass — get predictions
        # HINT: predictions = model(batch_X).squeeze(1)
        # (.squeeze(1) removes extra dim: (batch, 1) -> (batch,))
        predictions = model(batch_X).squeeze(1)

        # Step C: Compute loss
        # HINT: loss = criterion(predictions, batch_y.float())
        # Note: BCELoss needs float labels, not int
        loss = criterion(predictions, batch_y.float())

        # Step D: Backward pass — compute gradients
        # HINT: loss.backward()
        loss.backward()

        # Step E: Update weights using gradients
        # HINT: optimizer.step()
        optimizer.step()

        # Track metrics (provided)
        epoch_loss += loss.item() * len(batch_y)
        epoch_correct += ((predictions >= 0.5).float() == batch_y.float()).sum().item()
        epoch_total += len(batch_y)

    avg_train_loss = epoch_loss / epoch_total
    avg_train_acc = epoch_correct / epoch_total
    train_losses.append(avg_train_loss)
    train_accs.append(avg_train_acc)

    # ---- VALIDATION PHASE (provided — no gradient updates!) ----
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            preds = model(batch_X).squeeze(1)
            loss = criterion(preds, batch_y.float())
            v_loss += loss.item() * len(batch_y)
            v_correct += ((preds >= 0.5).float() == batch_y.float()).sum().item()
            v_total += len(batch_y)

    avg_val_loss = v_loss / v_total
    avg_val_acc = v_correct / v_total
    val_losses.append(avg_val_loss)
    val_accs.append(avg_val_acc)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:>6} | {avg_train_loss:>11.4f} {avg_train_acc:>9.1%} | "
              f"{avg_val_loss:>11.4f} {avg_val_acc:>9.1%}")

print("-" * 62)
print(f"{'FINAL':>6} | {train_losses[-1]:>11.4f} {train_accs[-1]:>9.1%} | "
      f"{val_losses[-1]:>11.4f} {val_accs[-1]:>9.1%}")


# =========================================================================
# STEP 4: SAVE MODEL AND VOCABULARY
# =========================================================================
# These files are what Activity 4's web app loads.

os.makedirs("saved_model", exist_ok=True)
save_model(model, "saved_model/model.pt")
vocab.save("saved_model/vocab.json")

# Also save max_length so the app knows the expected input size
import json
with open("saved_model/config.json", "w") as f:
    json.dump({"max_length": MAX_LENGTH}, f)

print(f"\nSaved to saved_model/:")
print(f"  model.pt   — trained neural network ({sum(p.numel() for p in model.parameters()):,} params)")
print(f"  vocab.json — vocabulary ({len(vocab)} words)")
print(f"  config.json — inference config (max_length={MAX_LENGTH})")


# =========================================================================
# STEP 5: PLOTS
# =========================================================================

os.makedirs("plots", exist_ok=True)

# --- Training curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, NUM_EPOCHS+1), train_losses, label='Train', linewidth=2)
axes[0].plot(range(1, NUM_EPOCHS+1), val_losses, label='Validation', linewidth=2)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss (BCE)')
axes[0].set_title('Training and Validation Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, NUM_EPOCHS+1), train_accs, label='Train', linewidth=2)
axes[1].plot(range(1, NUM_EPOCHS+1), val_accs, label='Validation', linewidth=2)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('plots/training_curves.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: plots/training_curves.png")

# --- Trained embeddings PCA ---
try:
    from sklearn.decomposition import PCA
    embeddings = model.get_embeddings().numpy()

    pos_words = ["great", "wonderful", "excellent", "brilliant", "amazing",
                 "loved", "best", "beautiful", "superb", "fantastic",
                 "outstanding", "incredible", "masterful", "perfect", "joy"]
    neg_words = ["terrible", "awful", "worst", "boring", "disappointing",
                 "horrible", "bad", "dull", "painful", "cheap",
                 "disaster", "mess", "waste", "forgettable", "unwatchable"]

    words, vecs, cats = [], [], []
    for w in pos_words:
        if w in vocab.word2idx:
            words.append(w); vecs.append(embeddings[vocab.word2idx[w]]); cats.append("positive")
    for w in neg_words:
        if w in vocab.word2idx:
            words.append(w); vecs.append(embeddings[vocab.word2idx[w]]); cats.append("negative")

    if len(vecs) > 4:
        pca = PCA(n_components=2)
        vecs_2d = pca.fit_transform(np.array(vecs))

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = {"positive": "#2ecc71", "negative": "#e74c3c"}
        for i, (w, cat) in enumerate(zip(words, cats)):
            ax.scatter(vecs_2d[i, 0], vecs_2d[i, 1], c=colors[cat], s=120,
                       edgecolors='black', linewidth=1.5, alpha=0.8)
            ax.annotate(w, (vecs_2d[i, 0], vecs_2d[i, 1]), fontsize=9,
                        ha='center', va='bottom', xytext=(0, 6),
                        textcoords='offset points', fontweight='bold')
        for cat, color in colors.items():
            ax.scatter([], [], c=color, s=120, label=cat, edgecolors='black')
        ax.legend(fontsize=12)
        ax.set_title('Trained Word Embeddings (PCA Projection)\n'
                     'Positive and negative words should form separate clusters')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/embeddings_pca.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved: plots/embeddings_pca.png")
except ImportError:
    print("(sklearn not installed — skipping PCA plot)")


# =========================================================================
# STEP 6: QUICK INFERENCE TEST
# =========================================================================
from activity1_preprocessing import preprocess_for_model

test_texts = [
    "I absolutely loved this movie it was fantastic",
    "What a terrible waste of time avoid this",
    "A great film with wonderful performances",
    "The worst movie I have ever seen in my life",
    "Pretty good movie overall enjoyable experience",
    "Boring dull and completely uninspired throughout",
]

print(f"\nInference test on unseen reviews:")
print("-" * 55)
model.eval()
for text in test_texts:
    tensor = preprocess_for_model(text, vocab, MAX_LENGTH)
    with torch.no_grad():
        prob = model(tensor).item()
    label = "POS" if prob >= 0.5 else "NEG"
    bar = "#" * int(prob * 20) + "." * (20 - int(prob * 20))
    print(f"  [{bar}] {prob:.2f} {label}  \"{text}\"")

print("\n" + "=" * 65)
print("  ACTIVITY 3 COMPLETE")
print("  Model and vocabulary saved to saved_model/")
print("  Next: Activity 4 — Build the web app and deploy it!")
print("=" * 65)

# ---- REFLECTION QUESTIONS ----
#
# 1. Look at the training curves plot. Does the model overfit?
#    How can you tell? What changes would reduce overfitting?
#    (More data, more dropout, fewer epochs, simpler model)
#
# 2. The training loop is the SAME as Topics 1-3. Which line of code
#    is the "forward pass"? The "backward pass"? The "weight update"?
#    Why do we zero gradients at the start of each batch?
#
# 3. Why do we build the vocabulary ONLY on training data?
#    What would go wrong if we included validation data words?
#
# 4. We saved model.pt, vocab.json, and config.json.
#    Why does the app need ALL THREE to make predictions?
#    What happens if vocab.json doesn't match the model's vocab_size?
