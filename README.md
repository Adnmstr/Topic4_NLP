# Topic 4: Deep Learning for Natural Language Processing (NLP)

## AIT-204 Deep Learning | Grand Canyon University

**Dates:** Feb 16, 2026 - Mar 1, 2026 | **Max Points:** 240

---

## Why This Topic Was Modernized

The original Topic 4 assignment asked students to compare online machine translation tools and write a report. While translation analysis is valuable, it contained **no deep learning coding** — a gap for a course titled "Deep Learning." Students who completed Topics 1-3 built neural networks in code. Topic 4 continues that momentum by having students build and **deploy a real NLP application** using a proper frontend/backend architecture.

### What Changed

| Aspect | Original | Modernized |
|--------|----------|------------|
| **Coding** | None | Full PyTorch sentiment analysis pipeline |
| **Architecture** | None | Frontend (Streamlit) + Backend (model_service.py) |
| **Deployment** | None | Streamlit Community Cloud (free, 3-step GitHub deploy) |
| **Deep Learning** | None | Embeddings, classifier, training, inference |
| **NLP Concepts** | Surface-level | Tokenization, embeddings, classification, evaluation |
| **Translation** | Entire assignment | Retained as a model-powered app feature |
| **Deliverable** | Word doc | Live deployed app + code + report |

### What Stayed

- All 3 original objectives still covered
- Translation analysis retained (now powered by the student's own model)
- Ethical considerations required
- Report deliverable preserved

---

## Application Architecture: Frontend / Backend

Every Activity 4 app follows this two-layer architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│  FRONTEND  ─  activity4_app.py  (Streamlit)                         │
│                                                                     │
│   st.title()   st.text_area()   st.button()   st.metric()          │
│   Creates web pages. Handles user input. Calls service methods.     │
│   Formats returned dicts as visual components. Zero model code.     │
├─────────────────────────────────────────────────────────────────────┤
│  BACKEND   ─  model_service.py  (Python / PyTorch)                  │
│                                                                     │
│   SentimentService.predict(text)   → dict                           │
│   SentimentService.compare(t1, t2) → dict                           │
│   Loads model. Runs ML pipeline. Returns structured data.           │
│   Zero UI code. Independently testable (python model_service.py).   │
└─────────────────────────────────────────────────────────────────────┘
          ↑ imported by              ↑ imports from
          activity4_app.py           activity1_preprocessing.py
                                     activity2_model.py
                                     saved_model/
```

**Why this separation matters:**
- You can swap Streamlit for React/Next.js without touching model code.
- You can change the model without touching the UI code.
- Each layer is independently testable.
- This is the architecture used in production ML systems.

**Optional Extension — FastAPI backend:**
Refactor `model_service.py` as a REST API so any frontend can call it over HTTP. See the comments in `model_service.py` for the FastAPI pattern.

---

## The Full Pipeline: Concept → Algorithm → Code → Deployed App

```
NLP Concept          → Algorithm              → Python Code              → Deployed Feature
─────────────────────────────────────────────────────────────────────────────────────────────
"Text must become     Tokenize, build vocab,   Vocabulary class,          User types text,
 numbers for NNs"     encode, pad              clean_text(), tokenize()   app preprocesses it

"Words need meaning,  Lookup table learned     nn.Embedding layer         Preprocessing
 not just IDs"        during training                                     expander in app

"Classify sentiment   Embed → Pool → Linear   SentimentClassifier        Prediction with
 from text"           → ReLU → Linear → Sig   class, training loop       confidence bar

"How does translation Run original + translated service.compare()         Translation
 affect prediction?"  through model, compare   on both versions           comparison tab
```

---

## Project Structure

```
Topic4_NLP/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
│
├── activity1_preprocessing.py       # Activity 1: Text preprocessing (9 TODOs)
├── activity2_model.py               # Activity 2: Model architecture (5 TODOs)
├── activity3_train.py               # Activity 3: Training pipeline (3 TODOs)
│
├── model_service.py                 # Activity 4a: BACKEND service (TODOs 1–4)
├── activity4_app.py                 # Activity 4b: FRONTEND Streamlit app (TODOs 5–7)
│
├── saved_model/                     # Created by Activity 3
│   ├── model.pt
│   ├── vocab.json
│   └── config.json
└── plots/                           # Created by Activity 3
    └── training_curves.png
```

**Data layer** → Activity 1 (`Vocabulary`, `preprocess_for_model`)
**Model layer** → Activity 2 (`SentimentClassifier`, `load_model`)
**Training**    → Activity 3 (saves artifacts to `saved_model/`)
**Backend**     → `model_service.py` (`SentimentService` wraps activities 1 & 2)
**Frontend**    → `activity4_app.py` (Streamlit UI calls backend service)

---

## 4 In-Class Activities

### Activity 1: Text Preprocessing Module (~75 min)
**File:** `activity1_preprocessing.py` | **TODOs:** 9
**What you build:** A reusable `Vocabulary` class and text cleaning functions. Run it standalone to see tokenization, encoding, and padding in action.
**Key bridge:** "In Topics 1-3, inputs were already numbers. In NLP, you must first convert words to tensors. This module is the front door of your app."

### Activity 2: Model Architecture (~75 min)
**File:** `activity2_model.py` | **TODOs:** 5
**What you build:** A `SentimentClassifier` class (`nn.Module`) using embeddings and feed-forward layers.
**Key bridge:** "This model is the same as Topic 2's neural network — we just added an embedding layer on the front to handle text input."

### Activity 3: Training Pipeline (~90 min)
**File:** `activity3_train.py` | **TODOs:** 3
**What you build:** A complete training script that trains the model, plots loss curves, and saves the model to disk for the app to load.
**Key bridge:** "Same training loop as Topics 1-3: forward pass, loss, backward, step. But now you save the result so it can be loaded by a web app."

### Activity 4: Deploy as a Web App (~90 min)
**Files:** `model_service.py` (TODOs 1–4) + `activity4_app.py` (TODOs 5–7)
**What you build:**
- `model_service.py` — the backend: loads model, exposes `predict()` and `compare()` as clean Python methods (the "API layer")
- `activity4_app.py` — the Streamlit frontend: creates the web UI, calls the backend, formats results

**Default deployment:** Streamlit Community Cloud (free, deploys from GitHub in 3 steps)
**Optional extension:** Refactor the backend as FastAPI + any frontend (React, plain HTML)

**Key bridge:** "You've gone from NLP concept to deployed app following the same frontend/backend separation used in production ML systems."

---

## Rewritten Assignment: "NLP Sentiment Analysis — From Model to Deployed App"

**Points:** 70 | **Due:** Mar 1, 2026

### Part 1: The ML Pipeline (35 points)

Complete Activities 1-3 to build the full machine learning pipeline:

1. **Preprocessing Module** (Activity 1) — tokenization, vocabulary, encoding, padding
2. **Model Architecture** (Activity 2) — `SentimentClassifier`: Embedding → Pool → FC
3. **Training and Evaluation** (Activity 3) — train/val split, loss curves, save artifacts

### Part 2: The Deployed Web App (25 points)

Complete Activity 4 to build and deploy the application:

1. **Backend service** (`model_service.py`) — `SentimentService.predict()` and `.compare()`
2. **Frontend** (`activity4_app.py`) — Sentiment Analysis tab + Translation Comparison tab
3. **Deployment** — Live URL on Streamlit Community Cloud (or equivalent)

### Part 3: Report and Ethical Considerations (10 points)

- Translation analysis: 10+ round-trip examples with error classification
- Ethical discussion: bias, privacy, deployment risks, engineering responsibility
- Architecture diagram showing the frontend/backend split

### Deliverables

1. **Source Code** — All Python files, fully commented
2. **Deployed App** — Live URL on Streamlit Cloud (or equivalent)
3. **Report** — PDF/docx with architecture diagram, analysis, screenshots, ethics
4. **Video/Presentation** — Demonstrate the app, explain design decisions

---

## Deployment Guide (Streamlit Community Cloud)

```bash
# Step 1: Make sure your app runs locally
streamlit run activity4_app.py
# → opens in browser at http://localhost:8501

# Step 2: Push to GitHub
git init
git add .
git commit -m "Topic 4 NLP sentiment app"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main

# Step 3: Deploy at share.streamlit.io
# → New app → connect repo → main file: activity4_app.py → Deploy
# → Live URL generated in ~2 minutes
```

**Files that must be in the repo root:**
```
activity4_app.py          ← Streamlit entry point
model_service.py
activity1_preprocessing.py
activity2_model.py
saved_model/model.pt
saved_model/vocab.json
saved_model/config.json
requirements.txt
```

**Alternative deployment:** Hugging Face Spaces — select "Streamlit" as the SDK. Same files, same process.

---

## Objectives Mapping

| Objective | Where Covered |
|-----------|--------------|
| 1. Identify errors introduced during translation | Translation Comparison tab + Assignment Part 3 |
| 2. Analyze language structure and features | Activities 1-2 (tokenization, embeddings) |
| 3. Apply foundational ML and NLP concepts | Activities 1-4 end-to-end |

---

## Setup Instructions

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run activities in order
python activity1_preprocessing.py   # verify preprocessing works
python activity2_model.py           # verify model architecture
python activity3_train.py           # train the model, saves to saved_model/
python model_service.py             # test the backend service
streamlit run activity4_app.py      # launch the full app locally
```

## Prerequisites from Previous Topics

- **Topic 1:** Gradient descent, loss functions, forward/backward pass
- **Topic 2:** Neural network layers, activation functions, training loops
- **Topic 3:** Feature extraction concept (CNNs for images; embeddings for text)
