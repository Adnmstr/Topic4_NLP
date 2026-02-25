"""
============================================================================
AIT-204 Deep Learning | Topic 4: Natural Language Processing
ACTIVITY 4 — Part A: Backend Service Layer
============================================================================

FRONTEND / BACKEND ARCHITECTURE
──────────────────────────────────────────────────────────────────────────
Modern web applications split responsibilities into two layers:

  BACKEND  — "what the app does" (business logic, data, ML inference)
  FRONTEND — "what the user sees" (web pages, buttons, charts)

    ┌──────────────────────────────────────────────────────────────────┐
    │  FRONTEND  (activity4_app.py — Streamlit)                        │
    │    Creates web pages. Handles user input. Calls backend methods. │
    │    Formats returned data as visual Streamlit components.         │
    ├──────────────────────────────────────────────────────────────────┤
    │  BACKEND   (THIS FILE — model_service.py)                        │
    │    Loads model. Runs ML pipeline. Returns structured dicts.      │
    │    Zero UI code. No Streamlit imports.                           │
    └──────────────────────────────────────────────────────────────────┘

This clean separation means:
  - You can swap Streamlit for React/Next.js without touching model code.
  - You can replace the model without touching the UI code.
  - Each layer is independently testable.

PRODUCTION EXTENSION: FastAPI Backend (optional)
──────────────────────────────────────────────────────────────────────────
Here, frontend and backend run in the same Python process (monolithic).
In production, you would expose this service as a REST API so any frontend
(Streamlit, React, mobile app) can call it over HTTP:

    # pip install fastapi uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()
    service = SentimentService()

    class TextIn(BaseModel):
        text: str

    @app.post("/predict")
    def predict(req: TextIn):
        return service.predict(req.text)       # same method, now an endpoint

    @app.post("/compare")
    def compare(req: TextIn, req2: TextIn):
        return service.compare(req.text, req2.text)

    # Run with: uvicorn model_service:app --reload
    # Frontend calls: requests.post("http://localhost:8000/predict", json={"text": "..."})
──────────────────────────────────────────────────────────────────────────

WHAT YOU WILL IMPLEMENT (TODOs 1–4):
    TODO 1: Load the vocabulary from disk
    TODO 2: Load the trained model from disk
    TODO 3: Implement predict()  — runs the full NLP pipeline
    TODO 4: Implement compare()  — scores two texts and diffs them

RUN THIS FILE:  python model_service.py   (runs a quick self-test)
NEXT FILE:      activity4_app.py          (imports and uses this service)
============================================================================
"""

import json
import torch

from activity1_preprocessing import (
    Vocabulary, clean_text, tokenize, preprocess_for_model
)
from activity2_model import load_model

MODEL_DIR = "saved_model"


# =========================================================================
# THE BACKEND SERVICE CLASS
# =========================================================================
# Think of SentimentService as a mini-API server.
# Each public method = one API endpoint.
# The Streamlit frontend creates ONE instance and calls its methods.
# =========================================================================

class SentimentService:
    """
    Backend service: wraps the trained model and exposes clean prediction
    methods to the frontend. Contains zero UI code.

    Each method maps to a conceptual REST endpoint:
        predict()   ──▶  POST /api/predict
        compare()   ──▶  POST /api/compare
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        """
        Load all model artifacts from disk.
        Called ONCE at app startup (Streamlit caches this with @st.cache_resource).
        """

        # ── TODO 1 ────────────────────────────────────────────────────────
        # Load the vocabulary saved by Activity 3.
        #
        # HINT: Use Vocabulary.load(path) where path = f"{model_dir}/vocab.json"
        #       Store the result as self.vocab
        self.vocab = Vocabulary.load(f"{model_dir}/vocab.json")

        # ── TODO 2 ────────────────────────────────────────────────────────
        # Load the trained model saved by Activity 3.
        #
        # HINT: Use load_model(path) where path = f"{model_dir}/model.pt"
        #       Then call self.model.eval() to disable Dropout for inference.
        #       Store the result as self.model
        self.model = load_model(f"{model_dir}/model.pt")
        self.model.eval()

        # Load max_length from config (already done for you)
        with open(f"{model_dir}/config.json") as f:
            config = json.load(f)
        self.max_length = config["max_length"]

        print(f"[Backend] Model loaded  ({sum(p.numel() for p in self.model.parameters()):,} params)")
        print(f"[Backend] Vocabulary    ({len(self.vocab)} words)")
        print(f"[Backend] Max length    ({self.max_length} tokens)")

    # ──────────────────────────────────────────────────────────────────────
    # ENDPOINT 1:  predict(text)  →  dict
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Backend endpoint: run the full NLP pipeline on one review.

        The frontend calls this method and receives a plain dict.
        It then formats the dict as Streamlit components — the backend
        never needs to know what the UI looks like.

        Args:
            text (str): Raw review text from the user.

        Returns:
            dict with keys:
                sentiment      (str):   "Positive" or "Negative"
                confidence     (float): 0.0 – 1.0
                positive_score (float): raw sigmoid output
                negative_score (float): 1 - positive_score
                cleaned        (str):   text after clean_text()
                tokens         (list):  string tokens
                encoded        (list):  integer IDs (capped at max_length)
                known_count    (int):   number of tokens found in vocab
        """
        # Handle empty input
        if not text or not text.strip():
            return {
                "sentiment": "Unknown",  "confidence": 0.5,
                "positive_score": 0.5,   "negative_score": 0.5,
                "cleaned": "",           "tokens": [],
                "encoded": [],           "known_count": 0,
            }

        # ── TODO 3 ────────────────────────────────────────────────────────
        # Implement the full prediction pipeline. Five steps:
        #
        # Step 1 — Preprocess (for the "show your work" display in the UI):
        #   cleaned = clean_text(text)
        #   tokens  = tokenize(cleaned)
        #   encoded = self.vocab.encode(tokens)
        #
        # Step 2 — Build the input tensor for the model:
        #   tensor  = preprocess_for_model(text, self.vocab, self.max_length)
        #
        # Step 3 — Run inference (ALWAYS use torch.no_grad() at inference time):
        #   with torch.no_grad():
        #       probability = self.model(tensor).item()
        #
        # Step 4 — Determine label and confidence:
        #   if probability >= 0.5:  sentiment = "Positive",  confidence = probability
        #   else:                   sentiment = "Negative",  confidence = 1 - probability
        #
        # Step 5 — Return the dict. Compute known_count as:
        #   sum(1 for t in tokens if t in self.vocab.word2idx)

        cleaned     = clean_text(text)
        tokens      = tokenize(cleaned)
        encoded     = self.vocab.encode(tokens)
        tensor      = preprocess_for_model(text, self.vocab, self.max_length)

        with torch.no_grad():
            probability = self.model(tensor).item()

        sentiment   = "Positive" if probability >= 0.5 else "Negative"
        confidence  = probability if probability >= 0.5 else 1 - probability
        known_count = sum(1 for t in tokens if t in self.vocab.word2idx)

        return {
            "sentiment":      sentiment,
            "confidence":     confidence,
            "positive_score": float(probability),
            "negative_score": float(1 - probability),
            "cleaned":        cleaned,
            "tokens":         tokens,
            "encoded":        encoded[:self.max_length],
            "known_count":    known_count,
        }

    # ──────────────────────────────────────────────────────────────────────
    # ENDPOINT 2:  compare(original, translated)  →  dict
    # ──────────────────────────────────────────────────────────────────────

    def compare(self, original: str, translated: str) -> dict:
        """
        Backend endpoint: score two texts and return a comparison dict.
        Used by the Translation Comparison tab in the frontend.

        Args:
            original   (str): Original English review.
            translated (str): Round-trip translated version.

        Returns:
            dict with keys:
                original   (dict): Full predict() result for original text
                translated (dict): Full predict() result for translated text
                delta      (float): translated score − original score (signed)
                changed    (bool):  True if sentiment label flipped
                lost_words (list):  vocab words present in original, absent in translated
                new_words  (list):  vocab words absent in original, present in translated
        """

        # ── TODO 4 ────────────────────────────────────────────────────────
        # Step 1 — Run predict() on both texts:
        #   orig_result  = self.predict(original)
        #   trans_result = self.predict(translated)
        #
        # Step 2 — Compute the score delta and changed flag:
        #   delta   = trans_result["positive_score"] - orig_result["positive_score"]
        #   changed = orig_result["sentiment"] != trans_result["sentiment"]
        #
        # Step 3 — Find words gained or lost during translation:
        #   orig_words  = set(t for t in orig_result["tokens"]
        #                     if t in self.vocab.word2idx)
        #   trans_words = set(t for t in trans_result["tokens"]
        #                     if t in self.vocab.word2idx)
        #   lost_words  = sorted(orig_words - trans_words)
        #   new_words   = sorted(trans_words - orig_words)
        #
        # Step 4 — Return the dict described in the docstring above.

        orig_result  = self.predict(original)
        trans_result = self.predict(translated)
        delta        = trans_result["positive_score"] - orig_result["positive_score"]
        changed      = orig_result["sentiment"] != trans_result["sentiment"]
        lost_words   = sorted(set(t for t in orig_result["tokens"] if t in self.vocab.word2idx) - set(t for t in trans_result["tokens"] if t in self.vocab.word2idx))
        new_words    = sorted(set(t for t in trans_result["tokens"] if t in self.vocab.word2idx) - set(t for t in orig_result["tokens"] if t in self.vocab.word2idx))

        return {
            "original":   orig_result,
            "translated": trans_result,
            "delta":      delta,
            "changed":    changed,
            "lost_words": lost_words,
            "new_words":  new_words,
        }


# =========================================================================
# SELF-TEST  (python model_service.py)
# =========================================================================
# Run this file directly to verify the backend works before building the UI.
# =========================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("  Backend Service — Self-Test")
    print("=" * 62)

    svc = SentimentService()

    print("\n[Test 1] predict() — positive review:")
    r = svc.predict("This movie was absolutely wonderful and I loved it")
    print(f"  Sentiment   : {r['sentiment']} ({r['confidence']:.1%} confidence)")
    print(f"  Score       : {r['positive_score']:.4f}")
    print(f"  Tokens      : {r['tokens']}")
    print(f"  Vocab hit   : {r['known_count']}/{len(r['tokens'])}")

    print("\n[Test 2] predict() — negative review:")
    r2 = svc.predict("Awful acting and a terrible waste of time from start to finish")
    print(f"  Sentiment   : {r2['sentiment']} ({r2['confidence']:.1%} confidence)")
    print(f"  Score       : {r2['positive_score']:.4f}")

    print("\n[Test 3] compare() — round-trip translation:")
    cmp = svc.compare(
        "This film was absolutely brilliant and moving",
        "This film was completely bright and moving",
    )
    print(f"  Original    : {cmp['original']['sentiment']}  ({cmp['original']['positive_score']:.4f})")
    print(f"  Translated  : {cmp['translated']['sentiment']} ({cmp['translated']['positive_score']:.4f})")
    print(f"  Delta       : {cmp['delta']:+.4f}")
    print(f"  Changed     : {cmp['changed']}")
    print(f"  Lost words  : {cmp['lost_words']}")
    print(f"  New words   : {cmp['new_words']}")

    print("\n" + "=" * 62)
    print("  Backend self-test complete.")
    print("  Next: streamlit run activity4_app.py")
    print("=" * 62)
