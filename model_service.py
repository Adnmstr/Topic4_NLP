"""
============================================================================
AIT-204 Deep Learning | Topic 4: Natural Language Processing
ACTIVITY 4 — Part A: Backend Service Layer
============================================================================

This backend supports multiple saved model folders.

Supported models:
  - Binary sentiment model (Topic 4)      -> saved_model/
  - Multi-level sentiment model (Topic 5) -> saved_model_multilevel/
  - Multi-task sentiment+intent model      -> saved_model_multitask/
============================================================================
"""

import json
import os
import torch

from activity1_preprocessing import (
    Vocabulary, clean_text, tokenize, preprocess_for_model
)

# These loaders must match how the checkpoints were saved.
from activity2_model import load_model as load_binary_model
from activity_part1_multilevel import load_model as load_multilevel_model, SENTIMENT_LABELS
from activity_part2_intent import load_dual_model, INTENT_LABELS


def _read_config(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config.json at: {cfg_path}")
    with open(cfg_path, "r") as f:
        return json.load(f)


def _infer_model_type(model_dir: str, config: dict) -> str:
    """
    Decide model type from directory name or config.
    """
    d = model_dir.lower()

    # Explicit folder naming (most reliable for your setup)
    if "multilevel" in d:
        return "multilevel"
    if "multitask" in d:
        return "multitask"

    # Config hint (if present)
    # If num_classes exists and > 1, it's multilevel.
    if "num_classes" in config:
        try:
            if int(config["num_classes"]) > 1:
                return "multilevel"
        except Exception:
            pass

    return "binary"


class SentimentService:
    """
    Backend service: wraps a trained model and exposes prediction methods.
    Works for both binary and multilevel sentiment models.
    """

    def __init__(self, model_dir: str = "saved_model"):
        self.model_dir = model_dir

        # ---- Load artifacts ----
        self.config = _read_config(model_dir)
        self.max_length = int(self.config["max_length"])

        vocab_path = os.path.join(model_dir, "vocab.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Missing vocab.json at: {vocab_path}")
        self.vocab = Vocabulary.load(vocab_path)

        model_path = os.path.join(model_dir, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model.pt at: {model_path}")

        # ---- Determine which loader to use ----
        self.model_type = _infer_model_type(model_dir, self.config)

        if self.model_type == "binary":
            self.model = load_binary_model(model_path)
        elif self.model_type == "multilevel":
            self.model = load_multilevel_model(model_path)
            self.num_classes = int(self.config.get("num_classes", 7))
        elif self.model_type == "multitask":
            self.model = load_dual_model(model_path)
            self.num_sentiments = int(self.config.get("num_sentiments", 7))
            self.num_intents = int(self.config.get("num_intents", 8))
            self.intent_labels = INTENT_LABELS
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.model.eval()

        print(f"[Backend] Model type     : {self.model_type}")
        print(f"[Backend] Model loaded   : {model_path}")
        print(f"[Backend] Vocabulary     : {len(self.vocab)} words")
        print(f"[Backend] Max length     : {self.max_length} tokens")
        print(f"[Backend] Params         : {sum(p.numel() for p in self.model.parameters()):,}")

    # ------------------------------------------------------------------
    # predict(text) -> dict
    # ------------------------------------------------------------------
    def predict(self, text: str) -> dict:
        # Handle empty input
        if not text or not text.strip():
            return {
                "model_type": self.model_type,
                "sentiment": "Unknown",
                "confidence": 0.0,
                "positive_score": 0.0,
                "negative_score": 0.0,
                "score_scalar": 0.0,   # always 0..1
                "scores": None,        # class distribution for multilevel
                "cleaned": "",
                "tokens": [],
                "encoded": [],
                "known_count": 0,
            }

        # Preprocess (for display)
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        encoded = self.vocab.encode(tokens)
        tensor = preprocess_for_model(text, self.vocab, self.max_length)

        known_count = sum(1 for t in tokens if t in self.vocab.word2idx)

        # =========================
        # Binary model (sigmoid)
        # =========================
        if self.model_type == "binary":
            with torch.no_grad():
                prob = float(self.model(tensor).item())  # shape (1,1) -> scalar

            sentiment = "Positive" if prob >= 0.5 else "Negative"
            confidence = prob if prob >= 0.5 else (1.0 - prob)

            return {
                "model_type": "binary",
                "sentiment": sentiment,
                "confidence": confidence,
                "positive_score": prob,
                "negative_score": 1.0 - prob,
                "score_scalar": prob,           # for compare() delta
                "scores": {"Positive": prob, "Negative": 1.0 - prob},
                "cleaned": cleaned,
                "tokens": tokens,
                "encoded": encoded[: self.max_length],
                "known_count": known_count,
            }

        # =========================
        # Multilevel model (logits -> softmax)
        # =========================
        if self.model_type == "multilevel":
            with torch.no_grad():
                logits = self.model(tensor)  # shape (1, 7)
                probs = torch.softmax(logits, dim=1).squeeze(0)  # shape (7,)

            pred_id = int(probs.argmax().item())
            confidence = float(probs[pred_id].item())

            # Label mapping (from your multilevel script)
            label = SENTIMENT_LABELS.get(pred_id, f"Class {pred_id}")

            # Provide a stable 0..1 scalar so compare() can still work.
            # Here: expected class index normalized to [0..1]
            k = int(probs.numel())
            expected = float((probs * torch.arange(k)).sum().item())
            score_scalar = expected / max(1, (k - 1))

            scores = {SENTIMENT_LABELS.get(i, f"Class {i}"): float(probs[i].item()) for i in range(k)}

            return {
                "model_type": "multilevel",
                "sentiment": label,
                "confidence": confidence,
                "positive_score": score_scalar,      # kept for UI compatibility
                "negative_score": 1.0 - score_scalar,
                "score_scalar": score_scalar,
                "scores": scores,                    # full distribution
                "cleaned": cleaned,
                "tokens": tokens,
                "encoded": encoded[: self.max_length],
                "known_count": known_count,
            }

        # =========================
        # Multitask model (sentiment + intent)
        # =========================
        if self.model_type == "multitask":
            with torch.no_grad():
                sentiment_logits, intent_logits = self.model(tensor)

                # Sentiment
                sentiment_probs = torch.softmax(sentiment_logits, dim=1).squeeze(0)
                sentiment_id = int(sentiment_probs.argmax().item())
                sentiment_conf = float(sentiment_probs[sentiment_id].item())
                sentiment_label = SENTIMENT_LABELS.get(sentiment_id, f"Class {sentiment_id}")

                k = int(sentiment_probs.numel())
                expected = float((sentiment_probs * torch.arange(k)).sum().item())
                score_scalar = expected / max(1, (k - 1))

                sentiment_scores = {
                    SENTIMENT_LABELS.get(i, f"Class {i}"): float(sentiment_probs[i].item())
                    for i in range(k)
                }

                # Intent
                intent_probs = torch.softmax(intent_logits, dim=1).squeeze(0)
                intent_id = int(intent_probs.argmax().item())
                intent_conf = float(intent_probs[intent_id].item())
                intent_label = self.intent_labels.get(intent_id, f"Class {intent_id}")

                intent_scores = {
                    self.intent_labels.get(i, f"Class {i}"): float(intent_probs[i].item())
                    for i in range(int(intent_probs.numel()))
                }

            return {
                "model_type": "multitask",
                "sentiment": sentiment_label,
                "confidence": sentiment_conf,
                "positive_score": score_scalar,
                "negative_score": 1.0 - score_scalar,
                "score_scalar": score_scalar,
                "scores": sentiment_scores,
                "intent": intent_label,
                "intent_confidence": intent_conf,
                "intent_scores": intent_scores,
                "cleaned": cleaned,
                "tokens": tokens,
                "encoded": encoded[: self.max_length],
                "known_count": known_count,
            }

        raise RuntimeError(f"Unhandled model_type: {self.model_type}")

    # ------------------------------------------------------------------
    # compare(original, translated) -> dict
    # ------------------------------------------------------------------
    def compare(self, original: str, translated: str) -> dict:
        orig_result = self.predict(original)
        trans_result = self.predict(translated)

        delta = float(trans_result["score_scalar"] - orig_result["score_scalar"])
        changed = orig_result["sentiment"] != trans_result["sentiment"]

        orig_words = set(t for t in orig_result["tokens"] if t in self.vocab.word2idx)
        trans_words = set(t for t in trans_result["tokens"] if t in self.vocab.word2idx)

        lost_words = sorted(orig_words - trans_words)
        new_words = sorted(trans_words - orig_words)

        return {
            "original": orig_result,
            "translated": trans_result,
            "delta": delta,
            "changed": changed,
            "lost_words": lost_words,
            "new_words": new_words,
        }

    # ------------------------------------------------------------------
    # analyze_with_translation(text, target_lang) -> dict
    # ------------------------------------------------------------------
    def analyze_with_translation(self, text: str, target_lang: str = "es") -> dict:
        """
        Full pipeline: translate round-trip, compare, classify errors.
        """
        translation = translate_round_trip(text, target_lang)
        comparison = self.compare(text, translation["back_translated"])
        errors = classify_translation_errors(comparison)

        return {
            **comparison,
            "intermediate_lang": translation["intermediate_lang"],
            "intermediate_text": translation["intermediate_text"],
            "back_translated": translation["back_translated"],
            "errors": errors,
        }


# ------------------------------------------------------------------
# Translation helpers
# ------------------------------------------------------------------
SUPPORTED_LANGUAGES = {
    "Spanish": "es-ES",
    "French": "fr-FR",
    "German": "de-DE",
    "Italian": "it-IT",
    "Portuguese": "pt-PT",
    "Japanese": "ja-JP",
    "Chinese (Simplified)": "zh-CN",
    "Korean": "ko-KR",
    "Russian": "ru-RU",
    "Arabic": "ar-SA",
    "Hindi": "hi-IN",
}


def translate_round_trip(text: str, target_lang: str = "es-ES") -> dict:
    """
    Translate text English -> target language -> back to English.
    Uses MyMemoryTranslator (free, no API key, 500 char limit).
    """
    from deep_translator import MyMemoryTranslator

    # MyMemory enforces a 500-character limit per request
    text = text[:500]

    forward = MyMemoryTranslator(source="en-GB", target=target_lang)
    intermediate = forward.translate(text)

    backward = MyMemoryTranslator(source=target_lang, target="en-GB")
    back_translated = backward.translate(intermediate)

    return {
        "intermediate_lang": target_lang,
        "intermediate_text": intermediate,
        "back_translated": back_translated,
    }


def classify_translation_errors(compare_result: dict) -> list:
    """
    Classify translation errors from a compare() result.
    Returns a list of {"type", "description", "severity"} dicts.
    """
    errors = []
    orig = compare_result["original"]
    trans = compare_result["translated"]
    lost = compare_result["lost_words"]
    new = compare_result["new_words"]

    orig_tokens = orig.get("tokens", [])
    trans_tokens = trans.get("tokens", [])

    # Omission: known words dropped
    if lost:
        sev = "high" if len(lost) > 3 else ("medium" if len(lost) > 1 else "low")
        errors.append({
            "type": "Omission",
            "description": f"{len(lost)} word(s) lost: {', '.join(lost)}",
            "severity": sev,
        })

    # Addition: new words introduced
    if new:
        sev = "medium" if len(new) > 2 else "low"
        errors.append({
            "type": "Addition",
            "description": f"{len(new)} new word(s): {', '.join(new)}",
            "severity": sev,
        })

    # Lexical substitution: tokens changed but total count similar
    shared = set(orig_tokens) & set(trans_tokens)
    only_orig = set(orig_tokens) - shared
    only_trans = set(trans_tokens) - shared
    if only_orig and only_trans and not lost and not new:
        errors.append({
            "type": "Lexical Substitution",
            "description": (
                f"Words swapped: {', '.join(sorted(only_orig))} "
                f"-> {', '.join(sorted(only_trans))}"
            ),
            "severity": "low",
        })

    # Reordering: same tokens, different positions
    if sorted(orig_tokens) == sorted(trans_tokens) and orig_tokens != trans_tokens:
        errors.append({
            "type": "Reordering",
            "description": "Same words appear in a different order",
            "severity": "low",
        })

    # Semantic drift: prediction label changed
    if compare_result["changed"]:
        errors.append({
            "type": "Semantic Drift",
            "description": (
                f"Prediction changed: {orig['sentiment']} -> {trans['sentiment']} "
                f"(delta: {compare_result['delta']:+.3f})"
            ),
            "severity": "high",
        })

    # Confidence shift: same label but notable delta
    delta = abs(compare_result["delta"])
    if not compare_result["changed"] and delta > 0.05:
        sev = "medium" if delta > 0.15 else "low"
        errors.append({
            "type": "Confidence Shift",
            "description": (
                f"Same prediction ({orig['sentiment']}) but score shifted "
                f"by {compare_result['delta']:+.3f}"
            ),
            "severity": sev,
        })

    # No errors detected
    if not errors:
        errors.append({
            "type": "Preserved",
            "description": "Translation preserved meaning with no detectable errors",
            "severity": "low",
        })

    return errors


if __name__ == "__main__":
    print("=" * 62)
    print("  Backend Service — Self-Test")
    print("=" * 62)

    svc = SentimentService("saved_model")
    r = svc.predict("This movie was absolutely wonderful and I loved it")
    print("\n[Test binary]")
    print(r["model_type"], r["sentiment"], r["confidence"], r["score_scalar"])

    try:
        svc2 = SentimentService("saved_model_multilevel")
        r2 = svc2.predict("This movie was absolutely wonderful and I loved it")
        print("\n[Test multilevel]")
        print(r2["model_type"], r2["sentiment"], r2["confidence"], r2["score_scalar"])
    except Exception as e:
        print("\n[Multilevel test skipped]", e)

    try:
        svc3 = SentimentService("saved_model_multitask")
        r3 = svc3.predict("This movie was absolutely wonderful and I loved it")
        print("\n[Test multitask]")
        print(r3["model_type"], r3["sentiment"], r3["confidence"], r3["score_scalar"])
        print("  Intent:", r3["intent"], r3["intent_confidence"])
    except Exception as e:
        print("\n[Multitask test skipped]", e)

    print("\n" + "=" * 62)