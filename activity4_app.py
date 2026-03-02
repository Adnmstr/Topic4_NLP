import os
import re
import time
from typing import Dict, List, Tuple

import streamlit as st

from model_service import SUPPORTED_LANGUAGES, SentimentService
from project_paths import (
    BINARY_MODEL_DIR,
    MULTILEVEL_MODEL_DIR,
    MULTITASK_MODEL_DIR,
    BINARY_PLOTS_DIR,
    MULTILEVEL_PLOTS_DIR,
    MULTITASK_PLOTS_DIR,
)


st.set_page_config(
    page_title="Movie NLP Analysis",
    page_icon="NLP",
    layout="wide",
)


MODEL_OPTIONS = {
    "Binary Sentiment (default)": BINARY_MODEL_DIR,
    "Multilevel Sentiment": MULTILEVEL_MODEL_DIR,
    "Multitask (Sentiment + Intent)": MULTITASK_MODEL_DIR,
}

TRANSLATION_FRAMEWORKS = [
    "GoogleTranslator",
    "MyMemoryTranslator",
    "LibreTranslator",
]


def short_lang(lang_code: str) -> str:
    return lang_code.split("-")[0].lower()


def tokenize_simple(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def jaccard_similarity(a: str, b: str) -> float:
    sa = set(tokenize_simple(a))
    sb = set(tokenize_simple(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def classify_errors(
    candidate: str,
    reference: str,
    candidate_sentiment: dict = None,
    reference_sentiment: dict = None,
) -> List[Tuple[str, str]]:
    errors = []
    c = candidate.strip()
    r = reference.strip()

    if "  " in c or "??" in c or "!!" in c:
        errors.append(("grammatical", "Repeated spacing/punctuation suggests fluency issues."))
    if len(c.split()) > 8 and c[-1] not in ".!?":
        errors.append(("grammatical", "Long sentence without terminal punctuation."))
    if re.search(r"\b(\w+)\s+\1\b", c.lower()):
        errors.append(("grammatical", "Repeated adjacent words detected."))

    if r:
        len_gap = abs(len(c.split()) - len(r.split()))
        if (len_gap / max(1, len(r.split()))) > 0.35:
            errors.append(("contextual", "Length divergence suggests omitted or added context."))

    if r and jaccard_similarity(c, r) < 0.35:
        errors.append(("semantic", "Low lexical overlap with reference may indicate meaning drift."))

    if candidate_sentiment and reference_sentiment:
        c_label = candidate_sentiment.get("sentiment")
        r_label = reference_sentiment.get("sentiment")
        if c_label and r_label and c_label != r_label:
            errors.append(("semantic", f"Sentiment changed: {r_label} -> {c_label}."))

    if not errors:
        errors.append(("none", "No obvious issues detected by heuristic checks."))
    return errors


def load_translator_classes():
    try:
        from deep_translator import GoogleTranslator, LibreTranslator, MyMemoryTranslator

        return {
            "GoogleTranslator": GoogleTranslator,
            "MyMemoryTranslator": MyMemoryTranslator,
            "LibreTranslator": LibreTranslator,
        }, None
    except Exception as exc:
        return {}, str(exc)


def translate_text(framework: str, text: str, source_lang: str, target_lang: str = "en") -> Dict[str, str]:
    translators, err = load_translator_classes()
    if err:
        return {"ok": False, "error": f"deep-translator unavailable: {err}"}
    if framework not in translators:
        return {"ok": False, "error": f"Unknown framework: {framework}"}

    src_short = "auto" if source_lang == "auto" else short_lang(source_lang)
    tgt_short = short_lang(target_lang)
    t0 = time.time()

    try:
        if framework == "MyMemoryTranslator":
            src_code = "auto" if source_lang == "auto" else source_lang
            translator = translators[framework](source=src_code, target="en-GB")
        else:
            translator = translators[framework](source=src_short, target=tgt_short)
        translated = translator.translate(text[:1000])
        return {"ok": True, "translation": translated, "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def list_pngs(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".png")]
    return sorted(files)


def describe_plot(filename: str) -> str:
    name = filename.lower()
    if "training_curves" in name:
        return "Training/validation learning curves. Use this to check convergence and possible overfitting."
    if "embeddings_pca" in name:
        return "2D PCA projection of learned word embeddings. Closer points indicate more similar learned representations."
    if "confusion_matrix" in name:
        return "Confusion matrix (actual vs predicted classes). Strong diagonal values indicate better class-level performance."
    if "label_distribution" in name:
        return "Class distribution in the dataset. Helps explain imbalance and why weighting may be needed."
    return "Model output figure from training/evaluation."


@st.cache_resource
def load_service(model_dir: str) -> SentimentService:
    return SentimentService(model_dir=model_dir)


def render_result_block(result: dict):
    st.caption(f"Model type: {result.get('model_type', 'unknown')}")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Prediction", result.get("sentiment", "Unknown"))
    with c2:
        st.metric("Confidence", f"{float(result.get('confidence', 0.0)):.1%}")
    st.progress(float(result.get("score_scalar", 0.0)), text=f"Scalar score: {float(result.get('score_scalar', 0.0)):.3f}")

    if result.get("model_type") == "multitask":
        st.markdown("**Intent Output**")
        i1, i2 = st.columns(2)
        with i1:
            st.metric("Intent", result.get("intent", "Unknown"))
        with i2:
            st.metric("Intent Confidence", f"{float(result.get('intent_confidence', 0.0)):.1%}")
        intent_scores = result.get("intent_scores") or {}
        if intent_scores:
            st.write("Intent probabilities:")
            st.table(intent_scores)

    with st.expander("Details"):
        st.write("Cleaned:", result.get("cleaned", ""))
        st.write("Tokens:", result.get("tokens", []))
        st.write("Encoded IDs:", result.get("encoded", []))
        sentiment_scores = result.get("scores") or {}
        if sentiment_scores:
            st.write("Sentiment probabilities:")
            st.table(sentiment_scores)


st.title("Movie Review Analysis")
st.caption("Analyze movie sentiment, compare translations, and view saved model plots.")

with st.sidebar:
    st.header("Settings")
    selected_model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=2)
    selected_model_dir = MODEL_OPTIONS[selected_model_label]
    source_options = {"Auto Detect": "auto", **SUPPORTED_LANGUAGES}
    source_label = st.selectbox("Source language", list(source_options.keys()), index=0)
    source_lang = source_options[source_label]

service = load_service(selected_model_dir)

tab1, tab2, tab3 = st.tabs(["Translation Comparison", "Movie Analysis", "Model Results"])

with tab1:
    st.subheader("Translation Comparison (3 Frameworks)")
    st.caption("Translate to English using multiple frameworks and compare model outputs.")

    source_text = st.text_area(
        "Source Text",
        height=150,
        placeholder="Paste non-English text here...",
        key="src_text",
    )
    reference_text = st.text_area(
        "Reference English Translation (optional)",
        height=120,
        placeholder="Optional baseline to improve error classification...",
        key="ref_text",
    )

    if st.button("Run Comparison", type="primary"):
        if not source_text.strip():
            st.warning("Please enter source text.")
        else:
            outputs = []
            for fw in TRANSLATION_FRAMEWORKS:
                outputs.append((fw, translate_text(fw, source_text, source_lang=source_lang, target_lang="en")))
            st.session_state["framework_outputs"] = outputs

    outputs = st.session_state.get("framework_outputs", [])
    if outputs:
        baseline_translation = reference_text.strip()
        if not baseline_translation:
            for _, item in outputs:
                if item.get("ok"):
                    baseline_translation = item["translation"]
                    break
        baseline_sentiment = service.predict(baseline_translation) if baseline_translation else None

        for fw, item in outputs:
            st.markdown(f"### {fw}")
            if not item.get("ok"):
                st.error(item.get("error", "Unknown error"))
                continue

            translation = item["translation"]
            st.code(translation, language="text")
            st.caption(f"Latency: {item.get('latency_ms', 0)} ms")

            fw_result = service.predict(translation)
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Framework Output Result**")
                render_result_block(fw_result)
            with cols[1]:
                st.markdown("**Reference/Baseline Result**")
                if baseline_sentiment:
                    render_result_block(baseline_sentiment)
                else:
                    st.info("No baseline available.")

            st.markdown("**Error Classification (heuristic):**")
            for cat, reason in classify_errors(
                translation,
                baseline_translation,
                candidate_sentiment=fw_result,
                reference_sentiment=baseline_sentiment,
            ):
                st.write(f"- `{cat}`: {reason}")

with tab2:
    st.subheader("Movie Analysis")
    review = st.text_area(
        "Movie Review",
        height=140,
        placeholder="Type or paste a movie review in English...",
    )
    if st.button("Analyze Review", type="primary"):
        if not review.strip():
            st.warning("Please enter a review.")
        else:
            render_result_block(service.predict(review))

with tab3:
    st.subheader("Model Results")
    st.caption("Plots loaded from artifacts/plots generated by training scripts.")

    plot_groups = [
        ("Binary Model Plots", BINARY_PLOTS_DIR),
        ("Multilevel Model Plots", MULTILEVEL_PLOTS_DIR),
        ("Multitask Model Plots", MULTITASK_PLOTS_DIR),
    ]

    for title, folder in plot_groups:
        st.markdown(f"### {title}")
        paths = list_pngs(folder)
        if not paths:
            st.info(f"No plot images found in: {folder}")
            continue
        for img_path in paths:
            filename = os.path.basename(img_path)
            st.image(img_path, caption=filename, use_container_width=True)
            st.caption(describe_plot(filename))
