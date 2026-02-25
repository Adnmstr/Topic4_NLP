"""
============================================================================
AIT-204 Deep Learning | Topic 4: Natural Language Processing
ACTIVITY 4 — Part B: Frontend Web Application (Streamlit)
============================================================================

FRONTEND / BACKEND ARCHITECTURE
──────────────────────────────────────────────────────────────────────────
This file is the FRONTEND. It contains zero model logic.

    ┌──────────────────────────────────────────────────────────────────┐
    │  FRONTEND  (THIS FILE — Streamlit)                               │
    │    Creates the web page.  Handles user input.                    │
    │    Calls service.predict() / service.compare().                  │
    │    Formats the returned dicts as visual components.              │
    ├──────────────────────────────────────────────────────────────────┤
    │  BACKEND   (model_service.py)                                    │
    │    Loads model. Runs PyTorch inference. Returns dicts.           │
    │    Zero UI code. Independently testable.                         │
    └──────────────────────────────────────────────────────────────────┘

This means you can:
  - Swap Streamlit for React/Next.js — model_service.py stays unchanged.
  - Change the model architecture    — activity4_app.py stays unchanged.
  - Test the backend (python model_service.py) before building any UI.
──────────────────────────────────────────────────────────────────────────

STREAMLIT QUICK REFERENCE
──────────────────────────────────────────────────────────────────────────
Layout:
    st.title("text")             Main heading
    st.subheader("text")         Section heading
    st.caption("text")           Small gray caption
    st.divider()                 Horizontal rule
    col1, col2 = st.columns(2)   Side-by-side columns
    with st.expander("title"):   Collapsible section
    tab1, tab2 = st.tabs([...])  Tab navigation

Input:
    st.text_area(label, ...)     Multi-line text input  → returns str
    st.button(label, type=...)   Clickable button       → returns bool

Output:
    st.metric(label, value)      Big KPI card
    st.progress(value, text)     Horizontal bar 0.0–1.0
    st.success("msg")            Green status box
    st.warning("msg")            Yellow status box
    st.error("msg")              Red status box
    st.write(label, value)       Generic display
    st.table(dict_or_df)         Simple table

Caching:
    @st.cache_resource           Cache return value across all users/sessions.
                                 Essential for model loading — runs once only.
──────────────────────────────────────────────────────────────────────────

DEPLOYMENT (Streamlit Community Cloud — free, 3 steps)
──────────────────────────────────────────────────────────────────────────
1. Push your project to a public GitHub repository.
2. Go to https://share.streamlit.io → "New app"
3. Connect repo, set main file to activity4_app.py → Deploy.
   A live public URL is generated in ~2 minutes.

Files that must be in the repo:
    activity4_app.py            ← this file (the app entry point)
    model_service.py            ← backend service
    activity1_preprocessing.py  ← imported by model_service.py
    activity2_model.py          ← imported by model_service.py
    saved_model/model.pt
    saved_model/vocab.json
    saved_model/config.json
    requirements.txt

ALTERNATIVE DEPLOYMENT:
    Hugging Face Spaces — select "Streamlit" as the SDK.
    Same files, same process. Identical result.
──────────────────────────────────────────────────────────────────────────

WHAT YOU WILL IMPLEMENT (TODOs 5–7):
    TODO 5: Initialize the backend service with @st.cache_resource
    TODO 6: Build the Sentiment Analysis tab (input → predict → display)
    TODO 7: Build the Translation Comparison tab (two inputs → compare → display)

RUN THIS FILE:  streamlit run activity4_app.py
    Opens in your browser at http://localhost:8501
============================================================================
"""

import streamlit as st
from model_service import SentimentService


# =========================================================================
# PAGE CONFIGURATION
# Must be the first Streamlit call in the script.
# =========================================================================
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)


# =========================================================================
# STEP 1: INITIALIZE THE BACKEND SERVICE
# =========================================================================
# @st.cache_resource runs the decorated function ONCE and caches the result
# for the lifetime of the app — across all users and all reruns.
# Without it, Streamlit would reload the model on every button click.
# =========================================================================

# ── TODO 5 ────────────────────────────────────────────────────────────────
# Complete the load_service() function so it returns a SentimentService.
#
# HINT: Return SentimentService() — that's the backend class from
#       model_service.py. The @st.cache_resource decorator handles caching.
#
# After completing this TODO, the frontend can call:
#   result = service.predict("some review")
#   result = service.compare("original", "translated")
# ──────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_service():
    return SentimentService()

service = load_service()


# =========================================================================
# APP HEADER
# =========================================================================
st.title("Movie Review Sentiment Analyzer")
st.caption(
    "AIT-204 Deep Learning · Topic 4 · "
    "Built with PyTorch + Streamlit · Trained on movie reviews"
)
st.divider()


# =========================================================================
# STEP 2: TAB NAVIGATION (frontend routing between features)
# =========================================================================
tab1, tab2 = st.tabs(["Sentiment Analysis", "Translation Comparison"])


# =========================================================================
# TAB 1 — SENTIMENT ANALYSIS
# =========================================================================
with tab1:
    st.subheader("Analyze a Movie Review")
    st.caption(
        "Type or paste a movie review. "
        "The backend runs the full NLP pipeline and returns a prediction."
    )

    # ── TODO 6 ────────────────────────────────────────────────────────────
    # Build the Sentiment Analysis tab.
    #
    # Required steps (you choose the exact Streamlit components):
    #
    # 1. Text input — let the user type a review:
    #      review = st.text_area("Movie Review",
    #                            placeholder="Type a movie review here...",
    #                            height=120)
    #
    # 2. Submit button:
    #      if st.button("Analyze Sentiment", type="primary"):
    #
    # 3. Call the BACKEND inside the button block:
    #      result = service.predict(review)
    #    result is a dict with keys: sentiment, confidence, positive_score,
    #    negative_score, cleaned, tokens, encoded, known_count
    #
    # 4. Display the result. Suggested layout (improve it if you like):
    #
    #      col1, col2 = st.columns(2)
    #      with col1:
    #          st.metric("Sentiment", result["sentiment"])
    #      with col2:
    #          st.metric("Confidence", f"{result['confidence']:.1%}")
    #
    #      st.progress(result["positive_score"],
    #                  text=f"Positive score: {result['positive_score']:.3f}")
    #
    #      with st.expander("Preprocessing Pipeline"):
    #          st.write("**Cleaned:**",     result["cleaned"])
    #          st.write("**Tokens:**",      result["tokens"])
    #          st.write("**Encoded IDs:**", result["encoded"])
    #          st.caption(
    #              f"Vocabulary coverage: "
    #              f"{result['known_count']}/{len(result['tokens'])} tokens known"
    #          )
    #
    # FRONTEND DESIGN NOTE:
    #   This is YOUR UI — feel free to redesign it. What matters is that
    #   you call service.predict() and display its return values.
    #   The backend dict is your contract; the UI is your creative space.
    # ──────────────────────────────────────────────────────────────────────

    review = st.text_area(
        "Movie Review",
        placeholder="Type or paste a movie review here...",
        height=120,
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter a review before clicking Analyze.")
        else:
            result = service.predict(review)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment", result["sentiment"])
            with col2:
                st.metric("Confidence", f"{result['confidence']:.1%}")

            st.progress(
                result["positive_score"],
                text=f"Positive score: {result['positive_score']:.3f}",
            )

            with st.expander("Preprocessing Pipeline"):
                st.write("**Cleaned:**", result["cleaned"])
                st.write("**Tokens:**", result["tokens"])
                st.write("**Encoded IDs:**", result["encoded"])
                st.caption(
                    f"Vocabulary coverage: "
                    f"{result['known_count']}/{len(result['tokens'])} tokens known"
                )


# =========================================================================
# TAB 2 — TRANSLATION COMPARISON
# =========================================================================
with tab2:
    st.subheader("Compare Original vs. Translated")
    st.caption(
        "Paste a review and its round-trip translation "
        "(English → other language → English back). "
        "The model scores both and shows the sentiment shift."
    )

    # ── TODO 7 ────────────────────────────────────────────────────────────
    # Build the Translation Comparison tab.
    #
    # Required steps:
    #
    # 1. Two side-by-side text inputs:
    #      col1, col2 = st.columns(2)
    #      with col1:
    #          original   = st.text_area("Original (English)", height=120)
    #      with col2:
    #          translated = st.text_area("Round-trip Translation", height=120)
    #
    # 2. Compare button:
    #      if st.button("Compare Sentiments", type="primary"):
    #
    # 3. Call the BACKEND:
    #      result = service.compare(original, translated)
    #    result is a dict with keys: original (dict), translated (dict),
    #    delta (float), changed (bool), lost_words (list), new_words (list)
    #
    # 4. Display the comparison. Suggested layout:
    #
    #      col1, col2 = st.columns(2)
    #      with col1:
    #          st.metric("Original",   result["original"]["sentiment"],
    #                    f'{result["original"]["positive_score"]:.3f}')
    #      with col2:
    #          st.metric("Translated", result["translated"]["sentiment"],
    #                    f'{result["translated"]["positive_score"]:.3f}')
    #
    #      if result["changed"]:
    #          st.warning(f"Sentiment CHANGED  (delta: {result['delta']:+.3f})")
    #      else:
    #          st.success(f"Sentiment preserved (delta: {result['delta']:+.3f})")
    #
    #      if result["lost_words"]:
    #          st.write("**Words lost in translation:**", result["lost_words"])
    #      if result["new_words"]:
    #          st.write("**New words from translation:**", result["new_words"])
    #
    # FRONTEND DESIGN NOTE:
    #   Again, this is your UI. Experiment with different components.
    #   A table view, color-coded score bars, or a diff display
    #   would all be valid (and impressive) design choices.
    # ──────────────────────────────────────────────────────────────────────

    col1, col2 = st.columns(2)
    with col1:
        original   = st.text_area("Original (English)", height=120)
    with col2:
        translated = st.text_area("Round-trip Translation", height=120)

    if st.button("Compare Sentiments", type="primary"):
        if not original.strip() or not translated.strip():
            st.warning("Please enter both texts before comparing.")
        else:
            result = service.compare(original, translated)
            score_col1, score_col2 = st.columns(2)
            with score_col1:
                st.metric(
                    "Original",
                    result["original"]["sentiment"],
                    f'{result["original"]["positive_score"]:.3f}',
                )
            with score_col2:
                st.metric(
                    "Translated",
                    result["translated"]["sentiment"],
                    f'{result["translated"]["positive_score"]:.3f}',
                )

            if result["changed"]:
                st.warning(f"Sentiment CHANGED (delta: {result['delta']:+.3f})")
            else:
                st.success(f"Sentiment preserved (delta: {result['delta']:+.3f})")

            if result["lost_words"]:
                st.write("**Words lost in translation:**", result["lost_words"])
            if result["new_words"]:
                st.write("**New words from translation:**", result["new_words"])


# =========================================================================
# FOOTER
# =========================================================================
st.divider()
st.caption(
    "AIT-204 Deep Learning · Topic 4 · Grand Canyon University  |  "
    "Architecture: Embedding → AvgPool → FC → Sigmoid"
)


# =========================================================================
# REFLECTION QUESTIONS (answer in your written report)
# =========================================================================
#
# 1. ARCHITECTURE: You imported SentimentService from model_service.py and
#    called service.predict() from Streamlit code. Why is keeping frontend
#    and backend in separate files considered good software engineering?
#    What would break first if both layers were in a single file as the app grew?
#
# 2. CACHING: Why is @st.cache_resource important here? What would happen
#    to app performance and user experience if the model reloaded on every
#    button click?
#
# 3. FASTAPI EXTENSION: How would you modify model_service.py to expose
#    predict() and compare() as HTTP endpoints? Write pseudocode for what
#    the Streamlit frontend would look like if it called those endpoints
#    via requests.post() instead of importing the service directly.
#
# 4. CROSS-LINGUAL ANALYSIS: Use the Translation Comparison tab to test
#    10+ examples with Google Translate or DeepL. Classify each outcome:
#    grammatical error / semantic shift / contextual loss / preserved.
#    Document all examples in your assignment report.
#
# 5. ETHICAL: Your model was trained on English movie reviews. How might
#    this affect predictions for reviews written by non-native English
#    speakers? What tests would you run before deploying this globally?
#
