"""
============================================================================
AIT-204 Deep Learning | Topic 4: Natural Language Processing
ACTIVITY 4 — Part B: Frontend Web Application (Streamlit)
============================================================================
"""

import streamlit as st
from model_service import SentimentService


# =========================================================================
# PAGE CONFIGURATION
# =========================================================================
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)


# =========================================================================
# MODEL SELECTION (Sidebar)
# =========================================================================
MODEL_OPTIONS = {
    "Binary Sentiment (default) — saved_model": {
        "model_dir": "saved_model",
        "enabled": True,
    },
    "Multilevel Sentiment — saved_model_multilevel": {
        "model_dir": "saved_model_multilevel",
        "enabled": True,
    },
    "Multitask Model — saved_model_multitask [placeholder]": {
        "model_dir": "saved_model_multitask",
        "enabled": False,  # placeholder for next class
    },
}

st.sidebar.header("Model Settings")
selected_label = st.sidebar.selectbox(
    "Choose a trained model",
    options=list(MODEL_OPTIONS.keys()),
    index=0,
)

selected_model_dir = MODEL_OPTIONS[selected_label]["model_dir"]
selected_enabled = MODEL_OPTIONS[selected_label]["enabled"]

st.sidebar.caption("Selected checkpoint folder:")
st.sidebar.code(f"{selected_model_dir}/model.pt", language="text")

if not selected_enabled:
    st.sidebar.warning("Placeholder: this model will be added next class.")


# =========================================================================
# BACKEND INITIALIZATION (cached per model_dir)
# =========================================================================
@st.cache_resource
def load_service(model_dir: str) -> SentimentService:
    return SentimentService(model_dir=model_dir)

service = None if not selected_enabled else load_service(selected_model_dir)


# =========================================================================
# APP HEADER
# =========================================================================
st.title("Movie Review Sentiment Analyzer")
st.caption(
    "AIT-204 Deep Learning · Topic 4 · "
    "Built with PyTorch + Streamlit · Trained on movie reviews"
)
st.divider()

st.info(f"**Current model:** {selected_label}", icon="🧠")


# =========================================================================
# Tabs
# =========================================================================
tab1, tab2 = st.tabs(["Sentiment Analysis", "Translation Comparison"])


# =========================================================================
# Display helpers
# =========================================================================
def render_result(result: dict):
    model_type = result.get("model_type", "binary")

    # Primary KPI row
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", result.get("sentiment", "Unknown"))
    with col2:
        conf = result.get("confidence", 0.0)
        st.metric("Confidence", f"{conf:.1%}")

    # Scalar score always exists (0..1)
    score = float(result.get("score_scalar", 0.0))
    st.progress(score, text=f"Scalar score (0–1): {score:.3f}")

    # Model-specific details
    if model_type == "binary":
        st.caption("Binary sentiment: scalar score = P(Positive)")
    elif model_type == "multilevel":
        st.caption("Multilevel sentiment: scalar score = expected class (normalized)")
        scores = result.get("scores") or {}
        if scores:
            st.write("**Class probability distribution:**")
            # Table-ish display without pandas dependency
            st.table(scores)
    elif model_type == "multitask":
        st.caption("Multitask: sentiment + intent")
        st.write(f"**Intent:** {result.get('intent')}  ({float(result.get('intent_confidence') or 0.0):.1%})")
        intent_scores = result.get("intent_scores") or {}
        if intent_scores:
            st.write("**Intent probabilities:**")
            st.table(intent_scores)
    else:
        st.warning(f"Unknown model_type: {model_type}")

    with st.expander("Preprocessing Pipeline"):
        st.write("**Cleaned:**", result.get("cleaned", ""))
        st.write("**Tokens:**", result.get("tokens", []))
        st.write("**Encoded IDs:**", result.get("encoded", []))
        tokens = result.get("tokens", [])
        known = int(result.get("known_count", 0))
        st.caption(f"Vocabulary coverage: {known}/{len(tokens)} tokens known")


# =========================================================================
# TAB 1 — SENTIMENT ANALYSIS
# =========================================================================
with tab1:
    st.subheader("Analyze a Movie Review")
    st.caption(
        "Type or paste a movie review. "
        "The backend runs the NLP pipeline and returns a prediction."
    )

    review = st.text_area(
        "Movie Review",
        placeholder="Type or paste a movie review here...",
        height=120,
    )

    if st.button("Analyze", type="primary", disabled=not selected_enabled):
        if not review.strip():
            st.warning("Please enter a review before clicking Analyze.")
        else:
            result = service.predict(review)
            render_result(result)

    if not selected_enabled:
        st.caption("This tab is disabled for the multitask placeholder model.")


# =========================================================================
# TAB 2 — TRANSLATION COMPARISON
# =========================================================================
with tab2:
    st.subheader("Compare Original vs. Translated")
    st.caption(
        "Paste a review and its round-trip translation (English → other language → English). "
        "The model scores both and shows the prediction shift."
    )

    col1, col2 = st.columns(2)
    with col1:
        original = st.text_area("Original (English)", height=120)
    with col2:
        translated = st.text_area("Round-trip Translation", height=120)

    if st.button("Compare", type="primary", disabled=not selected_enabled):
        if not original.strip() or not translated.strip():
            st.warning("Please enter both texts before comparing.")
        else:
            result = service.compare(original, translated)

            left, right = st.columns(2)
            with left:
                st.markdown("### Original")
                render_result(result["original"])
            with right:
                st.markdown("### Translated")
                render_result(result["translated"])

            st.divider()

            if result["changed"]:
                st.warning(f"Prediction CHANGED (delta: {result['delta']:+.3f})")
            else:
                st.success(f"Prediction preserved (delta: {result['delta']:+.3f})")

            if result["lost_words"]:
                st.write("**Words lost in translation:**", result["lost_words"])
            if result["new_words"]:
                st.write("**New words from translation:**", result["new_words"])

    if not selected_enabled:
        st.caption("This tab is disabled for the multitask placeholder model.")


# =========================================================================
# FOOTER
# =========================================================================
st.divider()
st.caption(
    "AIT-204 Deep Learning · Topic 4 · Grand Canyon University"
)