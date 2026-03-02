# Topic4_NLP

PyTorch NLP project for:
- binary sentiment classification,
- multilevel sentiment classification (7 classes),
- multitask sentiment + intent classification,
- Streamlit app + backend service for interactive inference and translation comparison.

## Current Project Layout

```text
Topic4_NLP/
|- activity1_preprocessing.py
|- activity2_model.py
|- activity3_train.py
|- activity_part1_multilevel.py
|- activity_part2_intent.py
|- model_service.py
|- activity4_app.py
|- project_paths.py
|- requirements.txt
|- artifacts/
|  |- models/
|  |  |- binary/
|  |  |- multilevel/
|  |  `- multitask/
|  `- plots/
|     |- binary/
|     |- multilevel/
|     `- multitask/
`- README.md
```

## What Each Script Does

- `activity1_preprocessing.py`:
  - text cleaning, tokenization, vocabulary, encoding/padding utilities.
- `activity2_model.py`:
  - binary sentiment model architecture and save/load helpers.
- `activity3_train.py`:
  - trains binary model and saves artifacts to `artifacts/models/binary/`.
- `activity_part1_multilevel.py`:
  - trains 7-class sentiment model and saves artifacts to `artifacts/models/multilevel/`.
- `activity_part2_intent.py`:
  - trains dual-head multitask model (sentiment + intent) and saves artifacts to `artifacts/models/multitask/`.
- `model_service.py`:
  - unified inference backend for all model types.
  - supports both new `artifacts/models/*` and legacy `saved_model*` directories.
- `activity4_app.py`:
  - Streamlit frontend with:
    - sentiment analysis tab,
    - translation comparison tab,
    - model switcher (binary, multilevel, multitask).
- `project_paths.py`:
  - centralized path constants for model/plot output directories.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train Models

Run any/all of these:

```bash
python activity3_train.py
python activity_part1_multilevel.py
python activity_part2_intent.py
```

Generated outputs are written under:
- `artifacts/models/binary/`
- `artifacts/models/multilevel/`
- `artifacts/models/multitask/`
- `artifacts/plots/binary/`
- `artifacts/plots/multilevel/`
- `artifacts/plots/multitask/`

## Run Backend Self-Test

```bash
python model_service.py
```

This verifies model loading + prediction paths for available checkpoints.

## Run Streamlit App

```bash
streamlit run activity4_app.py
```

In the sidebar, choose one of:
- `Binary Sentiment (default)`
- `Multilevel Sentiment`
- `Multitask (Sentiment + Intent)`

## Notes on Artifacts and Compatibility

- Canonical artifact location is `artifacts/`.
- Legacy folders (`saved_model`, `saved_model_multilevel`, `saved_model_multitask`) are still recognized by `model_service.py` if present.
- `.gitignore` is configured to ignore generated artifacts and cache files.

## Dependencies

From `requirements.txt`:
- `torch>=2.0.0`
- `numpy>=1.24.0`
- `matplotlib>=3.7.0`
- `scikit-learn>=1.2.0`
- `streamlit>=1.28.0`
- `deep-translator>=1.11.0`
