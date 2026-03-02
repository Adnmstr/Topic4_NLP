import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")

BINARY_MODEL_DIR = os.path.join(MODELS_DIR, "binary")
MULTILEVEL_MODEL_DIR = os.path.join(MODELS_DIR, "multilevel")
MULTITASK_MODEL_DIR = os.path.join(MODELS_DIR, "multitask")

BINARY_PLOTS_DIR = os.path.join(PLOTS_DIR, "binary")
MULTILEVEL_PLOTS_DIR = os.path.join(PLOTS_DIR, "multilevel")
MULTITASK_PLOTS_DIR = os.path.join(PLOTS_DIR, "multitask")

LEGACY_BINARY_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
LEGACY_MULTILEVEL_MODEL_DIR = os.path.join(BASE_DIR, "saved_model_multilevel")
LEGACY_MULTITASK_MODEL_DIR = os.path.join(BASE_DIR, "saved_model_multitask")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
