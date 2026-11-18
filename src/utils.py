from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset" / "registered_faces"
MODEL_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = DATASET_DIR / "registry.json"


def ensure_project_directories() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
