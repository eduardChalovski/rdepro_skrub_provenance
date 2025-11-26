from pathlib import Path

ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / "data"

SIMPLE_DATASET_PATH = DATA_DIR / "simple_dataset.csv"

__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "SIMPLE_DATASET_PATH",
]
