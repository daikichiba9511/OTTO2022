from pathlib import Path

ROOT = Path(__file__).parents[1]
INPUT_DIR = ROOT / "input"
OUT_DIR = ROOT / "output"
TYPE_LABELS = {"clicks": 0, "carts": 1, "orders": 2}
