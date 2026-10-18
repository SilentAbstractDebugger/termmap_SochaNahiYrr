"""Thin wrapper: python scripts/predict.py --input X --output Y [--config configs/evaluation.yaml]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from termmap.predict import main  # noqa: E402

if __name__ == "__main__":
    main()
