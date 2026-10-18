"""Write outputs/synthetic_eval.csv from tests/hard_cases.yaml (public statement examples + labelled synthetic
sentences only) for demos and the one-command smoke run. outputs/ is git-ignored."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
hard = yaml.safe_load((ROOT / "tests" / "hard_cases.yaml").read_text())
(ROOT / "outputs").mkdir(exist_ok=True)
pd.DataFrame({"chunk_id": [h["id"] for h in hard], "clinical_entity": [h["entity"] for h in hard],
              "chunk_text": [h["chunk_text"] for h in hard]}).to_csv(ROOT / "outputs" / "synthetic_eval.csv", index=False)
print("wrote outputs/synthetic_eval.csv", len(hard), "rows")
