"""Write predictions back in the SAME structure as the input, plus the wide file."""
from __future__ import annotations

import pandas as pd

from .input_schema import Schema, canon_system
from .output_validation import WIDE_COLUMNS


def wide_frame(q: pd.DataFrame, preds: dict) -> pd.DataFrame:
    return pd.DataFrame({
        WIDE_COLUMNS[0]: q["row_id"].values,
        WIDE_COLUMNS[1]: q["entity_raw"].values,
        WIDE_COLUMNS[2]: q["qid"].map(preds["ICD10CM"]).values,
        WIDE_COLUMNS[3]: q["qid"].map(preds["SNOMEDCT_US"]).values,
    })


def long_frame(inp: pd.DataFrame, q: pd.DataFrame, schema: Schema, preds: dict, kbs: dict) -> pd.DataFrame:
    out = inp.copy()
    systems = [canon_system(s) for s in inp[schema.system_col]]
    codes, disp = [], []
    for qid, s in zip(q["qid"], systems):
        c = preds[s].get(qid, "") if s in preds else ""
        codes.append(c)
        disp.append(kbs[s].preferred.get(c, "") if s in kbs and c else "")
    out[schema.code_col] = codes
    if "display" in out.columns:
        out["display"] = disp
    return out
