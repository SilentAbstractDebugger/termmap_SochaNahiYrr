"""Stage A: input adapter. Auto-detects the evaluation schema.

Supported layouts (assumption documented in README "Open questions"):
  * train-style LONG: entity, chunk_text, coding_system[, code, display, subconcept]
      -> one output row per input row; the `code` column is filled for that row's system.
  * statement-style WIDE: chunk_id, clinical_entity[, chunk_text]
      -> one output row per input row with icd10cm_code and snomed_code filled.
Both layouts also produce a wide file (chunk_id, clinical_entity, icd10cm_code, snomed_code).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ENTITY_COLS = ("entity", "clinical_entity", "term", "mention", "entity_text")
CONTEXT_COLS = ("chunk_text", "context", "chunk", "text", "sentence")
ID_COLS = ("chunk_id", "id", "row_id", "uid")
SYSTEM_COLS = ("coding_system", "system", "vocabulary")
SUBCONCEPT_COLS = ("subconcept", "entity_type", "type")
SYSTEM_ALIASES = {
    "icd10cm": "ICD10CM", "icd-10-cm": "ICD10CM", "icd10": "ICD10CM", "icd": "ICD10CM",
    "snomedct_us": "SNOMEDCT_US", "snomed": "SNOMEDCT_US", "snomedct": "SNOMEDCT_US",
    "snomed ct": "SNOMEDCT_US", "sct": "SNOMEDCT_US",
}


@dataclass
class Schema:
    layout: str              # "long" | "wide"
    entity_col: str
    context_col: str | None
    id_col: str | None
    system_col: str | None
    subconcept_col: str | None
    code_col: str | None     # long layout: column to fill


def _find(cols, options):
    low = {c.lower().strip(): c for c in cols}
    for o in options:
        if o in low:
            return low[o]
    return None


def detect_schema(df: pd.DataFrame) -> Schema:
    cols = list(df.columns)
    ent = _find(cols, ENTITY_COLS)
    if ent is None:
        raise ValueError(f"No entity column found among {cols}; expected one of {ENTITY_COLS}")
    ctx = _find(cols, CONTEXT_COLS)
    idc = _find(cols, ID_COLS)
    sysc = _find(cols, SYSTEM_COLS)
    sub = _find(cols, SUBCONCEPT_COLS)
    if sysc is not None:
        code = _find(cols, ("code",)) or "code"
        return Schema("long", ent, ctx, idc, sysc, sub, code)
    return Schema("wide", ent, ctx, idc, None, sub, None)


def read_input(path: str) -> pd.DataFrame:
    if str(path).endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, dtype=str)
    else:
        sep = "\t" if str(path).endswith(".tsv") else ","
        df = pd.read_csv(path, dtype=str, keep_default_na=False, sep=sep)
    return df.fillna("").astype(str)


def canon_system(v: str) -> str:
    v0 = str(v).strip()
    return SYSTEM_ALIASES.get(v0.lower(), v0)


def to_queries(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """One row per input row: qid (row position), entity, chunk_text, systems needed."""
    q = pd.DataFrame({
        "qid": range(len(df)),
        "entity": df[schema.entity_col].astype(str).values,
        "chunk_text": df[schema.context_col].astype(str).values if schema.context_col else "",
    })
    q["chunk_text"] = q["chunk_text"].replace({"nan": "", "None": ""})
    if schema.layout == "long":
        q["systems"] = [[canon_system(s)] for s in df[schema.system_col]]
    else:
        q["systems"] = [["ICD10CM", "SNOMEDCT_US"]] * len(df)
    q["row_id"] = (df[schema.id_col].astype(str).values if schema.id_col
                   else [f"row_{i}" for i in range(len(df))])
    return q
