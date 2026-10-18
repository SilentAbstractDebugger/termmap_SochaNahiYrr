"""Training data loading and per-system supervised tables (long format -> masks)."""
from __future__ import annotations

import pandas as pd

from .normalize import basic_norm
from .terminology import KB, norm_icd, norm_snomed
from .util import resolve

SYSTEMS = ("ICD10CM", "SNOMEDCT_US")


def load_train(cfg: dict, path: str | None = None) -> pd.DataFrame:
    p = resolve(path or cfg["data"]["train_path"])
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    return df.fillna("").astype(str)


def normalize_gold(system: str, code: str) -> str:
    return norm_icd(code) if system == "ICD10CM" else norm_snomed(code)


def diagnosis_pairs(df: pd.DataFrame, kbs: dict[str, KB], subconcept: str = "DiagnosisCode") -> pd.DataFrame:
    """One row per (entity, chunk_text) pair with per-system gold + masks.

    Returns columns: pair_id, entity, chunk_text, ent_norm, gold_ICD10CM, gold_SNOMEDCT_US,
    has_ICD10CM, has_SNOMEDCT_US, valid_ICD10CM, valid_SNOMEDCT_US.
    """
    d = df[df["subconcept"] == subconcept].copy()
    d["code_n"] = [normalize_gold(s, c) for s, c in zip(d["coding_system"], d["code"])]
    keys = d[["entity", "chunk_text"]].drop_duplicates(keep="first").reset_index(drop=True)
    keys["pair_id"] = [f"p{i:05d}" for i in range(len(keys))]
    d = d.merge(keys, on=["entity", "chunk_text"], how="left")
    out = keys.copy()
    out["ent_norm"] = out["entity"].map(basic_norm)
    for s in SYSTEMS:
        sub = d[d["coding_system"] == s].drop_duplicates("pair_id", keep="first")
        m = dict(zip(sub["pair_id"], sub["code_n"]))
        out[f"gold_{s}"] = out["pair_id"].map(m).fillna("")
        out[f"has_{s}"] = out[f"gold_{s}"] != ""
        out[f"valid_{s}"] = [bool(g) and kbs[s].valid(g) for g in out[f"gold_{s}"]] if kbs else False
    return out


def system_rows(pairs: pd.DataFrame, system: str) -> pd.DataFrame:
    """Rows with a label for `system` (label mask applied)."""
    r = pairs[pairs[f"has_{system}"]].copy()
    r["gold"] = r[f"gold_{system}"]
    r["gold_valid"] = r[f"valid_{system}"]
    return r[["pair_id", "entity", "chunk_text", "ent_norm", "gold", "gold_valid"]].reset_index(drop=True)
