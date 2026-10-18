"""Stage J: output validation. Raises on any violation (fail loudly, never ship bad output)."""
from __future__ import annotations

import pandas as pd

from .terminology import KB

WIDE_COLUMNS = ["chunk_id", "clinical_entity", "icd10cm_code", "snomed_code"]


class OutputValidationError(AssertionError):
    pass


def validate_wide(out: pd.DataFrame, n_input: int, kbs: dict[str, KB]) -> dict:
    errs = []
    if list(out.columns) != WIDE_COLUMNS:
        errs.append(f"columns {list(out.columns)} != {WIDE_COLUMNS}")
    if len(out) != n_input:
        errs.append(f"row count {len(out)} != input {n_input}")
    if out.isna().any().any():
        errs.append("NaN present")
    for col, s in (("icd10cm_code", "ICD10CM"), ("snomed_code", "SNOMEDCT_US")):
        if col in out:
            bad = out[~out[col].map(kbs[s].valid)]
            if len(bad):
                errs.append(f"{len(bad)} {col} values not in dictionary, e.g. {bad[col].head(3).tolist()}")
            if (out[col] == "").any():
                errs.append(f"empty {col}")
    if errs:
        raise OutputValidationError("; ".join(errs))
    return {"rows": len(out), "valid": True}


def validate_long(out: pd.DataFrame, inp: pd.DataFrame, code_col: str, systems: list[str],
                  kbs: dict[str, KB]) -> dict:
    """Long layout: same rows, same column order, every in-scope row's code in its dictionary."""
    errs = []
    if len(out) != len(inp):
        errs.append("row count changed")
    if list(out.columns[: len(inp.columns)]) != list(inp.columns):
        errs.append("input column order not preserved")
    unknown = 0
    for code, s in zip(out[code_col], systems):
        if s in kbs:
            if not kbs[s].valid(code):
                errs.append(f"{s} code {code!r} not in dictionary")
                break
        else:
            unknown += 1
    if errs:
        raise OutputValidationError("; ".join(errs))
    return {"rows": len(out), "valid": True, "rows_with_unknown_coding_system": unknown}
