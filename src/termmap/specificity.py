"""Stage E (decision side): explicit ICD specificity profile and dictionary-safe decoding.

The learned choice between "stop at category", "stop at unspecified/other leaf" and "go
specific" is made by the reranker over hierarchy-expanded candidates (features.py adds the
ancestors, the siblings incl. ...9/...89/...0 leaves, and the children of top candidates).
This module (1) summarises, per query, where the reranker's probability mass sits in the
hierarchy (the "specificity profile", written to diagnostics), and (2) guarantees that any
emitted code is in the supplied dictionary, falling back up/down the chain if not.
"""
from __future__ import annotations

import pandas as pd

from .terminology import KB, icd_nodot


def specificity_profile(top_all: pd.DataFrame, kb: KB) -> dict:
    """top_all: candidates of ONE query with columns code, prob (all candidates, not just top-3)."""
    if not len(top_all):
        return {"p_category": 0.0, "p_unspecified_leaf": 0.0, "p_specific": 0.0}
    p_cat = p_uns = p_spec = 0.0
    for c, p in zip(top_all["code"], top_all["prob"]):
        nd = icd_nodot(c)
        desc = " ".join(kb.descs_norm.get(c, [])[:3])
        if len(nd) == 3:
            p_cat += p
        elif nd.endswith("9") or "unspecified" in desc:
            p_uns += p
        else:
            p_spec += p
    return {"p_category": round(p_cat, 4), "p_unspecified_leaf": round(p_uns, 4), "p_specific": round(p_spec, 4)}


def safe_decode(code: str, kb: KB) -> tuple[str, str]:
    """Return (valid code, rule). Never emits a code outside the dictionary."""
    if kb.valid(code):
        return code, "as_predicted"
    if kb.system != "ICD10CM":
        return "", "invalid"
    nd = icd_nodot(code)
    # walk up the chain
    for L in range(len(nd) - 1, 2, -1):
        cand = nd[:3] + ("." + nd[3:L] if L > 3 else "")
        if kb.valid(cand):
            return cand, "fallback_ancestor"
    # category exists only via descendants: pick the unspecified-looking descendant, else first
    cat = nd[:3]
    desc = sorted(c for c in kb.codes if icd_nodot(c).startswith(cat))
    if desc:
        uns = [c for c in desc if icd_nodot(c).endswith("9")]
        return (uns[0] if uns else desc[0]), "fallback_descendant"
    return "", "invalid"
