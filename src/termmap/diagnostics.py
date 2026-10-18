"""Per-row diagnostics (separate from the required prediction schema)."""
from __future__ import annotations

import pandas as pd

ABSTAIN_PROB = 0.30
LOW_AGREEMENT = 0.15
LOW_MARGIN = 0.10


def top3_string(top: pd.DataFrame) -> pd.Series:
    if not len(top):
        return pd.Series(dtype=str)
    t = top.sort_values(["qid", "rank"])
    return t.groupby("qid").apply(
        lambda g: "|".join(f"{c}:{p:.3f}" for c, p in zip(g["code"], g["prob"])), include_groups=False)


def build_diagnostics(q: pd.DataFrame, preds: dict, tops: dict, joint: pd.DataFrame, rules: dict,
                      mode: dict, low_support: dict) -> pd.DataFrame:
    d = pd.DataFrame({"row_id": q["row_id"].values, "qid": q["qid"].values, "entity": q["entity"].values})
    for s, short in (("ICD10CM", "icd"), ("SNOMEDCT_US", "snomed")):
        top = tops[s]
        p1 = top[top["rank"] == 0].set_index("qid")["prob"] if len(top) else pd.Series(dtype=float)
        d[f"{short}_code"] = d["qid"].map(preds[s])
        d[f"{short}_conf"] = d["qid"].map(p1).fillna(0.0).round(4)
        d[f"{short}_top3"] = d["qid"].map(top3_string(top)).fillna("")
        d[f"{short}_decode_rule"] = d["qid"].map(rules[s]).fillna("")
        d[f"{short}_low_support"] = d["qid"].map(low_support[s]).fillna(True)
    j = joint.set_index("qid")
    for c in ("xsys_text_agreement", "icd_margin", "sno_margin", "crosswalk_compat"):
        d[c] = d["qid"].map(j[c])
    d["abstain_flag"] = ((d["icd_conf"] < ABSTAIN_PROB) | (d["snomed_conf"] < ABSTAIN_PROB) |
                         (d["xsys_text_agreement"] < LOW_AGREEMENT))
    # Step 7 routing: lexical-only evidence, small margin, or ICD/SNOMED disagreement -> review
    d["route_for_review"] = (d["icd_low_support"].astype(bool) | d["snomed_low_support"].astype(bool) |
                             (d["icd_margin"] < LOW_MARGIN) | (d["sno_margin"] < LOW_MARGIN) |
                             (d["xsys_text_agreement"] < LOW_AGREEMENT))
    d["mode"] = ";".join(f"{k}={v}" for k, v in sorted(mode.items()))
    return d.drop(columns=["qid"])
