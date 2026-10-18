"""Stage G: couple the two systems WITHOUT collapsing them.

Both systems are predicted independently. Afterwards we compute:
  * text agreement: char-TF-IDF cosine between the chosen ICD description and the chosen SNOMED
    description (both from the dictionaries);
  * per-system margin (top1 - top2 probability);
  * crosswalk compatibility (only if a crosswalk is supplied).
Low agreement is an explicit confidence feature / escalation flag. It never overwrites a code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .crosswalk import Crosswalk


def agreement(icd_codes, sno_codes, kbs, lex) -> np.ndarray:
    a = [kbs["ICD10CM"].descs_norm.get(c, [""])[0] or "_" for c in icd_codes]
    b = [kbs["SNOMEDCT_US"].descs_norm.get(c, [""])[0] or "_" for c in sno_codes]
    # use the SNOMED char vectoriser for both strings so they live in the same space
    va = lex["SNOMEDCT_US"].char_vec(a)
    vb = lex["SNOMEDCT_US"].char_vec(b)
    return np.asarray(va.multiply(vb).sum(axis=1)).ravel()


def margins(top: pd.DataFrame) -> pd.Series:
    if not len(top):
        return pd.Series(dtype=float)
    p1 = top[top["rank"] == 0].set_index("qid")["prob"]
    p2 = top[top["rank"] == 1].set_index("qid")["prob"].reindex(p1.index).fillna(0.0)
    return p1 - p2


def joint_signals(qids, icd_codes, sno_codes, top_icd, top_sno, kbs, lex,
                  xw: Crosswalk | None = None) -> pd.DataFrame:
    d = pd.DataFrame({"qid": list(qids), "icd": list(icd_codes), "sno": list(sno_codes)})
    d["xsys_text_agreement"] = agreement(d["icd"], d["sno"], kbs, lex).round(4)
    d["icd_margin"] = d["qid"].map(margins(top_icd)).fillna(0.0).round(4)
    d["sno_margin"] = d["qid"].map(margins(top_sno)).fillna(0.0).round(4)
    d["crosswalk_compat"] = [xw.compatible(s, i) if xw else np.nan for s, i in zip(d["sno"], d["icd"])]
    return d
