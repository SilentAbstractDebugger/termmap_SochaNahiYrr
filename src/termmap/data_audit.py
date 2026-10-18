"""Phase 0 data audit. Writes reports/data_audit.json (+ .md). Shareable: aggregate numbers only
(no proprietary row text) except the malformed-code list, which carries codes only."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .data import SYSTEMS, diagnosis_pairs, normalize_gold
from .normalize import basic_norm, strip_semantic_tag
from .terminology import KB, icd_nodot

EXPECTED = {  # facts from the task brief, section 2, to be verified
    "rows": 9646,
    "columns": ["entity", "chunk_text", "coding_system", "code", "display", "subconcept"],
    "diag_ICD10CM": 2305, "diag_SNOMEDCT_US": 2700, "diag_pairs": 2882, "pairs_both": 2123,
    "icd_distinct": 866, "snomed_distinct": 1669,
}


def _buckets(counts: Counter) -> dict:
    v = np.array(list(counts.values()))
    return {"distinct": int(len(v)), "codes_freq_1": int((v == 1).sum()),
            "codes_freq_2_5": int(((v >= 2) & (v <= 5)).sum()), "codes_freq_6p": int((v >= 6).sum()),
            "share_codes_singleton": round(float((v == 1).mean()), 4) if len(v) else 0,
            "share_rows_singleton_code": round(float(v[v == 1].sum() / v.sum()), 4) if len(v) else 0}


def run_audit(train: pd.DataFrame, kbs: dict[str, KB]) -> dict:
    a: dict = {}
    a["columns"] = list(train.columns)
    a["rows"] = int(len(train))
    a["dtypes"] = {c: str(t) for c, t in train.dtypes.items()}
    a["subconcept_counts"] = train["subconcept"].value_counts().to_dict()
    a["coding_system_counts"] = train["coding_system"].value_counts().to_dict()
    a["duplicate_full_rows"] = int(train.duplicated().sum())
    diag = train[train["subconcept"] == "DiagnosisCode"]
    a["diag_rows_by_system"] = diag["coding_system"].value_counts().to_dict()
    a["diag_duplicate_rows"] = int(diag.duplicated(["entity", "chunk_text", "coding_system"]).sum())
    pairs = diagnosis_pairs(train, kbs)
    hb = pairs["has_ICD10CM"] & pairs["has_SNOMEDCT_US"]
    a["diag_pairs"] = int(len(pairs))
    a["pairs_both_labels"] = int(hb.sum())
    a["pairs_icd_only"] = int((pairs["has_ICD10CM"] & ~pairs["has_SNOMEDCT_US"]).sum())
    a["pairs_snomed_only"] = int((~pairs["has_ICD10CM"] & pairs["has_SNOMEDCT_US"]).sum())
    ct = pairs["chunk_text"]
    a["empty_chunk_text_pairs"] = int((ct.str.strip() == "").sum())
    lens = ct.str.len()
    a["chunk_len"] = {q: float(lens.quantile(x)) for q, x in
                      (("min", 0), ("p25", .25), ("median", .5), ("p75", .75), ("p95", .95), ("max", 1))}
    a["chunk_has_header_markup"] = int(ct.str.contains("header_", regex=False).sum())
    a["chunk_has_br"] = int(ct.str.contains("<br>", regex=False).sum())
    a["distinct_entities"] = int(pairs["entity"].nunique())
    a["distinct_norm_entities"] = int(pairs["ent_norm"].nunique())
    a["entity_in_multiple_chunks"] = int((pairs.groupby("entity").size() > 1).sum())

    for s in SYSTEMS:
        kb = kbs[s]
        r = diag[diag["coding_system"] == s].copy()
        r["code_n"] = [normalize_gold(s, c) for c in r["code"]]
        sa: dict = {}
        cnt = Counter(r["code_n"])
        sa["buckets"] = _buckets(cnt)
        per_ent = r.groupby("entity")["code_n"].nunique()
        sa["entities_with_multiple_codes"] = int((per_ent > 1).sum())
        per_ent_n = r.assign(en=r["entity"].map(basic_norm)).groupby("en")["code_n"].nunique()
        sa["norm_entities_with_multiple_codes"] = int((per_ent_n > 1).sum())
        sa["duplicate_entity_strings"] = int(r["entity"].duplicated().sum())
        raw_in = r["code"].map(lambda c: c.strip() in kb.code_set)
        norm_in = r["code_n"].map(kb.valid)
        sa["gold_in_dict_exact"] = int(raw_in.sum())
        sa["gold_in_dict_normalized"] = int(norm_in.sum())
        sa["gold_not_in_dict"] = sorted(Counter(r.loc[~norm_in, "code"]).items())
        if s == "ICD10CM":
            valid = r.loc[norm_in, "code_n"]
            sa["gold_is_category_3char"] = int(valid.map(lambda c: len(icd_nodot(c)) == 3).sum())
            sa["gold_is_leaf"] = int(valid.map(lambda c: kb.is_leaf.get(c, False)).sum())
            sa["gold_non_leaf"] = int(valid.map(lambda c: not kb.is_leaf.get(c, True)).sum())
            sa["gold_is_range"] = int(r["code_n"].str.contains("-", regex=False).sum())
            sa["gold_depth_counts"] = Counter(valid.map(lambda c: len(icd_nodot(c)))).most_common()
            sa["gold_chapter_letter_counts"] = Counter(valid.str[0]).most_common()
            sa["dict_codes_with_multiple_desc"] = int(sum(len(v) > 1 for v in kb.descs.values()))
            sa["dict_range_headers"] = len(kb.range_headers)
        else:
            valid = r.loc[norm_in, "code_n"]
            sa["gold_semantic_tags"] = Counter(valid.map(lambda c: kb.sem_tag.get(c, "unknown"))).most_common()
            sa["dict_semantic_tags_top"] = Counter(kb.sem_tag.values()).most_common(25)
            sa["dict_concepts_with_multiple_desc"] = int(sum(len(v) > 1 for v in kb.descs.values()))
            sa["dict_concepts_without_fsn_tag"] = int(sum(1 for t in kb.sem_tag.values() if t == "unknown"))
        # display string present among dictionary descriptions of the gold code?
        disp_ok = 0
        disp_any = 0
        for c, dsp in zip(r["code_n"], r["display"]):
            if kb.valid(c):
                n = basic_norm(strip_semantic_tag(dsp)[0])
                disp_ok += n in kb.descs_norm[c]
            disp_any += basic_norm(strip_semantic_tag(dsp)[0]) in kb.norm_to_codes
        sa["display_matches_gold_code_desc"] = disp_ok
        sa["display_not_in_gold_code_desc"] = int(norm_in.sum() - disp_ok)
        sa["display_not_anywhere_in_dict"] = int(len(r) - disp_any)
        # entity exactly equals a dictionary description of the gold code (trivial cases)
        ex_gold = 0
        ex_any = 0
        for e, c in zip(r["entity"], r["code_n"]):
            n = basic_norm(e)
            ex_any += n in kb.norm_to_codes
            ex_gold += kb.valid(c) and n in kb.descs_norm[c]
        sa["entity_exact_desc_of_gold"] = ex_gold
        sa["entity_exact_desc_any_code"] = ex_any
        a[s] = sa
    # signs of evaluation split inside train: any id/split columns or flagged rows
    a["eval_split_signals"] = {
        "split_like_columns": [c for c in train.columns if c.lower() in ("split", "fold", "set", "is_test")],
        "id_like_columns": [c for c in train.columns if "id" in c.lower()],
    }
    a["discrepancies_vs_brief"] = _discrepancies(a)
    return a


def _discrepancies(a: dict) -> list[str]:
    out = []
    chk = [
        ("rows", a["rows"]), ("columns", a["columns"]),
        ("diag_ICD10CM", a["diag_rows_by_system"].get("ICD10CM")),
        ("diag_SNOMEDCT_US", a["diag_rows_by_system"].get("SNOMEDCT_US")),
        ("diag_pairs", a["diag_pairs"]), ("pairs_both", a["pairs_both_labels"]),
        ("icd_distinct", a["ICD10CM"]["buckets"]["distinct"]),
        ("snomed_distinct", a["SNOMEDCT_US"]["buckets"]["distinct"]),
    ]
    for k, v in chk:
        if EXPECTED[k] != v:
            out.append(f"{k}: brief={EXPECTED[k]} audit={v}")
    return out


def audit_markdown(a: dict) -> str:
    L = ["# Data audit (Phase 0)", "", "Aggregate statistics only; no proprietary row text.", ""]
    L.append(f"- Columns: `{a['columns']}`; rows: **{a['rows']}**; full duplicate rows: {a['duplicate_full_rows']}")
    L.append(f"- subconcept: {a['subconcept_counts']}")
    L.append(f"- Diagnosis rows by system: {a['diag_rows_by_system']}; duplicate (entity,chunk,system): {a['diag_duplicate_rows']}")
    L.append(f"- Diagnosis entity+chunk pairs: {a['diag_pairs']} (both labels {a['pairs_both_labels']}, "
             f"ICD only {a['pairs_icd_only']}, SNOMED only {a['pairs_snomed_only']})")
    L.append(f"- Empty chunk_text pairs: {a['empty_chunk_text_pairs']}; chunk length quantiles: {a['chunk_len']}")
    L.append(f"- Chunks with `header_` markup: {a['chunk_has_header_markup']}, with `<br>`: {a['chunk_has_br']}")
    L.append(f"- Distinct entities: {a['distinct_entities']} (normalized {a['distinct_norm_entities']}); "
             f"entities appearing in >1 chunk: {a['entity_in_multiple_chunks']}")
    for s in SYSTEMS:
        sa = a[s]
        L += ["", f"## {s}", ""]
        for k, v in sa.items():
            L.append(f"- {k}: {v}")
    L += ["", "## Discrepancies vs the brief (section 2)", ""]
    L += [f"- {d}" for d in a["discrepancies_vs_brief"]] or ["- none"]
    L += ["", f"Eval-split signals in train: {a['eval_split_signals']}"]
    return "\n".join(L) + "\n"
