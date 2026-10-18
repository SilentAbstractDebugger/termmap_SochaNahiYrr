"""Metrics aligned to the judging table. All numbers are computed, never estimated."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .terminology import KB, icd_nodot

KS = (1, 5, 10, 20, 50, 100, 200)


def bucket(n: int) -> str:
    return "0" if n == 0 else "1" if n == 1 else "2-5" if n <= 5 else "6+"


def predictions_table(rows: pd.DataFrame, top: pd.DataFrame, fallback: str | None) -> pd.DataFrame:
    t1 = top[top["rank"] == 0].set_index("qid") if len(top) else pd.DataFrame(columns=["code", "prob"])
    d = rows.copy()
    d["pred"] = d["pair_id"].map(t1["code"]) if len(t1) else np.nan
    d["prob"] = d["pair_id"].map(t1["prob"]) if len(t1) else 0.0
    d["fallback"] = d["pred"].isna()
    d["pred"] = d["pred"].fillna(fallback or "")
    d["prob"] = d["prob"].fillna(0.0)
    d["correct"] = (d["pred"] == d["gold"]).astype(int)
    return d


def _acc(x: pd.Series) -> dict:
    return {"n": int(len(x)), "acc": round(float(x.mean()), 4) if len(x) else None}


def ece(prob: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    if len(prob) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (prob > edges[i]) & (prob <= edges[i + 1]) if i else (prob >= 0) & (prob <= edges[1])
        if m.any():
            e += m.mean() * abs(prob[m].mean() - correct[m].mean())
    return round(float(e), 4)


def recall_at_k(feats: pd.DataFrame, rows: pd.DataFrame, order_col: str, ascending: bool) -> dict:
    """Fraction of eval rows whose gold appears in the top-K of a candidate ordering."""
    n = len(rows)
    if n == 0 or not len(feats):
        return {f"@{k}": None for k in KS}
    d = feats[["qid", "code", order_col, "label"]].sort_values(
        ["qid", order_col, "code"], ascending=[True, ascending, True], kind="mergesort")
    if order_col.endswith("_rank_orig"):
        d = d[d[order_col] < 99]
    d["r"] = d.groupby("qid").cumcount()
    pos = d[d["label"] == 1].groupby("qid")["r"].min()
    return {f"@{k}": round(float((pos < k).sum() / n), 4) for k in KS}


def system_metrics(system: str, kb: KB, rows: pd.DataFrame, top: pd.DataFrame, feats: pd.DataFrame,
                   train_freq: Counter, fallback: str | None, vocab=None) -> tuple[dict, pd.DataFrame]:
    d = predictions_table(rows, top, fallback)
    d["train_freq"] = d["gold"].map(lambda c: train_freq.get(c, 0))
    d["bucket"] = d["train_freq"].map(bucket)
    m: dict = {"overall": _acc(d["correct"]), "fallback_rows": int(d["fallback"].sum()),
               "gold_invalid_rows": int((~d["gold_valid"]).sum())}
    m["by_bucket"] = {b: _acc(g["correct"]) for b, g in d.groupby("bucket")}
    m["unseen_code"] = _acc(d.loc[d["train_freq"] == 0, "correct"])
    in_pool = feats.groupby("qid")["label"].max() if len(feats) else pd.Series(dtype=float)
    d["in_pool"] = d["pair_id"].map(in_pool).fillna(0).astype(int)
    err = d[d["correct"] == 0]
    m["errors"] = {"total": int(len(err)), "retrieval_miss": int((err["in_pool"] == 0).sum()),
                   "rank_miss": int((err["in_pool"] == 1).sum())}
    m["pool_recall"] = round(float(d["in_pool"].mean()), 4) if len(d) else None
    if len(feats):
        f = feats.copy()
        sc = top  # reranker order recall from top-3 only; full order computed via score column if present
        m["recall_first_stage"] = recall_at_k(f, rows, "s0", False)
        m["recall_char_channel"] = recall_at_k(f, rows, "char_rank_orig", True)
        m["recall_bm25_channel"] = recall_at_k(f, rows, "bm25_rank_orig", True)
        if "rerank_score" in f:
            m["recall_reranked"] = recall_at_k(f, rows, "rerank_score", False)
    # ICD partial credit + over/under coding
    if system == "ICD10CM":
        g = d["gold"].map(icd_nodot)
        p = d["pred"].map(icd_nodot)
        m["same_category"] = round(float((g.str[:3] == p.str[:3]).mean()), 4)
        m["same_letter"] = round(float((g.str[:1] == p.str[:1]).mean()), 4)
        wrong = d["correct"] == 0
        over = wrong & np.array([pp.startswith(gg) and len(pp) > len(gg) for gg, pp in zip(g, p)], dtype=bool)
        under = wrong & np.array([gg.startswith(pp) and len(gg) > len(pp) for gg, pp in zip(g, p)], dtype=bool)
        sib = wrong & (g.str[:3] == p.str[:3]) & ~over & ~under
        m["over_coding_rate"] = round(float(over.mean()), 4)
        m["under_coding_rate"] = round(float(under.mean()), 4)
        m["same_category_other_branch_rate"] = round(float(sib.mean()), 4)
        d["chapter"] = d["gold"].str[:1]
        ch = {k: _acc(v["correct"]) for k, v in d.groupby("chapter")}
        m["by_chapter_letter"] = ch
        ov = m["overall"]["acc"] or 0
        m["chapters_below_overall"] = sorted(k for k, v in ch.items() if v["n"] >= 10 and v["acc"] < ov)
    else:
        d["tag"] = d["gold"].map(lambda c: kb.sem_tag.get(c, "unknown"))
        m["by_semantic_tag"] = {k: _acc(v["correct"]) for k, v in d.groupby("tag") if len(v) >= 5}
        m["snomed_hierarchy_distance"] = "not computed: no SNOMED relationships supplied"
    # slices
    nt = d["ent_norm"].str.split().str.len().fillna(0)
    d["len_bucket"] = np.where(nt <= 1, "1", np.where(nt <= 3, "2-3", "4+"))
    m["by_entity_len"] = {k: _acc(v["correct"]) for k, v in d.groupby("len_bucket")}
    d["ctx_empty"] = d["chunk_text"].str.strip() == ""
    m["empty_context"] = _acc(d.loc[d["ctx_empty"], "correct"])
    m["with_context"] = _acc(d.loc[~d["ctx_empty"], "correct"])
    if len(feats):
        q1 = feats.groupby("qid")[["ent_has_abbr", "cue_neg", "cue_hist", "cue_unc"]].max()
        for col in q1.columns:
            d[col] = d["pair_id"].map(q1[col]).fillna(0)
        m["abbreviation_terms"] = _acc(d.loc[d["ent_has_abbr"] > 0, "correct"])
        m["negated_terms"] = _acc(d.loc[d["cue_neg"] > 0, "correct"])
        m["history_terms"] = _acc(d.loc[d["cue_hist"] > 0, "correct"])
        m["uncertain_terms"] = _acc(d.loc[d["cue_unc"] > 0, "correct"])
    if vocab is not None:
        d["noisy"] = [any(not vocab.known(t) for t in e.split()) for e in d["ent_norm"]]
        m["noisy_oov_terms"] = _acc(d.loc[d["noisy"], "correct"])
        m["clean_terms"] = _acc(d.loc[~d["noisy"], "correct"])
    # calibration + abstention
    m["ece"] = ece(d["prob"].values, d["correct"].values)
    cov = {}
    for thr in (0.0, 0.3, 0.5, 0.7, 0.9):
        keep = d["prob"] >= thr
        cov[str(thr)] = {"coverage": round(float(keep.mean()), 4),
                         "acc": round(float(d.loc[keep, "correct"].mean()), 4) if keep.any() else None}
    m["coverage_vs_accuracy"] = cov
    return m, d


def headline(ms: dict) -> dict:
    a = ms["ICD10CM"]["overall"]["acc"]
    b = ms["SNOMEDCT_US"]["overall"]["acc"]
    h = min(a, b)
    return {"ICD10CM": a, "SNOMEDCT_US": b, "headline_min": h, "gap_to_95": round(0.95 - h, 4)}


def bootstrap_ci(correct, n: int = 1000, seed: int = 13) -> dict:
    """Percentile bootstrap 95% CI of mean accuracy (row resampling)."""
    x = np.asarray(correct, dtype=float)
    if len(x) == 0:
        return {"acc": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return {"acc": round(float(x.mean()), 4), "lo": round(float(np.percentile(bs, 2.5)), 4),
            "hi": round(float(np.percentile(bs, 97.5)), 4), "n": int(len(x))}


def bootstrap_headline(correct_icd, correct_sno, n: int = 1000, seed: int = 13) -> dict:
    """CI of min(ICD, SNOMED): each system resampled independently, min taken per resample."""
    a, b = np.asarray(correct_icd, float), np.asarray(correct_sno, float)
    rng = np.random.default_rng(seed)
    ma = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    mb = rng.choice(b, size=(n, len(b)), replace=True).mean(axis=1)
    mn = np.minimum(ma, mb)
    return {"headline": round(float(min(a.mean(), b.mean())), 4), "lo": round(float(np.percentile(mn, 2.5)), 4),
            "hi": round(float(np.percentile(mn, 97.5)), 4)}
