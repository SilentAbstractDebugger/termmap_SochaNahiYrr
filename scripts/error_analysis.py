"""Automatic error classification on the validation run (reports/private/val_run.pkl).

Writes:
  reports/private/error_rows.csv   every error with entity, context window, gold, pred, top
                                   candidates + scores, qualifier evidence, assigned reason
                                   (contains proprietary rows -> gitignored)
  reports/error_analysis.json/.md  counts per category and per challenge failure mode (shareable)
"""
import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from termmap.context import build_context  # noqa: E402
from termmap.data import SYSTEMS  # noqa: E402
from termmap.features import build_vocab  # noqa: E402
from termmap.terminology import icd_nodot, load_kbs  # noqa: E402
from termmap.util import ROOT, load_config, read_json, write_json  # noqa: E402

CATS = ["dictionary/version mismatch", "gold absent from candidates (missing synonym)",
        "gold absent from candidates (lexical)", "wrong specificity: over-coding",
        "wrong specificity: under-coding", "hierarchy confusion (same category)",
        "abbreviation misinterpreted", "qualifier ignored", "negation/history mishandled",
        "context ignored", "label noise suspicion", "retrieved but misranked"]


def classify(s, kb, r, f_q, vocab):
    gold, pred = r.gold, r.pred
    if not r.gold_valid:
        return "dictionary/version mismatch"
    g = f_q[f_q["code"] == gold]
    p = f_q[f_q["code"] == pred]
    if not len(g):
        etoks = set(r.ent_norm.split())
        gt = set(t for d in kb.descs_norm.get(gold, []) for t in d.split())
        return ("gold absent from candidates (missing synonym)" if not (etoks & gt)
                else "gold absent from candidates (lexical)")
    g, p = g.iloc[0], (p.iloc[0] if len(p) else None)
    if s == "ICD10CM":
        gn, pn = icd_nodot(gold), icd_nodot(pred)
        if pn.startswith(gn) and len(pn) > len(gn):
            return "wrong specificity: over-coding"
        if gn.startswith(pn) and len(gn) > len(pn):
            return "wrong specificity: under-coding"
    if p is not None:
        if p.get("ent_has_abbr", 0) > 0 and (p.get("char_abbr", 0) > 0 or p.get("exact_abbr", 0) > 0) \
                and g.get("char_abbr", 0) == 0 and g.get("exact_abbr", 0) == 0:
            return "abbreviation misinterpreted"
        contra = [c for c in p.index if c.endswith("_contra") and p[c] > 0]
        if contra and not any(g[c] > 0 for c in contra):
            return "qualifier ignored"
        if (p.get("cue_neg", 0) or p.get("cue_hist", 0)) and \
                (p.get("is_z", 0) != g.get("is_z", 0) or p.get("tag_situation", 0) != g.get("tag_situation", 0)):
            return "negation/history mishandled"
        if g.get("ctx_overlap", 0) > p.get("ctx_overlap", 0):
            return "context ignored"
        if p.get("tset_best", 0) >= 90 and g.get("tset_best", 0) <= 50:
            return "label noise suspicion"
    if s == "ICD10CM" and icd_nodot(gold)[:3] == icd_nodot(pred)[:3]:
        return "hierarchy confusion (same category)"
    return "retrieved but misranked"


def failure_modes(s, d, f, vocab):
    """Counts per challenge failure mode 1-7 among errors (a row can hit several modes)."""
    e = d[d["correct"] == 0]
    fm = Counter()
    for r in e.itertuples():
        fq = f[f["qid"] == r.pair_id]
        g = fq[fq["code"] == r.gold]
        p = fq[fq["code"] == r.pred]
        if len(g) and len(p) and p.iloc[0]["tset_best"] > g.iloc[0]["tset_best"]:
            fm["FM1 nearest string is the wrong concept"] += 1
        if r.reason == "abbreviation misinterpreted" or (len(p) and p.iloc[0].get("ent_has_abbr", 0) > 0):
            fm["FM2 abbreviation (term has a token in the abbreviation/translation lexicon)"] += 1
        if r.reason.startswith("wrong specificity") or r.reason.startswith("hierarchy"):
            fm["FM3 specificity calibration"] += 1
        if r.train_freq == 0:
            fm["FM5 long tail (code unseen in training)"] += 1
        if any(not vocab.known(t) for t in r.ent_norm.split()):
            fm["FM6 upstream noise (OOV token in term)"] += 1
        if r.reason == "dictionary/version mismatch":
            fm["FM7 terminology/version mismatch"] += 1
    return dict(fm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "reports" / "private" / "val_run.pkl"))
    ap.add_argument("--tag", default="val")
    a = ap.parse_args()
    cfg = load_config()
    kbs = load_kbs(cfg)
    vocab = build_vocab(kbs, cfg)
    run = pickle.load(open(a.run, "rb"))
    out, rows_all = {}, []
    by_pair = {}
    for s in SYSTEMS:
        d, f, top = run[s]["rows"].copy(), run[s]["feats"], run[s]["top"]
        kb = kbs[s]
        reasons = []
        for r in d.itertuples():
            if r.correct:
                reasons.append("")
                continue
            reasons.append(classify(s, kb, r, f[f["qid"] == r.pair_id], vocab))
        d["reason"] = reasons
        by_pair[s] = d.set_index("pair_id")["correct"]
        e = d[d["correct"] == 0]
        cnt = Counter(e["reason"])
        out[s] = {"n": int(len(d)), "errors": int(len(e)),
                  "by_category": {c: int(cnt.get(c, 0)) for c in CATS},
                  "by_failure_mode": failure_modes(s, d, f, vocab)}
        t3 = top.groupby("qid").apply(lambda g: " | ".join(
            f"{c}:{p:.2f}:{kb.preferred.get(c, '')[:40]}" for c, p in zip(g["code"], g["prob"])), include_groups=False)
        for r in e.itertuples():
            ctx = build_context(r.entity, r.chunk_text)
            rows_all.append({"system": s, "pair_id": r.pair_id, "entity": r.entity, "context_window": ctx.window,
                             "gold": r.gold, "gold_desc": kb.preferred.get(r.gold, "?"), "pred": r.pred,
                             "pred_desc": kb.preferred.get(r.pred, "?"), "prob": round(r.prob, 3),
                             "train_freq": r.train_freq, "in_pool": r.in_pool, "reason": r.reason,
                             "top3": t3.get(r.pair_id, ""), "negated": ctx.negated, "history": ctx.history})
    # FM4 granularity mismatch: pairs right in one system and wrong in the other
    both = pd.concat([by_pair["ICD10CM"].rename("icd"), by_pair["SNOMEDCT_US"].rename("sno")], axis=1).dropna()
    out["FM4_granularity"] = {
        "pairs_with_both_labels": int(len(both)),
        "icd_right_snomed_wrong": int(((both.icd == 1) & (both.sno == 0)).sum()),
        "snomed_right_icd_wrong": int(((both.icd == 0) & (both.sno == 1)).sum()),
        "both_wrong": int(((both.icd == 0) & (both.sno == 0)).sum()),
        "both_right": int(((both.icd == 1) & (both.sno == 1)).sum())}
    priv = ROOT / "reports" / "private"
    priv.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_all).to_csv(priv / f"error_rows_{a.tag}.csv", index=False)
    log_p = ROOT / "reports" / "dev_log.json"
    out["fixes_log"] = read_json(log_p) if log_p.exists() else []
    write_json(ROOT / "reports" / f"error_analysis_{a.tag}.json", out)
    md = [f"# Error analysis ({a.tag} split)", "",
          "Automatic classification of every error (rules in scripts/error_analysis.py). Counts only;",
          "row-level detail with proprietary text is in reports/private/ (gitignored).", ""]
    bp = ROOT / "reports" / "private" / f"error_analysis_{a.tag}_before.json"
    before = read_json(bp) if bp.exists() else None
    for s in SYSTEMS:
        md += [f"## {s}: {out[s]['errors']} errors / {out[s]['n']} rows"
               + (f" (before this iteration: {before[s]['errors']} / {before[s]['n']}, previous split)" if before else ""),
               "", "| category | n | before |", "|---|---|---|"]
        md += [f"| {c} | {n} | {before[s]['by_category'].get(c, '') if before else ''} |"
               for c, n in out[s]["by_category"].items()]
        md += ["", "| challenge failure mode (errors touching it) | n |", "|---|---|"]
        md += [f"| {k} | {v} |" for k, v in sorted(out[s]["by_failure_mode"].items())]
        md.append("")
    md += ["## FM4 granularity (pairs with both labels)", "", str(out["FM4_granularity"]), ""]
    md += ["## Recurring general error patterns", "",
           "- Retrieval miss without lexical overlap (lay paraphrase vs clinical description): the 'missing synonym' rows; "
           "needs semantic (dense) retrieval.",
           "- Near-synonym misranking where the gold follows a coder convention seen once or never (e.g. finding vs disorder "
           "concept, 'other' vs 'unspecified' leaf).",
           "- Abbreviation sense chosen from string evidence when the context word is outside the window.",
           "- Specificity within the right 3-character category (hierarchy confusion).", "",
           "Per-step effects are in reports/experiment_log.md; iteration-1 fixes in reports/dev_log.json.", ""]
    (ROOT / "reports" / f"error_analysis_{a.tag}.md").write_text("\n".join(md) + "\n")
    print({s: out[s]["by_category"] for s in SYSTEMS})


if __name__ == "__main__":
    main()
