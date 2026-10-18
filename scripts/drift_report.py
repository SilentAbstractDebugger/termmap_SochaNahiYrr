"""Terminology version drift report.

    python scripts/drift_report.py --icd <new icd file> --snomed <new snomed file>

Compares the NEW dictionary pair to the configured (current) one and lists: training gold codes
missing in the new release, gold codes whose descriptions changed, affected training rows, codes
added in the new release (all of which have no training support), and removed codes.
Writes reports/drift_report.md + .json. Because every reranker feature is code-agnostic, new
codes are predictable without retraining; removed gold codes make those training rows unusable
as positives (they are reported, never silently remapped).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from termmap.data import SYSTEMS, diagnosis_pairs, load_train  # noqa: E402
from termmap.terminology import load_kbs  # noqa: E402
from termmap.util import ROOT, load_config, write_json  # noqa: E402


def drift(cfg, icd_new, snomed_new, use_cache=True):
    old = load_kbs(cfg)
    new = load_kbs(cfg, icd_new or None, snomed_new or None, use_cache=use_cache)
    pairs = diagnosis_pairs(load_train(cfg), old)
    rep = {}
    for s in SYSTEMS:
        o, n = old[s], new[s]
        gold = pairs.loc[pairs[f"has_{s}"], f"gold_{s}"]
        gset = set(gold)
        missing = sorted(c for c in gset if o.valid(c) and not n.valid(c))
        changed = sorted(c for c in gset if o.valid(c) and n.valid(c) and o.descs_norm[c] != n.descs_norm[c])
        added = sorted(n.code_set - o.code_set)
        removed = sorted(o.code_set - n.code_set)
        rep[s] = {"old_sha256": o.meta.get("sha256"), "new_sha256": n.meta.get("sha256"),
                  "old_codes": len(o.codes), "new_codes": len(n.codes),
                  "added_codes": len(added), "removed_codes": len(removed),
                  "added_examples": added[:20], "removed_examples": removed[:20],
                  "training_gold_codes_missing_in_new": missing,
                  "affected_training_rows": int(gold.isin(set(missing)).sum()),
                  "training_gold_codes_with_changed_descriptions": changed[:50],
                  "n_changed_description_gold_codes": len(changed),
                  "new_codes_without_training_support": len(added)}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icd", default=None)
    ap.add_argument("--snomed", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    rep = drift(cfg, a.icd, a.snomed)
    write_json(ROOT / "reports" / "drift_report.json", rep)
    md = ["# Terminology drift report", ""]
    for s, r in rep.items():
        md += [f"## {s}", ""] + [f"- {k}: {v}" for k, v in r.items()] + [""]
    (ROOT / "reports" / "drift_report.md").write_text("\n".join(md))
    print({s: {k: v for k, v in r.items() if not isinstance(v, list)} for s, r in rep.items()})


if __name__ == "__main__":
    main()
