"""Run the Phase 0 data audit: python scripts/audit.py [--config configs/default.yaml]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from termmap.data import load_train  # noqa: E402
from termmap.data_audit import audit_markdown, run_audit  # noqa: E402
from termmap.terminology import load_kbs  # noqa: E402
from termmap.util import ROOT, load_config, write_json  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    kbs = load_kbs(cfg)
    a = run_audit(load_train(cfg), kbs)
    a["dictionaries"] = {s: kb.meta for s, kb in kbs.items()}
    write_json(ROOT / "reports" / "data_audit.json", a)
    (ROOT / "reports" / "data_audit.md").write_text(audit_markdown(a))
    print("Discrepancies vs brief:", a["discrepancies_vs_brief"] or "none")
    for s in ("ICD10CM", "SNOMEDCT_US"):
        print(s, a[s]["buckets"], "not in dict:", a[s]["gold_not_in_dict"])
