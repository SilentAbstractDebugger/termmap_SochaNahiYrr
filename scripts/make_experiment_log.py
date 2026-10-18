"""reports/experiment_log.md: one line per experiment step (what, why, numbers, decision).
Numbers are read from report JSON files written by scripts/train.py in this iteration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from termmap.util import ROOT, read_json  # noqa: E402


def v(rep, split, name):
    try:
        x = rep[split]["variants"][name]
        return (f"ICD {x['ICD10CM']['overall_mean']}+-{x['ICD10CM']['overall_std']}, "
                f"SNOMED {x['SNOMEDCT_US']['overall_mean']}+-{x['SNOMEDCT_US']['overall_std']}"), x
    except KeyError:
        return "n/a", None


def main():
    base = read_json(ROOT / "reports" / "baseline_iter2.json")
    dev = read_json(ROOT / "reports" / "iter2_dev.json")
    decisions = read_json(ROOT / "reports" / "decisions_iter2.json") if (ROOT / "reports" / "decisions_iter2.json").exists() else {}
    L = ["# Experiment log (iteration 2)", "",
         "Development metric = group-validation (model selection) and code-disjoint (long tail). Mean +- std over "
         "3 reranker seeds; a change is kept only if it helps beyond ~2 points or CIs separate, or if it is neutral "
         "and required (calibration, routing). FINAL TEST is not used here.", "",
         "| step | what / why | validation | code-disjoint | decision |", "|---|---|---|---|---|"]
    bv, _ = v(base, "val", "stage1")
    bc, _ = v(base, "code_disjoint", "stage1")
    L.append(f"| 0 | Re-measured baseline: run-3 features, new 3-way split (train 1874 pairs; FINAL TEST carved out) | {bv} | {bc} | reference |")
    rows = [
        ("1-2", "stage1", "+ RRF fusion feature, + OCR glyph views, + noise-augmented training rows (25%), + learned conditional priors, + origin qualifier, + context-vs-candidate char cosine (all new features on)"),
        ("2", "stage1_no_aug_rows", "same tables WITHOUT the noise-augmented training rows"),
        ("1", "stage1_no_rrf", "without RRF features"),
        ("5", "stage1_no_priors", "without learned conditional priors"),
        ("-", "stage1_no_context", "without context features"),
        ("-", "stage1_no_dense", "without static-vector dense features"),
        ("5", "stage2", "+ second-stage joint reranker (other system's top-1 agreement)"),
        ("7", "stage2_calibrated", "+ isotonic calibration on OOF predictions (accuracy-neutral by design; see ECE)"),
    ]
    for step, name, what in rows:
        a, _ = v(dev, "val", name)
        b, _ = v(dev, "code_disjoint", name)
        L.append(f"| {step} | {what} | {a} | {b} | {decisions.get(name, 'see SUMMARY')} |")
    fin = read_json(ROOT / "reports" / "metrics.json") if (ROOT / "reports" / "metrics.json").exists() else None
    if fin:
        a, _ = v(fin, "val", "stage2_icd_only_calibrated")
        b, _ = v(fin, "code_disjoint", "stage2_icd_only_calibrated")
        L.append(f"| 5+7 | FROZEN: stage 2 for ICD only + isotonic calibration (both systems) | {a} | {b} | "
                 "selected on validation; FINAL TEST scored once (SUMMARY.md) |")
    L += ["| 1 | Biomedical dense encoders (gte-base, SapBERT, BioLORD, MedCPT, bge-m3) | NOT RUN (no GPU / no HF access) | NOT RUN | code + `scripts/kaggle_run.sh` |",
          "| 3 | Contrastive fine-tuning | NOT RUN for accuracy (mechanics tested with a tiny random BERT) | NOT RUN | code ready |",
          "| 4 | Cross-encoder features | NOT RUN for accuracy (mechanics tested with a tiny random BERT) | NOT RUN | code ready |",
          "| 6 | Local LLM adjudication | NOT RUN (no weights / GPU) | NOT RUN | not implemented |", ""]
    (ROOT / "reports" / "experiment_log.md").write_text("\n".join(L) + "\n")
    print("wrote reports/experiment_log.md")


if __name__ == "__main__":
    main()
