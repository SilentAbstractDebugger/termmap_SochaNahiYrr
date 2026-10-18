"""Evaluation utilities.

  python scripts/evaluate.py --render                    # reports/metrics.json -> reports/metrics.md + ablations.md
  python scripts/evaluate.py --pred P.csv --gold G.csv   # score a prediction file (long or wide)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from termmap.data import SYSTEMS, normalize_gold  # noqa: E402
from termmap.util import ROOT, read_json, write_json  # noqa: E402


def score_files(pred_p, gold_p):
    g = pd.read_csv(gold_p, dtype=str, keep_default_na=False)
    p = pd.read_csv(pred_p, dtype=str, keep_default_na=False)
    keep = g["subconcept"] == "DiagnosisCode" if "subconcept" in g else pd.Series(True, index=g.index)
    out = {}
    col = {"ICD10CM": "icd10cm_code", "SNOMEDCT_US": "snomed_code"}
    assert len(p) == len(g), "prediction file must be row-aligned with the gold file"
    for s in SYSTEMS:
        m = keep & (g["coding_system"] == s)
        gold = [normalize_gold(s, c) for c in g.loc[m, "code"]]
        src = p.loc[m, "code"] if "coding_system" in p.columns else p.loc[m, col[s]]
        pred = [normalize_gold(s, c) for c in src]
        out[s] = round(sum(a == b for a, b in zip(gold, pred)) / max(1, len(gold)), 4)
    out["headline_min"] = min(out.values())
    return out


def _ci(x):
    return f"{x['acc']} [{x['lo']}, {x['hi']}]" if x and x.get("acc") is not None else "n/a"


def render(tag="metrics"):
    m = read_json(ROOT / "reports" / f"{tag}.json")
    L = ["# Measured results (iteration 2)", "", f"Run mode: `{m['mode']}`  ", f"Reranker seeds: {m['seeds']}", ""]
    A = ["# Ablations (iteration 2)", "", "Mean +- std over reranker seeds; same candidate tables and splits per "
         "experiment. Differences < 2 points are noise unless bootstrap CIs separate (CIs in metrics.md).", ""]
    for split in ("val", "code_disjoint", "final_test"):
        if split not in m:
            continue
        x = m[split]["metrics"]
        L += [f"## {split} (main variant: {m[split]['main_variant']}, primary seed)", "",
              "| metric | ICD-10-CM | SNOMED CT |", "|---|---|---|"]
        L.append(f"| exact match [95% bootstrap CI] | {_ci(x['ICD10CM']['ci95'])} | {_ci(x['SNOMEDCT_US']['ci95'])} |")
        h = x["headline"]["ci95"]
        L.append(f"| headline (min) [CI] | {h['headline']} [{h['lo']}, {h['hi']}]; gap to 95%: {x['headline']['gap_to_95']} | |")
        for b in ("0", "1", "2-5", "6+"):
            L.append(f"| acc, code train-freq {b} | {x['ICD10CM']['by_bucket'].get(b)} | {x['SNOMEDCT_US']['by_bucket'].get(b)} |")
        L.append(f"| candidate pool recall | {x['ICD10CM']['pool_recall']} | {x['SNOMEDCT_US']['pool_recall']} |")
        L.append(f"| errors: retrieval / rank miss | {x['ICD10CM']['errors']} | {x['SNOMEDCT_US']['errors']} |")
        L.append(f"| first-stage recall@K | {x['ICD10CM'].get('recall_first_stage')} | {x['SNOMEDCT_US'].get('recall_first_stage')} |")
        L.append(f"| ECE (reported prob) | {x['ICD10CM']['ece']} | {x['SNOMEDCT_US']['ece']} |")
        if "ece_uncalibrated" in x["ICD10CM"]:
            L.append(f"| ECE before calibration | {x['ICD10CM']['ece_uncalibrated']} | {x['SNOMEDCT_US']['ece_uncalibrated']} |")
        L.append(f"| coverage vs acc | {x['ICD10CM']['coverage_vs_accuracy']} | {x['SNOMEDCT_US']['coverage_vs_accuracy']} |")
        for k in ("abbreviation_terms", "noisy_oov_terms", "negated_terms", "history_terms", "empty_context"):
            L.append(f"| {k} | {x['ICD10CM'].get(k)} | {x['SNOMEDCT_US'].get(k)} |")
        L.append(f"| ICD same category / same letter (partial credit, separate) | {x['ICD10CM']['same_category']} / "
                 f"{x['ICD10CM']['same_letter']} | n/a |")
        L.append(f"| ICD over / under coding rate | {x['ICD10CM']['over_coding_rate']} / {x['ICD10CM']['under_coding_rate']} | n/a |")
        L.append(f"| ICD letters below overall | {x['ICD10CM']['chapters_below_overall']} | |")
        L.append(f"| SNOMED hierarchy distance | n/a | {x['SNOMEDCT_US'].get('snomed_hierarchy_distance')} |")
        L.append("")
        A += [f"## {split}", "", "| configuration | ICD mean +- std | SNOMED mean +- std | headline (mean) | ICD unseen | SNOMED unseen |",
              "|---|---|---|---|---|---|"]
        for k, v in m[split]["variants"].items():
            A.append(f"| {k} | {v['ICD10CM']['overall_mean']} +- {v['ICD10CM']['overall_std']} | "
                     f"{v['SNOMEDCT_US']['overall_mean']} +- {v['SNOMEDCT_US']['overall_std']} | {v['headline_mean']} | "
                     f"{v['ICD10CM']['unseen_mean']} | {v['SNOMEDCT_US']['unseen_mean']} |")
        for k, v in m[split].get("heuristics", {}).items():
            A.append(f"| heuristic: {k} | {v.get('ICD10CM', {}).get('overall')} | {v.get('SNOMEDCT_US', {}).get('overall')} | "
                     f"{min(v.get('ICD10CM', {}).get('overall') or 0, v.get('SNOMEDCT_US', {}).get('overall') or 0)} | "
                     f"{v.get('ICD10CM', {}).get('unseen')} | {v.get('SNOMEDCT_US', {}).get('unseen')} |")
        A.append("")
    A += ["Not run in this environment (no GPU / no Hugging Face access): +gte-base, +SapBERT, +BioLORD, +MedCPT, "
          "+bge-m3 dense channels, +fine-tuned encoder, +cross-encoder, +LLM. Rows appear here after "
          "`scripts/kaggle_run.sh`.", ""]
    (ROOT / "reports" / "metrics.md").write_text("\n".join(L) + "\n")
    (ROOT / "reports" / "ablations.md").write_text("\n".join(A) + "\n")
    write_json(ROOT / "reports" / "ablations.json", {s: {"variants": m[s]["variants"], "heuristics": m[s].get("heuristics", {})}
                                                     for s in ("val", "code_disjoint", "final_test") if s in m})
    print("wrote reports/metrics.md, reports/ablations.md, reports/ablations.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--tag", default="metrics")
    ap.add_argument("--pred")
    ap.add_argument("--gold")
    a = ap.parse_args()
    if a.render:
        render(a.tag)
    if a.pred and a.gold:
        r = score_files(a.pred, a.gold)
        write_json(ROOT / "reports" / "score_file.json", r)
        print(r)
