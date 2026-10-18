"""One-command prediction.

    python -m termmap.predict --input <eval file> --output predictions.csv --config configs/evaluation.yaml

Writes (1) predictions in the SAME structure as the input (long or wide), (2) the wide file
[chunk_id, clinical_entity, icd10cm_code, snomed_code] (same path when the input is wide,
otherwise <output stem>_wide.csv), (3) diagnostics (reports/diagnostics.csv by default, kept out
of the prediction schema). Runs offline; if models/final.pkl is missing it is rebuilt
deterministically from the training file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from .data import SYSTEMS, diagnosis_pairs, load_train
from .input_schema import detect_schema, read_input, to_queries
from .io_adapter import long_frame, wide_frame
from .joint import joint_signals
from .crosswalk import Crosswalk
from .diagnostics import build_diagnostics
from .input_schema import canon_system
from .output_validation import validate_long, validate_wide
from .pipeline import Predictor, fit_final, load_model, load_resources, save_model
from .specificity import safe_decode
from .splits import make_splits
from .util import ROOT, file_sha256, load_config, resolve

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def ensure_model(R, cfg, model_path: Path) -> dict:
    if model_path.exists():
        m = load_model(model_path)
        stale = [s for s in SYSTEMS if m["kb_hashes"].get(s) != R.kbs[s].meta.get("sha256")]
        if stale:
            print(f"[termmap] note: model trained against different dictionary files for {stale}; "
                  f"features are code-agnostic so it still runs (see drift_report).", file=sys.stderr)
        return m
    print("[termmap] models/final.pkl missing -> rebuilding deterministically from training data",
          file=sys.stderr)
    pairs = diagnosis_pairs(load_train(cfg), R.kbs)
    s = cfg["splits"]
    sp = make_splits(pairs, cfg["seed"], s["val_fraction"], s["code_disjoint_fraction"], s["oof_folds"],
                     s.get("test_fraction", 0.0))
    m = fit_final(R, pairs, sp["oof_fold_all"], cfg["rerank"]["code_dropout"])
    save_model(m, model_path)
    return m


def run(input_path: str, output_path: str, config: str | None = None, model_path: str | None = None,
        diagnostics_path: str | None = None, wide_path: str | None = None, R=None) -> dict:
    t0 = time.time()
    cfg = load_config(config)
    R = R or load_resources(cfg)
    mp = resolve(model_path or cfg.get("model_path", "models/final.pkl"))
    model = ensure_model(R, cfg, mp)
    inp = read_input(input_path)
    schema = detect_schema(inp)
    q = to_queries(inp, schema)
    q["entity_raw"] = q["entity"]
    # dedupe identical (entity, chunk) for compute; results mapped back to every row (order kept)
    key = q["entity"] + "\x00" + q["chunk_text"]
    uniq = key.drop_duplicates()
    uq = pd.DataFrame({"qid": [f"u{i:07d}" for i in range(len(uniq))], "key": uniq.values})
    qmap = dict(zip(uq["key"], uq["qid"]))
    q["qid"] = key.map(qmap)
    uqq = q.drop_duplicates("qid")[["qid", "entity", "chunk_text"]]
    pred = Predictor(R, model)
    tops = pred.predict(uqq)
    preds, rules = {}, {}
    for s in SYSTEMS:
        top = tops[s]
        t1 = dict(zip(top.loc[top["rank"] == 0, "qid"], top.loc[top["rank"] == 0, "code"])) if len(top) else {}
        fb = model["tres"][s].most_frequent
        p, r = {}, {}
        for qid in uqq["qid"]:
            c = t1.get(qid)
            if c is None:
                c, rule = fb, "fallback_most_frequent_training_code"
            else:
                c, rule = safe_decode(c, R.kbs[s])
                if not c:
                    c, rule = fb, "fallback_most_frequent_training_code"
            p[qid], r[qid] = c, rule
        preds[s], rules[s] = p, r
    wide = wide_frame(q, preds)
    validate_wide(wide, len(inp), R.kbs)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if schema.layout == "long":
        lf = long_frame(inp, q, schema, preds, R.kbs)
        validate_long(lf, inp, schema.code_col, [canon_system(x) for x in inp[schema.system_col]], R.kbs)
        lf.to_csv(out_p, index=False)
        wide_p = Path(wide_path) if wide_path else out_p.with_name(out_p.stem + "_wide.csv")
    else:
        wide_p = Path(wide_path) if wide_path else out_p
    wide.to_csv(wide_p, index=False)
    xw = Crosswalk.load(resolve(cfg["resources"].get("crosswalk_path")))
    js = joint_signals(uqq["qid"], [preds["ICD10CM"][x] for x in uqq["qid"]],
                       [preds["SNOMEDCT_US"][x] for x in uqq["qid"]], tops["ICD10CM"], tops["SNOMEDCT_US"],
                       R.kbs, R.lex, xw)
    diag = build_diagnostics(q, preds, tops, js, rules, R.mode, pred.low_support)
    dp = Path(diagnostics_path) if diagnostics_path else ROOT / "reports" / "diagnostics.csv"
    dp.parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(dp, index=False)
    info = {"input": str(input_path), "layout": schema.layout, "rows": len(inp), "unique_queries": len(uqq),
            "output": str(out_p), "wide_output": str(wide_p), "diagnostics": str(dp),
            "mode": R.mode, "seconds": round(time.time() - t0, 1),
            "output_sha256": file_sha256(out_p), "wide_sha256": file_sha256(wide_p)}
    print(json.dumps(info, indent=2))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="configs/evaluation.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--diagnostics", default=None)
    ap.add_argument("--wide-output", default=None)
    a = ap.parse_args(argv)
    run(a.input, a.output, a.config, a.model, a.diagnostics, a.wide_output)


if __name__ == "__main__":
    main()
