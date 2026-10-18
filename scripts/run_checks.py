"""Produce reports/hard_cases.json and reports/determinism.json (public/synthetic inputs only),
and dry-run predictions on the validation split + a held-out sample in the submission schema.

    python scripts/run_checks.py
"""
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from termmap.data import diagnosis_pairs, load_train  # noqa: E402
from termmap.pipeline import Predictor, load_model, load_resources  # noqa: E402
from termmap.predict import run  # noqa: E402
from termmap.util import ROOT, file_sha256, load_config, read_json, write_json  # noqa: E402


def main():
    cfg = load_config("configs/evaluation.yaml")
    R = load_resources(cfg)
    model = load_model(ROOT / "models" / "final.pkl")
    P = Predictor(R, model)
    hard = yaml.safe_load((ROOT / "tests" / "hard_cases.yaml").read_text())
    res = []
    for h in hard:
        q = pd.DataFrame({"qid": ["h"], "entity": [h["entity"]], "chunk_text": [h["chunk_text"]]})
        t = P.predict(q)
        icd, sno = t["ICD10CM"].iloc[0]["code"], t["SNOMEDCT_US"].iloc[0]["code"]
        ok, why = True, []
        if "expect_icd" in h and icd != h["expect_icd"]:
            ok = False
            why.append(f"expected ICD {h['expect_icd']}")
        if "expect_snomed" in h and h.get("conflict") != "snomed" and sno != h["expect_snomed"]:
            ok = False
            why.append(f"expected SNOMED {h['expect_snomed']}")
        if "plaus_icd_prefix" in h and not any(icd.startswith(p) for p in h["plaus_icd_prefix"]):
            ok = False
            why.append(f"ICD not in {h['plaus_icd_prefix']}")
        if "plaus_snomed_words" in h and not any(w in " ".join(R.kbs["SNOMEDCT_US"].descs_norm[sno])
                                                 for w in h["plaus_snomed_words"]):
            ok = False
            why.append(f"SNOMED lacks {h['plaus_snomed_words']}")
        res.append({"id": h["id"], "source": h["source"], "entity": h["entity"], "icd": icd,
                    "icd_desc": R.kbs["ICD10CM"].preferred[icd], "snomed": sno,
                    "snomed_desc": R.kbs["SNOMEDCT_US"].preferred[sno], "pass": ok, "why": "; ".join(why),
                    "conflict": h.get("conflict", "")})
        print(("PASS " if ok else "FAIL ") + f"{h['id']}: {icd} | {sno} {R.kbs['SNOMEDCT_US'].preferred[sno]} {why}")
    write_json(ROOT / "reports" / "hard_cases.json", res)

    # determinism: two full one-command runs on the synthetic file
    tmp = ROOT / "outputs"
    tmp.mkdir(exist_ok=True)
    syn = pd.DataFrame({"chunk_id": [h["id"] for h in hard], "clinical_entity": [h["entity"] for h in hard],
                        "chunk_text": [h["chunk_text"] for h in hard]})
    syn.to_csv(tmp / "synthetic_eval.csv", index=False)
    hs = []
    info = run(str(tmp / "synthetic_eval.csv"), str(tmp / "synthetic_pred_1.csv"), "configs/evaluation.yaml",
               diagnostics_path=str(tmp / "synthetic_diag_1.csv"), R=R)
    hs.append(info["output_sha256"])
    # second run in a FRESH process (one-command path, cold resources) -> stronger determinism check.
    # Free this process's resources first (an 8 GB machine cannot hold two copies).
    import gc
    import subprocess

    del R, P, model
    gc.collect()

    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run([sys.executable, "-m", "termmap.predict", "--input", str(tmp / "synthetic_eval.csv"),
                    "--output", str(tmp / "synthetic_pred_2.csv"), "--config", "configs/evaluation.yaml",
                    "--diagnostics", str(tmp / "synthetic_diag_2.csv")], check=True, cwd=ROOT, env=env,
                   stdout=subprocess.DEVNULL)
    hs.append(file_sha256(tmp / "synthetic_pred_2.csv"))
    write_json(ROOT / "reports" / "determinism.json", {
        "runs": 2, "hashes": hs, "identical": hs[0] == hs[1],
        "result": ("identical output hashes across 2 runs (2nd in a fresh process)" if hs[0] == hs[1]
                   else "MISMATCH"),
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "cpu_only": True}})

    # dry run in submission schema: validation split (long, train-style) + wide held-out sample,
    # each through the one-command path in its own process
    train = load_train(cfg)
    from termmap.terminology import load_kbs

    kbs = load_kbs(cfg)
    pairs = diagnosis_pairs(train, kbs)
    del kbs
    gc.collect()
    sp = read_json(ROOT / "reports" / "splits.json")
    val_keys = pairs[pairs["pair_id"].isin(set(sp["val_pair_ids"]))][["entity", "chunk_text"]]
    priv = ROOT / "reports" / "private"
    vl = train[train["subconcept"] == "DiagnosisCode"].merge(val_keys, on=["entity", "chunk_text"])
    vl.assign(code="", display="").to_csv(priv / "dryrun_val_long_input.csv", index=False)
    vl.to_csv(priv / "dryrun_val_long_gold.csv", index=False)
    wide = val_keys.reset_index(drop=True)
    pd.DataFrame({"chunk_id": [f"val_{i:04d}" for i in range(len(wide))], "clinical_entity": wide["entity"],
                  "chunk_text": wide["chunk_text"]}).to_csv(priv / "dryrun_val_wide_input.csv", index=False)
    for src, dst, dg in (("dryrun_val_long_input.csv", "predictions_val_long.csv", "diagnostics_val.csv"),
                         ("dryrun_val_wide_input.csv", "predictions_val_wide.csv", "diagnostics_val_wide.csv")):
        subprocess.run([sys.executable, "-m", "termmap.predict", "--input", str(priv / src), "--output",
                        str(priv / dst), "--config", "configs/evaluation.yaml", "--diagnostics", str(priv / dg)],
                       check=True, cwd=ROOT, env=env, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
