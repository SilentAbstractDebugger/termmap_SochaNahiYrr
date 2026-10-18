"""Integration tests (need the challenge files locally; skipped otherwise).

Inputs used here are PUBLIC statement examples or SYNTHETIC strings written for this repo.
"""
import copy
import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from conftest import ROOT, needs_data, needs_model

HARD = yaml.safe_load((ROOT / "tests" / "hard_cases.yaml").read_text())


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _synthetic_wide(tmp_path):
    df = pd.DataFrame({"chunk_id": [f"c{i}" for i in range(len(HARD))],
                       "clinical_entity": [h["entity"] for h in HARD],
                       "chunk_text": [h["chunk_text"] for h in HARD]})
    p = tmp_path / "synthetic_wide.csv"
    df.to_csv(p, index=False)
    return p, df


def _synthetic_long(tmp_path):
    rows = []
    for h in HARD:
        for s in ("ICD10CM", "SNOMEDCT_US"):
            rows.append({"entity": h["entity"], "chunk_text": h["chunk_text"], "coding_system": s,
                         "code": "", "display": "", "subconcept": "DiagnosisCode"})
    p = tmp_path / "synthetic_long.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p, pd.DataFrame(rows)


@pytest.fixture(scope="module")
def runs(tmp_path_factory, R, model):
    from termmap.predict import run

    t = tmp_path_factory.mktemp("pred")
    wp, wdf = _synthetic_wide(t)
    lp, ldf = _synthetic_long(t)
    out = {}
    for name, p in (("wide", wp), ("long", lp)):
        for k in (1, 2):
            o = t / f"{name}_{k}.csv"
            info = run(str(p), str(o), "configs/evaluation.yaml", diagnostics_path=str(t / f"diag_{name}_{k}.csv"), R=R)
            out[(name, k)] = (o, info)
    return out, wdf, ldf


# ---------------- test_schema (output side)
@needs_model
def test_output_schema_wide(runs):
    out, wdf, _ = runs
    o, _ = out[("wide", 1)]
    p = pd.read_csv(o, dtype=str, keep_default_na=False)
    assert list(p.columns) == ["chunk_id", "clinical_entity", "icd10cm_code", "snomed_code"]
    assert len(p) == len(wdf) and list(p["chunk_id"]) == list(wdf["chunk_id"])  # order preserved


@needs_model
def test_output_schema_long(runs):
    out, _, ldf = runs
    o, info = out[("long", 1)]
    p = pd.read_csv(o, dtype=str, keep_default_na=False)
    assert list(p.columns) == list(ldf.columns) and len(p) == len(ldf)
    assert list(p["entity"]) == list(ldf["entity"]) and list(p["coding_system"]) == list(ldf["coding_system"])
    w = pd.read_csv(info["wide_output"], dtype=str, keep_default_na=False)
    assert len(w) == len(ldf)


# ---------------- test_dictionary_validity / both systems / no duplicates
@needs_model
def test_dictionary_validity_and_both_systems(runs, R):
    out, _, _ = runs
    for key in (("wide", 1), ("long", 1)):
        o, info = out[key]
        w = pd.read_csv(info["wide_output"], dtype=str, keep_default_na=False)
        assert w["icd10cm_code"].map(R.kbs["ICD10CM"].valid).all()
        assert w["snomed_code"].map(R.kbs["SNOMEDCT_US"].valid).all()
        assert (w["icd10cm_code"] != "").all() and (w["snomed_code"] != "").all()


@needs_model
def test_no_duplicate_rows_introduced(runs):
    out, wdf, ldf = runs
    assert len(pd.read_csv(out[("wide", 1)][0])) == len(wdf)
    assert len(pd.read_csv(out[("long", 1)][0])) == len(ldf)


@needs_data
def test_gold_normalization_reports_malformed(R):
    from termmap.data import diagnosis_pairs, load_train

    pairs = diagnosis_pairs(load_train(R.cfg), R.kbs)
    bad = pairs[pairs["has_ICD10CM"] & ~pairs["valid_ICD10CM"]]
    # every invalid gold is reported by the audit, nothing silently remapped
    assert set(bad["gold_ICD10CM"]) <= {"L6.0"}


# ---------------- test_determinism
@needs_model
def test_determinism(runs):
    out, _, _ = runs
    for name in ("wide", "long"):
        assert _sha(out[(name, 1)][0]) == _sha(out[(name, 2)][0])


# ---------------- test_hard_cases
@needs_model
@pytest.mark.parametrize("case", HARD, ids=[h["id"] for h in HARD])
def test_hard_case(case, predictor, R):
    q = pd.DataFrame({"qid": ["h"], "entity": [case["entity"]], "chunk_text": [case["chunk_text"]]})
    tops = predictor.predict(q)
    icd = tops["ICD10CM"].iloc[0]["code"]
    sno = tops["SNOMEDCT_US"].iloc[0]["code"]
    assert R.kbs["ICD10CM"].valid(icd) and R.kbs["SNOMEDCT_US"].valid(sno)
    msgs = []
    if "expect_icd" in case and icd != case["expect_icd"]:
        msgs.append(f"ICD {icd} != {case['expect_icd']}")
    if "expect_snomed" in case and case.get("conflict") != "snomed" and sno != case["expect_snomed"]:
        msgs.append(f"SNOMED {sno} != {case['expect_snomed']}")
    if "plaus_icd_prefix" in case and not any(icd.startswith(p) for p in case["plaus_icd_prefix"]):
        msgs.append(f"ICD {icd} not in plausible families {case['plaus_icd_prefix']}")
    if "plaus_snomed_words" in case:
        txt = " ".join(R.kbs["SNOMEDCT_US"].descs_norm[sno])
        if not any(w in txt for w in case["plaus_snomed_words"]):
            msgs.append(f"SNOMED {sno} ({R.kbs['SNOMEDCT_US'].preferred[sno]}) lacks {case['plaus_snomed_words']}")
    assert not msgs, "; ".join(msgs)


# ---------------- test_leakage
@needs_data
def test_leakage_group_and_code_disjoint(R):
    from termmap.data import SYSTEMS, diagnosis_pairs, load_train, system_rows
    from termmap.retrieve_training import TrainingResources
    from termmap.util import read_json

    sp = read_json(ROOT / "reports" / "splits.json")
    pairs = diagnosis_pairs(load_train(R.cfg), R.kbs)
    tr = pairs[pairs["pair_id"].isin(set(sp["train_pair_ids"]))]
    va = pairs[pairs["pair_id"].isin(set(sp["val_pair_ids"]))]
    te = pairs[pairs["pair_id"].isin(set(sp["test_pair_ids"]))]
    # FINAL TEST split: disjoint from train and val, no shared normalized entity or chunk
    assert len(te) and not (set(te["pair_id"]) & (set(tr["pair_id"]) | set(va["pair_id"])))
    assert not (set(te["ent_norm"]) & (set(tr["ent_norm"]) | set(va["ent_norm"])))
    assert not (set(sp["oof_fold_dev"]) & set(te["pair_id"]))
    assert not (set(tr["ent_norm"]) & set(va["ent_norm"]))
    ctr = {c for c in tr["chunk_text"] if c.strip()}
    assert not (ctr & {c for c in va["chunk_text"] if c.strip()})
    for s in SYSTEMS:
        res = TrainingResources(system_rows(tr, s), R.kbs[s])
        assert not (set(res.alias) & set(system_rows(va, s)["ent_norm"]))
    # code-disjoint: no held-out code reachable from the training-derived resources
    cd = {s: set(sp["code_disjoint_codes"][s]) for s in SYSTEMS}
    keep = ~(pairs["gold_ICD10CM"].isin(cd["ICD10CM"]) | pairs["gold_SNOMEDCT_US"].isin(cd["SNOMEDCT_US"]))
    keep &= ~pairs["pair_id"].isin(set(sp["test_pair_ids"]))
    cdt = pairs[keep]
    for s in SYSTEMS:
        res = TrainingResources(system_rows(cdt, s), R.kbs[s])
        assert not (set(res.codes) & cd[s]) and not (set(res.alias.values()) & cd[s])


# ---------------- test_drift
@needs_model
def test_drift_modified_dictionary(tmp_path, cfg, model, R):
    """Swap in a modified ICD dictionary (codes removed + synthetic codes added). SNOMED components
    are reused from the loaded resources to keep memory bounded."""
    from termmap.pipeline import Predictor, load_resources
    from termmap.retrieve_lexical import LexicalIndex
    from termmap.terminology import build_icd_kb

    icd = pd.read_excel(ROOT / "resources" / "icd10cm_codes.xlsx", dtype=str)
    removed = ["I10", "E11.9", "R07.9"]
    icd2 = icd[~icd["Code"].isin(removed)].copy()
    added = pd.DataFrame({"Code": ["Z99.991", "Z99.992"],
                          "Description": ["synthetic new code alpha condition", "synthetic new code beta condition"]})
    icd2 = pd.concat([icd2, added], ignore_index=True)
    kb2 = build_icd_kb(icd2)
    kb2.meta = {"sha256": "modified-test-dictionary"}
    c2 = copy.deepcopy(cfg)
    c2["dense"]["enabled"] = False
    R2 = load_resources(c2, kbs={"ICD10CM": kb2, "SNOMEDCT_US": R.kbs["SNOMEDCT_US"]},
                        lex={"ICD10CM": LexicalIndex(kb2), "SNOMEDCT_US": R.lex["SNOMEDCT_US"]})
    assert not any(R2.kbs["ICD10CM"].valid(c) for c in removed)
    pred = Predictor(R2, model)
    q = pd.DataFrame({"qid": [f"h{i}" for i in range(len(HARD))] + ["new"],
                      "entity": [h["entity"] for h in HARD] + ["synthetic new code alpha condition"],
                      "chunk_text": [h["chunk_text"] for h in HARD] + [""]})
    tops = pred.predict(q)
    t1 = tops["ICD10CM"][tops["ICD10CM"]["rank"] == 0]
    assert t1["code"].map(R2.kbs["ICD10CM"].valid).all()
    assert not set(t1["code"]) & set(removed)
    assert len(t1) == len(q)
    # a brand-new code with no training support can still win when the text matches it exactly
    # (code-agnostic features). NB: with only partial overlap ("alpha condition") the model preferred
    # an older training-supported code -- reported in the README as a known limitation.
    new_code = t1.set_index("qid").loc["new", "code"]
    del R2, pred
    import gc

    gc.collect()
    assert new_code in {"Z99.991", "Z99.992"}


# ---------------- test_fallback_modes
@needs_model
def test_fallback_modes(cfg, model, R):
    """Dense model missing + crosswalk missing -> engine still runs and reports its mode."""
    from termmap.pipeline import Predictor, load_resources

    c2 = copy.deepcopy(cfg)
    c2["dense"]["enabled"] = True
    c2["dense"]["backend"] = "sentence_transformer"
    c2["dense"]["model_path"] = "resources/models/does-not-exist"
    c2["resources"]["crosswalk_path"] = "resources/no-crosswalk.tsv"
    R2 = load_resources(c2, kbs=R.kbs, lex=R.lex)
    assert str(R2.mode["dense"]).startswith("unavailable") and R2.mode["crosswalk"] == "not supplied"
    q = pd.DataFrame({"qid": ["a"], "entity": ["chest pain"], "chunk_text": [""]})
    tops = Predictor(R2, model).predict(q)
    assert len(tops["ICD10CM"]) and len(tops["SNOMEDCT_US"])


# ---------------- test_empty_context (pipeline level)
@needs_model
def test_empty_entity_and_context(predictor, R):
    q = pd.DataFrame({"qid": ["e1", "e2"], "entity": ["", "chest pain"], "chunk_text": ["", ""]})
    tops = predictor.predict(q)
    # empty entity + empty context yields no candidates -> predict.run falls back to a valid code
    assert set(tops["ICD10CM"]["qid"]) >= {"e2"}
