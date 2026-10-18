"""Mechanical tests for the iteration-2 neural code paths, using TINY randomly initialised local
models (scripts/make_tiny_test_models.py). They verify plumbing, caching, determinism, offline
operation and leakage guards - NOT accuracy."""
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")
TINY = ROOT / "models" / "test"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


@pytest.fixture(scope="module")
def tiny():
    if not (TINY / "tiny-bi").exists():
        from importlib import util

        spec = util.spec_from_file_location("mk", ROOT / "scripts" / "make_tiny_test_models.py")
        mk = util.module_from_spec(spec)
        spec.loader.exec_module(mk)
        mk.main(TINY)
    return TINY


def _kbs():
    from termmap.terminology import build_icd_kb, build_snomed_kb

    icd = build_icd_kb(pd.DataFrame([
        ("I05", "Rheumatic mitral valve diseases"), ("I05.0", "Rheumatic mitral stenosis"),
        ("I34", "Nonrheumatic mitral valve disorders"), ("I34.2", "Nonrheumatic mitral (valve) stenosis"),
        ("Q23", "Congenital malformations of aortic and mitral valves"), ("Q23.2", "Congenital mitral stenosis"),
        ("G35", "Multiple sclerosis"), ("I10", "Essential (primary) hypertension"),
        ("N18.30", "Chronic kidney disease, stage 3 unspecified"), ("R07.9", "Chest pain, unspecified")],
        columns=["Code", "Description"]))
    icd.meta = {"sha256": "toy-icd"}
    sno = build_snomed_kb(pd.DataFrame([
        ("79619009", "Mitral valve stenosis", "1"), ("79619009", "Mitral valve stenosis (disorder)", "2"),
        ("24700007", "Multiple sclerosis", "1"), ("24700007", "Multiple sclerosis (disorder)", "2"),
        ("38341003", "Hypertensive disorder", "1"), ("38341003", "Hypertensive disorder (disorder)", "2"),
        ("29857009", "Chest pain", "1"), ("29857009", "Chest pain (finding)", "2")],
        columns=["snomed_code", "description", "description_order"]))
    sno.meta = {"sha256": "toy-sno"}
    return {"ICD10CM": icd, "SNOMEDCT_US": sno}


def _cfg(tmp_path):
    from termmap.util import load_config

    c = load_config("configs/evaluation.yaml")
    c["resources"]["cache_dir"] = str(tmp_path / "cache")
    return c


def test_dense_index_build_cache_offline_and_deterministic(tiny, tmp_path):
    from termmap.retrieve_dense import DenseIndex

    kbs, cfg = _kbs(), _cfg(tmp_path)
    spec = {"name": "tiny", "model_path": str(tiny / "tiny-bi"), "batch_size": 4, "save_every_batches": 2}
    a = DenseIndex(cfg, kbs, spec)
    n_store = len(a.store.keys)
    assert n_store >= len(kbs["ICD10CM"].doc_text) + len(kbs["SNOMEDCT_US"].doc_text)
    r1 = a.search("ICD10CM", ["mitral stenosis", "chest pain"], 5)
    b = DenseIndex(cfg, kbs, spec)  # reload: nothing re-embedded, identical results
    assert len(b.store.keys) == n_store
    assert b.search("ICD10CM", ["mitral stenosis", "chest pain"], 5) == r1
    assert all(all(isinstance(c, str) for c in a.doc_codes["ICD10CM"][di]) for di, _ in r1[0])
    # ICD path enrichment present
    assert any("; category:" in t for t in a.docs["ICD10CM"])


def test_dense_incremental_reembedding_on_drift(tiny, tmp_path):
    from termmap.retrieve_dense import DenseIndex
    from termmap.terminology import build_icd_kb

    kbs, cfg = _kbs(), _cfg(tmp_path)
    spec = {"name": "tiny", "model_path": str(tiny / "tiny-bi"), "batch_size": 4}
    a = DenseIndex(cfg, kbs, spec)
    n0 = len(a.store.keys)
    df = pd.DataFrame([(c, d) for c in kbs["ICD10CM"].codes for d in kbs["ICD10CM"].descs[c]] +
                      [("Z99.991", "Synthetic new condition alpha")], columns=["Code", "Description"])
    kb2 = build_icd_kb(df)
    kb2.meta = {"sha256": "toy-icd-2"}
    b = DenseIndex(cfg, {"ICD10CM": kb2, "SNOMEDCT_US": kbs["SNOMEDCT_US"]}, spec)
    assert 0 < len(b.store.keys) - n0 <= 2  # only the new description (+ its path text) embedded


def test_feature_builder_with_named_dense_channel(tiny, tmp_path):
    from termmap.features import FeatureBuilder, build_vocab
    from termmap.retrieve_dense import DenseIndex
    from termmap.retrieve_lexical import LexicalIndex
    from termmap.retrieve_training import TrainingResources

    kbs, cfg = _kbs(), _cfg(tmp_path)
    lex = {s: LexicalIndex(kb) for s, kb in kbs.items()}
    d = DenseIndex(cfg, kbs, {"name": "tiny", "model_path": str(tiny / "tiny-bi"), "batch_size": 8})
    d.context_view, d.top_k = True, 5
    fb = FeatureBuilder(cfg, kbs, lex, build_vocab(kbs, cfg), {"tiny": d})
    rows = pd.DataFrame({"entity": ["mitral stenosis"], "chunk_text": ["echo: mitral stenosis"],
                         "ent_norm": ["mitral stenosis"], "gold": ["I34.2"], "gold_valid": [True]})
    tres = TrainingResources(rows, kbs["ICD10CM"])
    q = pd.DataFrame({"qid": ["a"], "entity": ["MS"], "chunk_text": ["Echo: thickened mitral valve, MS gradient 10"]})
    f = fb.build(q, "ICD10CM", tres, {"ms": [("mitral stenosis", 1.0), ("multiple sclerosis", 0.8)]})
    assert {"dense_tiny_orig", "dense_tiny_rank_abbr", "dense_tiny_ctx", "rrf"} <= set(f.columns)
    assert f["code"].map(kbs["ICD10CM"].valid).all()


def test_cross_encoder_train_score_cache_and_merge(tiny, tmp_path):
    from termmap.cross_encoder import CrossEncoderScorer, merge_scores, train_cross_encoder

    kbs = _kbs()
    ex = [("ms ; mitral valve", "nonrheumatic mitral valve stenosis", 1.0),
          ("ms ; mitral valve", "multiple sclerosis", 0.0)] * 4
    p = train_cross_encoder(ex, tmp_path / "ce", str(tiny / "tiny-ce"), epochs=1, batch_size=4, save_every_steps=1)
    assert (Path(p) / "DONE").exists()
    sc = CrossEncoderScorer(p, kbs, top_n=2, cache_dir=tmp_path / "ce_cache.json")
    feats = pd.DataFrame({"qid": ["a", "a", "a"], "code": ["I34.2", "Q23.2", "G35"], "s0": [0.9, 0.8, 0.1]})
    q = pd.DataFrame({"qid": ["a"], "entity": ["MS"], "chunk_text": ["Echo: mitral valve MS"]})
    t1 = sc.score_feats(feats, q, "ICD10CM")
    t2 = CrossEncoderScorer(p, kbs, top_n=2, cache_dir=tmp_path / "ce_cache.json").score_feats(feats, q, "ICD10CM")
    assert len(t1) == 2 and np.allclose(t1["ce_score"], t2["ce_score"])  # cached replay identical
    m = merge_scores(feats, t1)
    assert {"ce_score", "ce_rank", "ce_gap_top"} <= set(m.columns)
    assert m.loc[m["code"] == "G35", "ce_score"].isna().all()  # outside top-N -> NaN


def test_calibration_monotone():
    from termmap.stages import TwoStage
    from termmap.util import load_config

    cfg = load_config("configs/evaluation.yaml")
    cfg["rerank"]["n_estimators"] = 20
    rng = np.random.default_rng(0)
    rows = []
    for q in range(120):
        for c in range(6):
            x = rng.normal()
            rows.append({"qid": f"q{q}", "code": f"C{c}", "f1": x + (1.5 if c == 0 else 0), "f2": rng.normal(),
                         "label": int(c == 0)})
    df = pd.DataFrame(rows)
    trf = {"ICD10CM": df, "SNOMEDCT_US": df}
    ts = TwoStage(cfg, {}, {}, 13, stage2=False, calibrate=True).fit(trf, {f"q{q}": q % 5 for q in range(120)}, 5)
    xs = np.linspace(0, 1, 50)
    ys = ts.cal["ICD10CM"].predict(xs)
    assert np.all(np.diff(ys) >= -1e-12) and ys.min() >= 0 and ys.max() <= 1


@pytest.mark.skipif(not (ROOT / "data" / "hackathon_train.csv").exists(), reason="no data")
def test_finetune_pairs_exclude_heldout_codes_and_rows(R):
    import importlib.util

    from termmap.data import SYSTEMS, diagnosis_pairs, load_train
    from termmap.util import read_json

    spec = importlib.util.spec_from_file_location("ft", ROOT / "scripts" / "finetune_encoder.py")
    ft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft)
    pairs = diagnosis_pairs(load_train(R.cfg), R.kbs)
    sp = read_json(ROOT / "reports" / "splits.json")
    cd = {s: set(sp["code_disjoint_codes"][s]) for s in SYSTEMS}
    test, val = set(sp["test_pair_ids"]), set(sp["val_pair_ids"])
    dev = pairs[~pairs["pair_id"].isin(test | val)]
    dev = dev[~(dev["gold_ICD10CM"].isin(cd["ICD10CM"]) | dev["gold_SNOMEDCT_US"].isin(cd["SNOMEDCT_US"]))]
    ids = sorted(dev["pair_id"])[:300]
    data = ft.build_pairs(R, pairs, ids, cd, 13)
    assert data and all(c not in cd[s] for *_, s, c in data)
    assert not (set(ids) & (test | val))
