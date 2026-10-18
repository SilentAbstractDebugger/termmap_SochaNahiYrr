"""End-to-end orchestration: resources, experiments (val / code-disjoint / final refit), predictor."""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .context import clean_chunk
from .data import SYSTEMS, diagnosis_pairs, load_train, system_rows
from .features import FeatureBuilder, build_vocab, merge_abbr
from .rerank import Reranker, feature_columns, select_top
from .retrieve_lexical import load_or_build
from .retrieve_training import TrainingResources
from .terminology import load_kbs
from .util import ROOT, resolve, set_seeds, text_sha256

MODEL_VERSION = "termmap-model-v1"


@dataclass
class Resources:
    cfg: dict
    kbs: dict
    lex: dict
    vocab: object
    fb: FeatureBuilder
    mode: dict = field(default_factory=dict)


def load_resources(cfg: dict, icd_path=None, snomed_path=None, kbs: dict | None = None,
                   lex: dict | None = None) -> Resources:
    """kbs / lex may be passed in to reuse already-loaded components (tests, drift checks)."""
    set_seeds(cfg["seed"])
    kbs = kbs or load_kbs(cfg, icd_path, snomed_path)
    cache = resolve(cfg["resources"]["cache_dir"])
    r = cfg["retrieval"]
    lex = lex or {s: load_or_build(kb, cache, r["bm25_k1"], r["bm25_b"]) for s, kb in kbs.items()}
    vocab = build_vocab(kbs, cfg)
    dense, mode = {}, {"lexical": True, "training_neighbours": True, "reranker": True}
    dcfg = cfg.get("dense", {})
    if dcfg.get("enabled"):
        try:
            if dcfg.get("backend", "sentence_transformer") == "static_vectors":
                from .retrieve_static import StaticVectorIndex

                dense[""] = StaticVectorIndex(cfg, kbs)
                mode["dense"] = f"static_vectors:{dcfg.get('static_model')}"
            else:
                from .retrieve_dense import DenseIndex

                dense[""] = DenseIndex(cfg, kbs)
                mode["dense"] = dcfg.get("model_id")
        except Exception as e:  # graceful degradation
            mode["dense"] = f"unavailable ({type(e).__name__}: {e})"
    else:
        mode["dense"] = "disabled"
    # additional sentence encoders (Step 1): dense.models = [{name, model_path, model_id, ...}]
    for spec in dcfg.get("models", []) or []:
        if not spec.get("enabled", True):
            continue
        try:
            from .retrieve_dense import DenseIndex

            idx = DenseIndex(cfg, kbs, spec)
            idx.context_view = bool(spec.get("context_view", True))
            idx.top_k = int(spec.get("top_k", dcfg.get("top_k", 30)))
            dense[spec["name"]] = idx
            mode[f"dense_{spec['name']}"] = f"{spec.get('model_id')}@{spec.get('revision', 'unpinned')}"
        except Exception as e:
            mode[f"dense_{spec['name']}"] = f"unavailable ({type(e).__name__}: {e})"
    xw = resolve(cfg["resources"].get("crosswalk_path"))
    mode["crosswalk"] = str(xw) if xw and xw.exists() else "not supplied"
    rel = resolve(cfg["resources"].get("snomed_relationships_path"))
    mode["snomed_hierarchy"] = str(rel) if rel and rel.exists() else "not supplied (no IS-A features)"
    fb = FeatureBuilder(cfg, kbs, lex, vocab, dense or None)
    ce = cfg.get("cross_encoder", {})
    if ce.get("enabled"):
        try:
            from .cross_encoder import CrossEncoderScorer

            fb.ce = CrossEncoderScorer(resolve(ce["model_path"]), kbs, top_n=int(ce.get("top_n", 30)),
                                       batch_size=int(ce.get("batch_size", 64)))
            mode["cross_encoder"] = str(ce["model_path"])
        except Exception as e:
            mode["cross_encoder"] = f"unavailable ({type(e).__name__}: {e})"
    else:
        mode["cross_encoder"] = "disabled"
    return Resources(cfg, kbs, lex, vocab, fb, mode)


def as_queries(rows: pd.DataFrame) -> pd.DataFrame:
    q = rows.rename(columns={"pair_id": "qid"})[["qid", "entity", "chunk_text"]].copy()
    # empty entity: fall back to the start of the cleaned chunk (documented Stage I behaviour)
    empty = q["entity"].str.strip() == ""
    if empty.any():
        q.loc[empty, "entity"] = [" ".join(clean_chunk(c).split()[:8]) for c in q.loc[empty, "chunk_text"]]
    return q


def _tres_pair(R: Resources, pairs_tr: pd.DataFrame) -> dict:
    return {s: TrainingResources(system_rows(pairs_tr, s), R.kbs[s], R.cfg["context"]["window_tokens"])
            for s in SYSTEMS}


FEATURE_VERSION = "feat-v7"  # bump whenever candidate generation / features change


def _cache_path(R: Resources, tag: str, *parts) -> Path:
    import hashlib
    import json

    sub = {k: R.cfg.get(k) for k in ("normalize", "context", "retrieval", "dense", "augment")}
    if isinstance(sub.get("dense"), dict) and not sub["dense"].get("models"):
        sub["dense"] = {k: v for k, v in sub["dense"].items() if k != "models"}  # empty list == absent
    dense_mode = {k: v for k, v in sorted(R.mode.items()) if k.startswith("dense")}
    h = hashlib.sha256(json.dumps([FEATURE_VERSION, sub, R.cfg["seed"], tag, parts, dense_mode,
                                   {s: kb.meta.get("sha256") for s, kb in R.kbs.items()}],
                                  sort_keys=True, default=str).encode()).hexdigest()[:20]
    d = resolve(R.cfg.get("feature_cache_dir", "models/cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{tag}_{h}.pkl"


def _cached(path: Path, fn):
    """Resumable stage: load if the hash-keyed file exists, else compute and write atomically."""
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    obj = fn()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return obj


def augment_rows(rows: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Train-time noise augmentation: corrupted copies of a seeded fraction of rows (typo, deletion,
    glueing, truncation, OCR confusion, punctuation). qid suffix '~aug'. Gold unchanged."""
    from .util import stable_rng

    if frac <= 0 or not len(rows):
        return rows.iloc[0:0]
    out = []
    for r in rows.itertuples(index=False):
        rng = stable_rng(seed, "augment", r.pair_id)
        if rng.random() >= frac:
            continue
        e = r.entity
        noisy = _corrupt(e, rng)
        if noisy == e or not noisy.strip():
            continue
        ch = r.chunk_text.replace(e, noisy) if e and e in r.chunk_text else r.chunk_text
        d = r._asdict()
        d.update(pair_id=f"{r.pair_id}~aug", entity=noisy, chunk_text=ch)
        out.append(d)
    return pd.DataFrame(out, columns=rows.columns)


_OCR = (("l", "1"), ("o", "0"), ("m", "rn"), ("s", "5"), ("i", "l"), ("e", "c"))


def _corrupt(e: str, rng) -> str:
    kind = int(rng.integers(0, 6))
    toks = e.split()
    if kind == 0 and len(e) > 4:            # swap adjacent chars
        i = int(rng.integers(1, len(e) - 2))
        return e[:i] + e[i + 1] + e[i] + e[i + 2:]
    if kind == 1 and len(e) > 4:            # delete a char
        i = int(rng.integers(1, len(e) - 1))
        return e[:i] + e[i + 1:]
    if kind == 2 and len(toks) > 1:         # glue two tokens
        i = int(rng.integers(0, len(toks) - 1))
        return " ".join(toks[:i] + [toks[i] + toks[i + 1]] + toks[i + 2:])
    if kind == 3 and len(toks[-1]) > 5:     # truncate last token
        return " ".join(toks[:-1] + [toks[-1][: max(3, len(toks[-1]) - 2)]])
    if kind == 4:                           # OCR confusion
        for a, b in _OCR:
            if a in e:
                return e.replace(a, b, 1)
    return e.replace(" ", "  ").replace(",", " ,") + "."   # punctuation/spacing noise


def build_oof_features(R: Resources, pairs_tr: pd.DataFrame, fold_of: dict, dropout: float,
                       aug_frac: float | None = None) -> dict:
    """Reranker training tables: each training pair gets features from resources built on the
    OTHER folds (no self-neighbour), plus code dropout and optional noise-augmented copies.
    Each fold is cached on disk (hash-keyed) so an interrupted run resumes."""
    aug_frac = R.cfg.get("augment", {}).get("frac", 0.0) if aug_frac is None else aug_frac
    folds = sorted({fold_of[p] for p in pairs_tr["pair_id"]})
    out = {s: [] for s in SYSTEMS}
    for f in folds:
        mask = pairs_tr["pair_id"].map(fold_of) == f
        tr, ho = pairs_tr[~mask], pairs_tr[mask]
        path = _cache_path(R, "ooffold", sorted(tr["pair_id"]), sorted(ho["pair_id"]), dropout, aug_frac)

        def compute(tr=tr, ho=ho):
            tres = _tres_pair(R, tr)
            abbr = merge_abbr(R.kbs, list(tres.values()))
            res = {}
            for s in SYSTEMS:
                rows = system_rows(ho, s)
                rows = rows[rows["gold_valid"]]
                rows = pd.concat([rows, augment_rows(rows, aug_frac, R.cfg["seed"])], ignore_index=True)
                if not len(rows):
                    res[s] = pd.DataFrame()
                    continue
                res[s] = R.fb.build(as_queries(rows), s, tres[s], abbr, gold=dict(zip(rows.pair_id, rows.gold)),
                                    dropout=dropout, seed=R.cfg["seed"])
            return res

        t0 = time.time()
        res = _cached(path, compute)
        R.fb.clear_caches()
        print(f"  [features] fold {f + 1}/{len(folds)} ready ({time.time() - t0:.0f}s) {path.name}", flush=True)
        for s in SYSTEMS:
            if len(res[s]):
                out[s].append(res[s])
    return {s: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for s, v in out.items()}


def build_eval_features(R: Resources, pairs_tr: pd.DataFrame, pairs_ev: pd.DataFrame,
                        systems=SYSTEMS, cd_codes: dict | None = None) -> tuple[dict, dict, dict]:
    tres = _tres_pair(R, pairs_tr)
    abbr = merge_abbr(R.kbs, list(tres.values()))
    feats, rows_by = {}, {}
    for s in systems:
        rows = system_rows(pairs_ev, s)
        if cd_codes is not None:
            rows = rows[rows["gold"].isin(set(cd_codes[s]))]
        rows_by[s] = rows.reset_index(drop=True)
        path = _cache_path(R, f"eval_{s}", sorted(pairs_tr["pair_id"]), sorted(rows["pair_id"]))
        feats[s] = _cached(path, lambda rows=rows, s=s: R.fb.build(
            as_queries(rows), s, tres[s], abbr, gold=dict(zip(rows.pair_id, rows.gold))))
        R.fb.clear_caches()
    return feats, rows_by, tres


def train_rerankers(R: Resources, train_feats: dict, drop_groups=()) -> dict:
    models = {}
    for s in SYSTEMS:
        df = train_feats[s]
        models[s] = Reranker(R.cfg, feature_columns(df, drop_groups)).fit(df)
    return models


def run_experiment(R: Resources, pairs: pd.DataFrame, train_ids, eval_ids, fold_of, *,
                   dropout: float, cd_codes: dict | None = None, drop_groups=(),
                   cache: dict | None = None) -> dict:
    """Returns per-system dict with eval rows, eval feats, top-3 predictions, models, train feats."""
    tr = pairs[pairs["pair_id"].isin(set(train_ids))].reset_index(drop=True)
    ev = pairs[pairs["pair_id"].isin(set(eval_ids))].reset_index(drop=True)
    t0 = time.time()
    key = ("train", tuple(sorted(train_ids)), dropout)
    if cache is not None and key in cache:
        trf = cache[key]
    else:
        trf = build_oof_features(R, tr, fold_of, dropout)
        if cache is not None:
            cache[key] = trf
    key2 = ("eval", tuple(sorted(train_ids)), tuple(sorted(eval_ids)), cd_codes is not None)
    if cache is not None and key2 in cache:
        evf, evrows, tres = cache[key2]
    else:
        evf, evrows, tres = build_eval_features(R, tr, ev, cd_codes=cd_codes)
        if cache is not None:
            cache[key2] = (evf, evrows, tres)
    t1 = time.time()
    models = train_rerankers(R, trf, drop_groups)
    out = {"timing_features_s": round(t1 - t0, 1), "systems": {}}
    for s in SYSTEMS:
        f = evf[s]
        top = select_top(f, models[s].score(f)) if len(f) else pd.DataFrame(columns=["qid", "code", "score", "prob", "rank"])
        out["systems"][s] = {"rows": evrows[s], "feats": f, "top": top, "model": models[s],
                             "train_feats": trf[s], "tres": tres[s]}
    return out


# ---------------------------------------------------------------------- final model + predictor
def fit_final(R: Resources, pairs: pd.DataFrame, fold_of: dict, dropout: float) -> dict:
    trf = build_oof_features(R, pairs, fold_of, dropout)
    models = train_rerankers(R, trf)
    tres = _tres_pair(R, pairs)
    return {"version": MODEL_VERSION, "models": {s: models[s].model for s in SYSTEMS},
            "features": {s: models[s].features for s in SYSTEMS}, "tres": tres,
            "train_hash": text_sha256("\n".join(pairs["pair_id"] + pairs["entity"] + pairs["gold_ICD10CM"] +
                                                pairs["gold_SNOMEDCT_US"])),
            "kb_hashes": {s: R.kbs[s].meta.get("sha256") for s in SYSTEMS},
            "config_seed": R.cfg["seed"]}


def save_model(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


class Predictor:
    """Batch predictor: queries -> both systems' top-3 with probabilities."""

    def __init__(self, R: Resources, model: dict):
        self.R, self.model = R, model
        self.abbr = merge_abbr(R.kbs, list(model["tres"].values()))
        self.low_support: dict = {}

    def predict(self, queries: pd.DataFrame) -> dict[str, pd.DataFrame]:
        q = queries[["qid", "entity", "chunk_text"]].copy()
        empty = q["entity"].str.strip() == ""
        if empty.any():
            q.loc[empty, "entity"] = [" ".join(clean_chunk(c).split()[:8]) for c in q.loc[empty, "chunk_text"]]
        feats = {s: self.R.fb.build(q, s, self.model["tres"][s], self.abbr) for s in SYSTEMS}
        if self.R.fb.ce is not None:  # cross-encoder features for the top-N candidates
            from .cross_encoder import merge_scores

            feats = {s: merge_scores(f, self.R.fb.ce.score_feats(f, q, s)) if len(f) else f for s, f in feats.items()}
        ts = self.model.get("two_stage")
        if ts is not None:
            if ts.jf.kbs is None:
                ts.jf.bind(self.R.kbs, self.R.lex)
            for s in SYSTEMS:
                f = feats[s]
                missing = [c for c in ts.f1[s] if c not in f] if len(f) else []
                if missing:  # tolerate feature drift (e.g. dense channel unavailable)
                    feats[s] = pd.concat([f, pd.DataFrame(0.0, index=f.index, columns=missing)], axis=1)
            tops = ts.predict(feats)
        else:  # legacy single-stage model dict
            tops = {}
            for s in SYSTEMS:
                f = feats[s]
                if not len(f):
                    tops[s] = pd.DataFrame(columns=["qid", "code", "score", "prob", "rank"])
                    continue
                missing = [c for c in self.model["features"][s] if c not in f]
                if missing:
                    f = pd.concat([f, pd.DataFrame(0.0, index=f.index, columns=missing)], axis=1)
                sc = self.model["models"][s].predict(f[self.model["features"][s]].astype(np.float32).values,
                                                     num_threads=self.R.cfg["num_threads"])
                tops[s] = select_top(f, sc)
        out = {}
        for s in SYSTEMS:
            top, f = tops[s], feats[s]
            if len(top) and len(f):
                t1 = top[top["rank"] == 0][["qid", "code"]].merge(f, on=["qid", "code"], how="left")
                ex = t1[[c for c in t1.columns if c.startswith("exact_")]].max(axis=1)
                ls = (ex == 0) & (t1["tset_best"] < 70) & (t1["nb_max"] < 0.5) & (t1["alias_exact"] == 0)
                self.low_support[s] = dict(zip(t1["qid"], ls.astype(bool)))
            else:
                top = pd.DataFrame(columns=["qid", "code", "score", "prob", "rank"])
                self.low_support[s] = {}
            out[s] = top
        return out
