"""Decision stages on top of the candidate feature tables.

Stage 1   per-system LightGBM LambdaRank (code-agnostic features).
Stage 2   (optional) joint pass: for every candidate, agreement between its description and the
          OTHER system's stage-1 top-1 description (char-TF-IDF cosine, token Jaccard), plus the
          other system's top-1 probability and margin. Trained on out-of-fold stage-1 scores, so
          no query sees a model that was fit on itself. Never derives one code from the other and
          never overwrites: it only adds evidence to a second ranking.
Calib     (optional) isotonic regression of top-1 correctness on the out-of-fold top-1 softmax
          probability (fit on training OOF predictions only, never on evaluation rows).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .data import SYSTEMS
from .normalize import STOPWORDS
from .rerank import Reranker, feature_columns, select_top

JOINT_COLS = ["xs_char", "xs_tok", "xs_other_prob", "xs_other_margin", "xs_has_other"]


def base_qid(q: str) -> str:
    return str(q).split("~", 1)[0]


def _softmax_top(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    t = select_top(df, scores, k=2)
    p1 = t[t["rank"] == 0].set_index("qid")
    p2 = t[t["rank"] == 1].set_index("qid")["prob"].reindex(p1.index).fillna(0.0)
    return pd.DataFrame({"code": p1["code"], "prob": p1["prob"], "margin": p1["prob"] - p2})


def oof_scores(cfg: dict, df: pd.DataFrame, features: list[str], fold_of: dict, n_folds: int) -> np.ndarray:
    """Out-of-fold reranker scores for a training table (fold = fold of the base pair id)."""
    folds = df["qid"].map(lambda q: fold_of.get(base_qid(q), 0) % n_folds).values
    out = np.zeros(len(df), dtype=np.float64)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if not te.any() or not tr.any():
            continue
        m = Reranker(cfg, features).fit(df[tr])
        out[te] = m.score(df[te])
    return out


class JointFeaturizer:
    def __init__(self, kbs: dict, lex: dict):
        self.kbs, self.lex = kbs, lex
        self._vec: dict = {}

    def bind(self, kbs: dict, lex: dict) -> "JointFeaturizer":
        self.kbs, self.lex, self._vec = kbs, lex, {}
        return self

    def __getstate__(self):
        # never pickle the dictionaries / indexes / vector cache with the model (they are re-bound
        # from the loaded resources at prediction time)
        return {"kbs": None, "lex": None, "_vec": {}}

    def _code_text(self, s: str, c: str) -> str:
        return " ".join(self.kbs[s].descs_norm.get(c, [])[:3]) or "_"

    def _vectors(self, s: str, codes) -> dict:
        todo = sorted({c for c in codes if (s, c) not in self._vec})
        if todo:
            V = self.lex["SNOMEDCT_US"].char_vec([self._code_text(s, c) for c in todo])
            for i, c in enumerate(todo):
                self._vec[(s, c)] = V[i]
        return {c: self._vec[(s, c)] for c in codes}

    def add(self, feats: dict, scores: dict) -> dict:
        """feats/scores per system -> copies of feats with JOINT_COLS added."""
        tops = {s: _softmax_top(feats[s], scores[s]) if len(feats[s]) else None for s in SYSTEMS}
        out = {}
        for s in SYSTEMS:
            o = "SNOMEDCT_US" if s == "ICD10CM" else "ICD10CM"
            df = feats[s].copy()
            if not len(df):
                out[s] = df
                continue
            ot = tops[o]
            oc = df["qid"].map(ot["code"]) if ot is not None else pd.Series(np.nan, index=df.index)
            has = oc.notna()
            df["xs_has_other"] = has.astype(np.float32)
            df["xs_other_prob"] = df["qid"].map(ot["prob"]).astype(np.float32) if ot is not None else np.nan
            df["xs_other_margin"] = df["qid"].map(ot["margin"]).astype(np.float32) if ot is not None else np.nan
            cand_v = self._vectors(s, df["code"].unique())
            oth_v = self._vectors(o, oc[has].unique())
            xs = np.full(len(df), np.nan, dtype=np.float32)
            xt = np.full(len(df), np.nan, dtype=np.float32)
            tok_cache: dict = {}

            def toks(sys_, c):
                k = (sys_, c)
                if k not in tok_cache:
                    tok_cache[k] = set(self._code_text(sys_, c).split()) - set(STOPWORDS)
                return tok_cache[k]

            idx = np.where(has.values)[0]
            codes, ocs = df["code"].values, oc.values
            for i in idx:
                a, b = cand_v[codes[i]], oth_v[ocs[i]]
                xs[i] = float(a.multiply(b).sum())
                ta, tb = toks(s, codes[i]), toks(o, ocs[i])
                xt[i] = len(ta & tb) / max(1, len(ta | tb))
            df["xs_char"], df["xs_tok"] = xs, xt
            out[s] = df
        return out


class TwoStage:
    """Fit/apply stage-1 (+ optional stage-2 and calibration) for both systems."""

    def __init__(self, cfg: dict, kbs: dict, lex: dict, seed: int, stage2, calibrate: bool,
                 drop_groups=()):
        """stage2: False | True (both systems) | iterable of systems that use the second stage."""
        self.cfg = copy.deepcopy(cfg)
        self.cfg["seed"] = seed
        self.s2_systems = (set(SYSTEMS) if stage2 is True else set(stage2) if stage2 else set())
        self.stage2 = bool(self.s2_systems)
        self.calibrate, self.drop_groups = calibrate, drop_groups
        self.jf = JointFeaturizer(kbs, lex)
        self.m1, self.m2, self.cal, self.f1, self.f2 = {}, {}, {}, {}, {}

    def fit(self, trf: dict, fold_of: dict, n_folds: int) -> "TwoStage":
        need_oof = self.stage2 or self.calibrate
        oof1 = {}
        for s in SYSTEMS:
            self.f1[s] = feature_columns(trf[s], self.drop_groups)
            self.m1[s] = Reranker(self.cfg, self.f1[s]).fit(trf[s])
            if need_oof:
                oof1[s] = oof_scores(self.cfg, trf[s], self.f1[s], fold_of, n_folds)
        final_oof = oof1
        if self.stage2:
            tr2 = self.jf.add(trf, oof1)
            final_oof = dict(oof1)
            for s in sorted(self.s2_systems):
                self.f2[s] = self.f1[s] + JOINT_COLS
                self.m2[s] = Reranker(self.cfg, self.f2[s]).fit(tr2[s])
                if self.calibrate:
                    final_oof[s] = oof_scores(self.cfg, tr2[s], self.f2[s], fold_of, n_folds)
        if self.calibrate:
            from sklearn.isotonic import IsotonicRegression

            for s in SYSTEMS:
                t = select_top(trf[s], final_oof[s], k=1)
                lab = trf[s].set_index(["qid", "code"])["label"]
                y = np.array([lab.get((q, c), 0) for q, c in zip(t["qid"], t["code"])], dtype=float)
                self.cal[s] = IsotonicRegression(out_of_bounds="clip", increasing=True, y_min=0, y_max=1).fit(
                    t["prob"].values, y)
        return self

    def scores(self, evf: dict) -> dict:
        sc = {s: self.m1[s].score(evf[s]) if len(evf[s]) else np.zeros(0) for s in SYSTEMS}
        if self.stage2:
            ev2 = self.jf.add(evf, sc)
            sc = {s: (self.m2[s].score(ev2[s]) if len(ev2[s]) else np.zeros(0)) if s in self.m2 else sc[s]
                  for s in SYSTEMS}
        return sc

    def predict(self, evf: dict, k: int = 3) -> dict:
        sc = self.scores(evf)
        out = {}
        for s in SYSTEMS:
            if not len(evf[s]):
                out[s] = pd.DataFrame(columns=["qid", "code", "score", "prob", "rank", "prob_raw"])
                continue
            t = select_top(evf[s], sc[s], k=k)
            t["prob_raw"] = t["prob"]
            if s in self.cal:
                t["prob"] = self.cal[s].predict(t["prob_raw"].values)
            out[s] = t
        return out
