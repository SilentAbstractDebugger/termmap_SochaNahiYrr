"""Stage F: supervised, code-agnostic reranker (LightGBM LambdaRank) + simple baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd

NON_FEATURES = {"qid", "code", "label"}

FEATURE_GROUPS = {
    "training": ("nb_", "alias_exact", "code_freq"),
    "hierarchy": ("depth", "is_leaf", "is_cat3", "n_children", "n_siblings", "last_char_", "cat_mass",
                  "cat_rank", "is_anc_of_top", "is_desc_of_top", "is_7char_ext", "ext_A", "hier_added",
                  "depth_prior", "letter_prior", "is_top", "cat_train_freq"),
    "qualifier": ("q_", "cov_rel", "drop_", "cand_extra", "unspec_desc", "other_desc"),
    "context": ("ctx_", "cue_", "hist_x", "cand_extra_in_ctx", "cand_extra_in_chunk", "chunk_overlap",
                "abbr_ctx_support"),
    "dense": ("dense_",),
    "rrf": ("rrf", "n_channels_hit"),
    "priors": ("prior_",),
    "cross_encoder": ("ce_",),
}


def feature_columns(df: pd.DataFrame, drop_groups: tuple[str, ...] = ()) -> list[str]:
    cols = [c for c in df.columns if c not in NON_FEATURES]
    for g in drop_groups:
        pref = FEATURE_GROUPS[g]
        cols = [c for c in cols if not any(c.startswith(p) for p in pref)]
    return sorted(cols)


def _groups(df: pd.DataFrame) -> np.ndarray:
    # df is sorted by qid; groupby(sort=False) keeps appearance order
    return df.groupby("qid", sort=False).size().values


class Reranker:
    def __init__(self, cfg: dict, features: list[str]):
        self.cfg, self.features = cfg, features
        self.model = None

    def fit(self, df: pd.DataFrame) -> "Reranker":
        import lightgbm as lgb

        pos = df.groupby("qid")["label"].transform("max") > 0
        d = df[pos].sort_values(["qid", "code"], kind="mergesort")
        r = self.cfg["rerank"]
        self.model = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=r["n_estimators"], learning_rate=r["learning_rate"],
            num_leaves=r["num_leaves"], min_child_samples=r["min_child_samples"],
            colsample_bytree=r["feature_fraction"], subsample=r["bagging_fraction"],
            lambdarank_truncation_level=r["lambdarank_truncation_level"],
            random_state=self.cfg["seed"], deterministic=True, force_row_wise=True,
            n_jobs=self.cfg["num_threads"], verbose=-1,
        )
        self.model.fit(d[self.features].astype(np.float32).values, d["label"].astype(int).values, group=_groups(d))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.features].astype(np.float32).values,
                                  num_threads=self.cfg["num_threads"])

    def importance(self) -> list[tuple[str, int]]:
        imp = self.model.booster_.feature_importance("gain")
        return sorted(zip(self.features, imp.round(1).tolist()), key=lambda x: -x[1])


class LogRegBaseline:
    def __init__(self, cfg: dict, features: list[str]):
        self.cfg, self.features = cfg, features

    def fit(self, df: pd.DataFrame) -> "LogRegBaseline":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        self.m.fit(df[self.features].values, df["label"].astype(int).values)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.m.decision_function(df[self.features].values)


def heuristic_scores(df: pd.DataFrame, name: str) -> np.ndarray:
    """Untrained baselines evaluated on the same candidate tables."""
    ex = df[[c for c in df.columns if c.startswith("exact_") and not c.startswith("exact_sorted")]].max(axis=1)
    if name == "exact_fuzzy":
        return (ex * 1000 + df["wratio_best"] + df["tset_best"] * 0.01).values
    if name == "bm25_only":
        return df["bm25_orig"].values
    if name == "char_only":
        return df["char_orig"].values
    if name == "hybrid_sum":
        return df["s0"].values
    if name == "rrf_only":
        return df["rrf"].values if "rrf" in df else np.zeros(len(df))
    if name == "dense_only":
        return df["dense_orig"].values if "dense_orig" in df else np.zeros(len(df))
    raise KeyError(name)


def select_top(df: pd.DataFrame, scores: np.ndarray, k: int = 3) -> pd.DataFrame:
    """Per qid: top-k by score desc, code asc (deterministic tie-break); adds softmax prob."""
    d = df[["qid", "code"]].copy()
    d["score"] = scores
    d = d.sort_values(["qid", "score", "code"], ascending=[True, False, True], kind="mergesort")
    mx = d.groupby("qid")["score"].transform("max")
    e = np.exp(d["score"] - mx)
    d["prob"] = e / e.groupby(d["qid"]).transform("sum")
    d["rank"] = d.groupby("qid").cumcount()
    return d[d["rank"] < k].reset_index(drop=True)
