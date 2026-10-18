"""Leakage-proof splits.

(a) group split: pairs connected by the same normalized entity OR the same non-empty chunk_text
    form one group (union-find); 80/20 over groups, seeded.
(b) code-disjoint fold per system: a seeded random sample of gold codes; every row with such a
    code is evaluation-only for that system and removed from all training-derived resources.
(c) OOF folds over the training partition for reranker feature generation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SYSTEMS
from .util import stable_rng, text_sha256


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def group_ids(pairs: pd.DataFrame) -> np.ndarray:
    uf = _UF(len(pairs))
    for col in ("ent_norm", "chunk_text"):
        first: dict[str, int] = {}
        for i, v in enumerate(pairs[col].values):
            if col == "chunk_text" and not str(v).strip():
                continue
            if v in first:
                uf.union(first[v], i)
            else:
                first[v] = i
    return np.array([uf.find(i) for i in range(len(pairs))])


def make_splits(pairs: pd.DataFrame, seed: int, val_fraction: float, cd_fraction: float,
                oof_folds: int, test_fraction: float = 0.0) -> dict:
    g = group_ids(pairs)
    uniq = np.unique(g)
    rng = stable_rng(seed, "group-split")
    perm = rng.permutation(uniq)
    n_val_target = int(round(val_fraction * len(pairs)))
    sizes = pd.Series(g).value_counts()
    val_groups, n = set(), 0
    for grp in perm:
        if n >= n_val_target:
            break
        val_groups.add(int(grp))
        n += int(sizes[grp])
    is_val = np.array([int(x) in val_groups for x in g])
    # FINAL TEST split: carved from the non-validation groups with an independent seeded stream,
    # so the validation split (and the earlier baseline) stays identical. Never used for tuning.
    test_groups, n = set(), 0
    n_test_target = int(round(test_fraction * len(pairs)))
    for grp in stable_rng(seed, "final-test").permutation(uniq):
        if n >= n_test_target:
            break
        if int(grp) in val_groups:
            continue
        test_groups.add(int(grp))
        n += int(sizes[grp])
    is_test = np.array([int(x) in test_groups for x in g])
    pid = pairs["pair_id"].values
    train_ids = sorted(pid[~is_val & ~is_test].tolist())
    val_ids = sorted(pid[is_val].tolist())
    test_ids = sorted(pid[is_test].tolist())
    is_val = is_val | is_test  # excluded from the training-partition OOF folds below

    # OOF folds over training groups
    tr_groups = sorted({int(x) for x, v in zip(g, is_val) if not v})
    frng = stable_rng(seed, "oof")
    fold_of_group = {grp: int(i % oof_folds) for i, grp in enumerate(frng.permutation(tr_groups))}
    oof = {p: fold_of_group[int(x)] for p, x, v in zip(pid, g, is_val) if not v}

    # OOF folds over ALL pairs (used for the code-disjoint experiment and the final refit)
    arng = stable_rng(seed, "oof-all")
    fold_all_g = {int(grp): int(i % oof_folds) for i, grp in enumerate(arng.permutation(uniq))}
    oof_all = {p: fold_all_g[int(x)] for p, x in zip(pid, g)}
    # OOF folds over all NON-test pairs (development: code-disjoint experiment, train+val refit)
    dev_groups = sorted({int(x) for x, t in zip(g, is_test) if not t})
    drng = stable_rng(seed, "oof-dev")
    fold_dev_g = {grp: int(i % oof_folds) for i, grp in enumerate(drng.permutation(dev_groups))}
    oof_dev = {p: fold_dev_g[int(x)] for p, x, t in zip(pid, g, is_test) if not t}

    # code-disjoint codes per system (sampled over distinct gold codes)
    cd = {}
    for s in SYSTEMS:
        codes = sorted({c for c in pairs[f"gold_{s}"] if c})
        crng = stable_rng(seed, "code-disjoint", s)
        k = int(round(cd_fraction * len(codes)))
        cd[s] = sorted(crng.choice(codes, size=k, replace=False).tolist())
    return {
        "seed": seed,
        "data_hash": text_sha256("\n".join(pairs["pair_id"] + "\t" + pairs["entity"] + "\t" +
                                           pairs["chunk_text"])),
        "n_groups": int(len(uniq)),
        "largest_group": int(sizes.max()),
        "train_pair_ids": train_ids,
        "val_pair_ids": val_ids,
        "test_pair_ids": test_ids,
        "oof_fold": oof,
        "oof_fold_dev": oof_dev,
        "oof_fold_all": oof_all,
        "code_disjoint_codes": cd,
    }
