"""Stage D1/D2: exact lookup, BM25 (word) and char n-gram TF-IDF over ALL dictionary descriptions.

Both channels are exact sparse dot products (no approximation) -> deterministic. Ties are broken
by document index, and document order is the sorted normalized description text.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .normalize import sorted_key
from .terminology import KB

INDEX_VERSION = "lex-v2"


def _topk_rows(S: sp.csr_matrix, k: int) -> list[list[tuple[int, float]]]:
    out = []
    S = S.tocsr()
    for i in range(S.shape[0]):
        a, b = S.indptr[i], S.indptr[i + 1]
        idx, val = S.indices[a:b], S.data[a:b]
        if len(val) == 0:
            out.append([])
            continue
        if len(val) > k:
            part = np.argpartition(-val, k - 1)[:k]
            idx, val = idx[part], val[part]
        order = np.lexsort((idx, -val))  # score desc, doc index asc
        out.append([(int(idx[j]), float(val[j])) for j in order])
    return out


class LexicalIndex:
    def __init__(self, kb: KB, k1: float = 1.2, b: float = 0.5):
        self.system = kb.system
        docs = kb.doc_text
        self.doc_codes = kb.doc_codes
        self.norm_to_doc = {t: i for i, t in enumerate(docs)}
        self.sorted_to_codes: dict[str, list[str]] = {}
        for t, codes in zip(docs, kb.doc_codes):
            self.sorted_to_codes.setdefault(sorted_key(t), [])
            for c in codes:
                if c not in self.sorted_to_codes[sorted_key(t)]:
                    self.sorted_to_codes[sorted_key(t)].append(c)
        # BM25 document weights
        self.cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
        tf = self.cv.fit_transform(docs).tocsr().astype(np.float32)
        n = tf.shape[0]
        dfreq = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float32)
        self.idf = np.log(1 + (n - dfreq + 0.5) / (dfreq + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avg = dl.mean()
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        denom = tf.data + k1 * (1 - b + b * dl[rows] / avg)
        w = tf.data * (k1 + 1) / denom * self.idf[tf.indices]
        self.bm25 = sp.csr_matrix((w.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape).T.tocsr()
        # char n-grams
        self.tv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                                  sublinear_tf=True, dtype=np.float32)
        self.char = self.tv.fit_transform(docs).T.tocsr()
        self.docs = docs

    # --- queries ---
    def exact(self, q: str) -> list[str]:
        i = self.norm_to_doc.get(q)
        return list(self.doc_codes[i]) if i is not None else []

    def exact_sorted(self, q: str) -> list[str]:
        return list(self.sorted_to_codes.get(sorted_key(q), []))

    def search_bm25(self, queries: list[str], k: int, batch: int = 256):
        out = []
        for s in range(0, len(queries), batch):
            Q = self.cv.transform(queries[s:s + batch])
            Q.data[:] = 1.0
            out.extend(_topk_rows(Q @ self.bm25, k))
        return out

    def search_char(self, queries: list[str], k: int, batch: int = 64):
        out = []
        for s in range(0, len(queries), batch):
            Q = self.tv.transform(queries[s:s + batch])
            out.extend(_topk_rows(Q @ self.char, k))
        return out

    def char_vec(self, texts: list[str]):
        return self.tv.transform(texts)


def load_or_build(kb: KB, cache_dir: Path, k1: float, b: float) -> LexicalIndex:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{kb.system}_{INDEX_VERSION}_{kb.meta.get('sha256', 'nohash')[:16]}_{k1}_{b}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    idx = LexicalIndex(kb, k1, b)
    with open(p, "wb") as f:
        pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    return idx
