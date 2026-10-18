"""Dense channel backend that WAS runnable offline in the build sandbox: IDF-weighted mean of
static word vectors (spaCy en_core_web_lg 3.8.0, 300-d, general-domain English) over every
dictionary description, exact flat inner-product search (numpy matmul -> deterministic).

This is a weak semantic channel compared with biomedical sentence encoders (SapBERT, gte-base,
BioLORD, MedCPT; see retrieve_dense.py), but it adds lay-term -> clinical-term similarity
("itchy" ~ "pruritus") that purely lexical channels cannot provide. Same interface as DenseIndex.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .normalize import STOPWORDS
from .util import resolve

CACHE_VERSION = "static-v1"


class StaticVectorIndex:
    def __init__(self, cfg: dict, kbs: dict):
        import spacy  # optional dependency

        d = cfg["dense"]
        self.model_name = d.get("static_model", "en_core_web_lg")
        nlp = spacy.load(self.model_name, exclude=["tagger", "parser", "ner", "lemmatizer",
                                                  "attribute_ruler", "senter", "tok2vec"])
        self.vectors = nlp.vocab.vectors
        self.strings = nlp.vocab.strings
        self.dim = self.vectors.shape[1]
        self._tok: dict[str, np.ndarray | None] = {}
        cache = resolve(cfg["resources"]["cache_dir"])
        self.docs, self.idf, self.default_idf = {}, {}, {}
        for s, kb in kbs.items():
            dfc: Counter = Counter()
            for t in kb.doc_text:
                dfc.update(set(t.split()))
            n = len(kb.doc_text)
            self.idf[s] = {t: math.log(1 + n / c) for t, c in dfc.items()}
            self.default_idf[s] = math.log(1 + n)
            p = Path(cache) / f"{CACHE_VERSION}_{s}_{self.model_name}_{kb.meta.get('sha256', '')[:16]}.npy"
            if p.exists():
                self.docs[s] = np.ascontiguousarray(np.load(p), dtype=np.float32)
            else:
                emb = self._encode_docs(kb.doc_text, s)
                np.save(p, emb)
                self.docs[s] = np.ascontiguousarray(np.load(p), dtype=np.float32)

    def _vec(self, t: str):
        if t not in self._tok:
            key = self.strings[t] if t in self.strings else None
            v = None
            if key is not None and key in self.vectors:
                v = np.asarray(self.vectors[key], dtype=np.float32)
                nv = np.linalg.norm(v)
                v = v / nv if nv > 0 else None
            self._tok[t] = v
        return self._tok[t]

    def _encode_docs(self, texts: list[str], system: str) -> np.ndarray:
        vocab = sorted({t for x in texts for t in x.split()})
        tid = {t: i for i, t in enumerate(vocab)}
        E = np.zeros((len(vocab), self.dim), dtype=np.float32)
        w = np.zeros(len(vocab), dtype=np.float32)
        for t, i in tid.items():
            v = self._vec(t)
            if v is not None and t not in STOPWORDS:
                E[i] = v
                w[i] = self.idf[system].get(t, self.default_idf[system])
        rows, cols = [], []
        for r, x in enumerate(texts):
            for t in x.split():
                rows.append(r)
                cols.append(tid[t])
        M = sp.csr_matrix((w[cols], (rows, cols)), shape=(len(texts), len(vocab)), dtype=np.float32)
        out = np.asarray(M @ E, dtype=np.float32)
        nrm = np.linalg.norm(out, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return (out / nrm).astype(np.float32)

    def encode_queries(self, texts: list[str], system: str) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, x in enumerate(texts):
            acc = np.zeros(self.dim, dtype=np.float32)
            for t in x.split():
                if t in STOPWORDS:
                    continue
                v = self._vec(t)
                if v is not None:
                    acc += v * self.idf[system].get(t, self.default_idf[system])
            n = np.linalg.norm(acc)
            out[i] = acc / n if n > 0 else acc
        return out

    def search(self, system: str, queries: list[str], k: int, batch: int = 128):
        Q = self.encode_queries(queries, system)
        D = self.docs[system]
        res = []
        for s in range(0, len(Q), batch):
            S = Q[s:s + batch] @ D.T
            kk = min(k, S.shape[1])
            idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            for r in range(S.shape[0]):
                ii = idx[r]
                vals = S[r, ii]
                if not np.any(vals > 0):
                    res.append([])
                    continue
                order = np.lexsort((ii, -vals))
                res.append([(int(ii[j]), float(vals[j])) for j in order if vals[j] > 0])
        return res
