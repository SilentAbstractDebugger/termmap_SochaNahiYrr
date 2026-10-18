"""Stage D4: training-derived resources, always built from a TRAINING partition only.

* alias map (normalized entity -> code)
* neighbour index (char TF-IDF over training entities; context TF-IDF for context similarity)
* training-mined abbreviations (entity token == initials of a span of the gold description)
* learned qualifier/token relevance
* priors: SNOMED semantic tag, ICD depth, ICD first letter, code frequency
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .context import build_context
from .normalize import STOPWORDS, basic_norm
from .qualifiers import Relevance
from .terminology import KB, icd_nodot


def _gold_tokens(kb: KB, code: str) -> list[str]:
    toks: list[str] = []
    for d in kb.descs_norm.get(code, []):
        toks.extend(d.split())
    return toks


def mine_training_abbr(entities, codes, kb: KB) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    for e, c in zip(entities, codes):
        if not kb.valid(c):
            continue
        etoks = basic_norm(e).split()
        for d in kb.descs_norm[c]:
            dt = d.split()
            dset = set(dt)
            cands = list(dict.fromkeys(etoks + ([basic_norm(e).replace(" ", "")] if len(etoks) <= 3 else [])))
            for t in cands:
                if t in dset or t in STOPWORDS or not t.isalpha() or not (2 <= len(t) <= 6):
                    continue
                n = len(t)
                for i in range(len(dt) - n + 1):
                    if all(dt[i + j][:1] == t[j] for j in range(n)):
                        out.setdefault(t, Counter())[" ".join(dt[i:i + n])] += 1
                        break
    return out


def mine_translation(entities, codes, kb: KB, min_count: int = 2, min_p: float = 0.3) -> dict[str, Counter]:
    """Token-level 'translation' table mined from training pairs: an entity token that never
    appears in its gold description, paired with gold-description tokens absent from the entity
    (e.g. lay word -> clinical word). Kept only if seen >= min_count times with P >= min_p."""
    from .qualifiers import token_supported

    pair_c: Counter = Counter()
    tok_c: Counter = Counter()
    for e, c in zip(entities, codes):
        if not kb.valid(c):
            continue
        etoks = [t for t in dict.fromkeys(basic_norm(e).split()) if t not in STOPWORDS and t.isalpha()]
        dtoks = list(dict.fromkeys(t for d in kb.descs_norm[c][:1] for t in d.split()
                                   if t not in STOPWORDS and t.isalpha()))
        miss_e = [t for t in etoks if not token_supported(t, dtoks)]
        miss_d = [u for u in dtoks if not token_supported(u, etoks)]
        for t in miss_e:
            tok_c[t] += 1
            for u in miss_d:
                pair_c[(t, u)] += 1
    out: dict[str, Counter] = {}
    for (t, u), n in pair_c.items():
        if n >= min_count and n / tok_c[t] >= min_p:
            out.setdefault(t, Counter())[u] = n
    return out


class TrainingResources:
    def __init__(self, rows, kb: KB, window_tokens: int = 12):
        """rows: DataFrame with entity, chunk_text, ent_norm, gold, gold_valid (one system)."""
        rows = rows[rows["gold_valid"]].reset_index(drop=True)
        self.system = kb.system
        self.n = len(rows)
        self.ent = rows["ent_norm"].tolist()
        self.codes = rows["gold"].tolist()
        self.alias: dict[str, str] = {}
        for e, c in sorted(zip(self.ent, self.codes)):
            self.alias.setdefault(e, c)
        self.code_freq = Counter(self.codes)
        self.windows = [build_context(e, ch, window_tokens).window for e, ch in
                        zip(rows["entity"], rows["chunk_text"])]
        if self.n:
            self.ent_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True,
                                           dtype=np.float32)
            self.E = self.ent_vec.fit_transform([e or "_" for e in self.ent])
            self.ctx_vec = TfidfVectorizer(token_pattern=r"(?u)\b[a-z][a-z0-9]+\b", sublinear_tf=True,
                                           stop_words=sorted(STOPWORDS), dtype=np.float32)
            try:
                self.C = self.ctx_vec.fit_transform([w or "_" for w in self.windows])
            except ValueError:
                self.ctx_vec, self.C = None, None
        self.abbr = mine_training_abbr(rows["entity"], self.codes, kb)
        self.trans = mine_translation(rows["entity"], self.codes, kb)
        self.relevance = Relevance().fit(rows["entity"], [_gold_tokens(kb, c) for c in self.codes])
        if kb.system == "SNOMEDCT_US":
            tags = Counter(kb.sem_tag.get(c, "unknown") for c in self.codes)
        else:
            tags = Counter()
        tot = max(1, sum(tags.values()))
        self.tag_prior = {t: n / tot for t, n in tags.items()}
        if kb.system == "ICD10CM":
            dep = Counter(len(icd_nodot(c)) for c in self.codes)
            let = Counter(c[0] for c in self.codes)
            self.depth_prior = {d: n / max(1, self.n) for d, n in dep.items()}
            self.letter_prior = {d: n / max(1, self.n) for d, n in let.items()}
            self.cat_freq = Counter(icd_nodot(c)[:3] for c in self.codes)
            self.leaf_rate = float(np.mean([kb.is_leaf.get(c, True) for c in self.codes])) if self.n else 1.0
        else:
            self.depth_prior, self.letter_prior, self.leaf_rate = {}, {}, 1.0
            self.cat_freq = Counter()
        # Learned conditional priors (code-agnostic coarse labels conditioned on entity tokens):
        # P(label | token) for SNOMED semantic tag, ICD first letter, ICD 'unspecified/other' leaf,
        # ICD depth. Used as candidate features ("symptom words push to finding", etc.).
        self.tok_label: dict[str, dict[str, Counter]] = {}
        if self.n:
            def lab_fns():
                if kb.system == "SNOMEDCT_US":
                    return {"tag": lambda c: kb.sem_tag.get(c, "unknown")}
                return {"letter": lambda c: c[0],
                        "unspec": lambda c: str(int(any(w in (kb.descs_norm.get(c) or [""])[0].split()
                                                        for w in ("unspecified", "nos", "other")))),
                        "depth": lambda c: str(len(icd_nodot(c)))}
            for name, fn in lab_fns().items():
                tab: dict[str, Counter] = {}
                for e, c in zip(self.ent, self.codes):
                    lab = fn(c)
                    for t in set(e.split()) - set(STOPWORDS):
                        tab.setdefault(t, Counter())[lab] += 1
                tab["__all__"] = Counter(fn(c) for c in self.codes)
                self.tok_label[name] = tab
        self.most_frequent = (sorted(self.code_freq.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                              if self.n else None)

    def label_affinity(self, name: str, tokens: list[str], label: str, alpha: float = 2.0) -> float:
        """Mean over entity tokens of smoothed P(label | token); falls back to the global prior."""
        tab = self.tok_label.get(name)
        if not tab:
            return 0.0
        g = tab["__all__"]
        gp = g[label] / max(1, sum(g.values()))
        vals = []
        for t in tokens:
            c = tab.get(t)
            if c:
                vals.append((c[label] + alpha * gp) / (sum(c.values()) + alpha))
        return float(sum(vals) / len(vals)) if vals else gp

    def neighbours(self, texts: list[str], k: int) -> list[list[tuple[int, float]]]:
        if not self.n:
            return [[] for _ in texts]
        Q = self.ent_vec.transform([t or "_" for t in texts])
        S = (Q @ self.E.T).toarray()
        out = []
        for row in S:
            kk = min(k, len(row))
            idx = np.argpartition(-row, kk - 1)[:kk]
            idx = idx[np.lexsort((idx, -row[idx]))]
            out.append([(int(i), float(row[i])) for i in idx if row[i] > 0])
        return out

    def ctx_sims(self, window: str, idxs: list[int]) -> list[float]:
        if self.ctx_vec is None or not window or not idxs:
            return [0.0] * len(idxs)
        q = self.ctx_vec.transform([window])
        return (self.C[idxs] @ q.T).toarray().ravel().tolist()
