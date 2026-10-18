"""Stage B: multi-view, conservative normalization.

The original string is never destroyed. Every transformation produces an *additional*
retrieval view (QueryVariant) tagged with how it was produced, so the reranker can weigh it.
Nothing here contains codes, descriptions or code mappings. The vocabulary used for typo /
glue / truncation repair is built from the dictionary files at runtime; the abbreviation
lexicon is mined from the dictionary ("ABBR - expansion" descriptions and "(ABBR)"
parentheticals) and from the TRAINING partition only (see training.py).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WS = re.compile(r"\s+")
_ALNUM_BOUNDARY = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")
_SEM_TAG = re.compile(r"\s\(([a-z][a-z /+\-]*)\)\s*$")

# Small domain-general English stopword list (function words only).
STOPWORDS = frozenset(
    "a an the of and or in on at to for with by from as is are was were be been being "
    "this that these those it its his her their your my our has have had do does did "
    "s nos".split()
)


def basic_norm(s: str | None) -> str:
    """Unicode fold, lowercase, punctuation -> space, collapse whitespace."""
    if s is None:
        return ""
    s = str(s)
    if s.lower() in ("nan", "none"):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace("&", " and ")
    s = _NON_ALNUM.sub(" ", s)
    return _WS.sub(" ", s).strip()


def strip_semantic_tag(desc: str) -> tuple[str, str | None]:
    m = _SEM_TAG.search(desc or "")
    if not m:
        return desc, None
    return desc[: m.start()].rstrip(), m.group(1)


def tokens(s: str) -> list[str]:
    return basic_norm(s).split()


def content_tokens(s: str) -> list[str]:
    return [t for t in tokens(s) if t not in STOPWORDS]


def sorted_key(s: str) -> str:
    """Order-insensitive key: sorted unique content tokens."""
    return " ".join(sorted(set(content_tokens(s))))


@dataclass
class QueryVariant:
    text: str
    kind: str  # original | typo | glue | prefix | abbr | abbr+typo
    weight: float = 1.0
    meta: dict = field(default_factory=dict)


class Vocab:
    """Token vocabulary from dictionary descriptions (plus training entities, if given)."""

    def __init__(self, counter: Counter, typo_min_len: int = 5, typo_max_distance: int = 2,
                 prefix_min_len: int = 3):
        self.freq = counter
        self.words = sorted(w for w in counter if len(w) >= 2 and not w.isdigit())
        self.word_set = set(counter)
        self.typo_min_len = typo_min_len
        self.typo_max_distance = typo_max_distance
        self.prefix_min_len = prefix_min_len
        # bucket by length for fast typo search
        self._by_len: dict[int, list[str]] = {}
        for w in self.words:
            self._by_len.setdefault(len(w), []).append(w)
        self._sorted = sorted(self.words)
        self._cache: dict[str, str | None] = {}

    def known(self, t: str) -> bool:
        return t in self.word_set or t.isdigit()

    def correct(self, t: str) -> str | None:
        """Closest vocabulary word within edit distance (frequency tie-break, then alphabetical)."""
        if t in self._cache:
            return self._cache[t]
        best = None
        if len(t) >= self.typo_min_len and not self.known(t):
            d = 1 if len(t) < 8 else self.typo_max_distance
            pool: list[str] = []
            for L in range(len(t) - d, len(t) + d + 1):
                pool.extend(self._by_len.get(L, []))
            if pool:
                res = process.extract(t, pool, scorer=Levenshtein.distance, limit=20,
                                      score_cutoff=d)
                if res:
                    md = min(r[1] for r in res)
                    cands = [r[0] for r in res if r[1] == md]
                    cands.sort(key=lambda w: (-self.freq[w], w))
                    best = cands[0]
        self._cache[t] = best
        return best

    def complete_prefix(self, t: str) -> str | None:
        """Truncation repair: most frequent vocab word that starts with t (t itself unknown)."""
        if len(t) < self.prefix_min_len or self.known(t) or t.isdigit():
            return None
        import bisect

        i = bisect.bisect_left(self._sorted, t)
        cands = []
        while i < len(self._sorted) and self._sorted[i].startswith(t):
            cands.append(self._sorted[i])
            i += 1
            if len(cands) > 400:
                break
        if not cands:
            return None
        cands.sort(key=lambda w: (-self.freq[w], w))
        return cands[0]

    def split_glued(self, t: str) -> str | None:
        """Split an unknown token into two known words, e.g. 'chronickidney'."""
        if len(t) < 7 or self.known(t):
            return None
        best, best_score = None, 0
        for i in range(3, len(t) - 2):
            a, b = t[:i], t[i:]
            if a in self.word_set and b in self.word_set and not a.isdigit() and not b.isdigit():
                sc = min(self.freq[a], self.freq[b])
                if sc > best_score:
                    best, best_score = f"{a} {b}", sc
        return best


# General OCR confusion pairs (glyph look-alikes), applied only to tokens unknown to the vocabulary,
# producing an ALTERNATIVE view. Not fitted to any data.
OCR_CONFUSIONS = (("1", "l"), ("0", "o"), ("rn", "m"), ("vv", "w"), ("cl", "d"), ("5", "s"), ("8", "b"),
                  ("l", "i"), ("ii", "u"), ("c", "e"))


def ocr_fix(t: str, vocab: "Vocab") -> str | None:
    """Single or double glyph substitution that turns an unknown token into a known one."""
    if len(t) < 3 or t.isdigit():
        return None
    cands = []
    for a, b in OCR_CONFUSIONS:
        start = 0
        while True:
            i = t.find(a, start)
            if i < 0:
                break
            cand = t[:i] + b + t[i + len(a):]
            if vocab.known(cand) and not cand.isdigit():
                cands.append(cand)
            start = i + 1
    if not cands:
        # all-occurrence replacement (e.g. 'hypertensi0n' / 'f1u1d')
        for a, b in OCR_CONFUSIONS:
            if a in t:
                cand = t.replace(a, b)
                if vocab.known(cand) and not cand.isdigit():
                    cands.append(cand)
    if not cands:
        return None
    cands.sort(key=lambda w: (-vocab.freq[w], w))
    return cands[0]


def repair_tokens(toks: list[str], vocab: Vocab) -> tuple[list[str], list[str]]:
    """Return (repaired tokens, list of repair kinds applied)."""
    out, kinds = [], []
    for t in toks:
        if vocab.known(t):
            out.append(t)
            continue
        g = vocab.split_glued(t)
        if g:
            out.extend(g.split())
            kinds.append("glue")
            continue
        o = ocr_fix(t, vocab)
        if o:
            out.append(o)
            kinds.append("ocr")
            continue
        c = vocab.correct(t)
        if c:
            out.append(c)
            kinds.append("typo")
            continue
        p = vocab.complete_prefix(t)
        if p:
            out.append(p)
            kinds.append("prefix")
            continue
        out.append(t)
    return out, kinds


def make_variants(entity: str, vocab: Vocab, abbr: dict[str, list[tuple[str, float]]],
                  max_abbr_expansions: int = 6, max_variants: int = 10) -> list[QueryVariant]:
    """Build retrieval views. Abbreviation hypotheses are kept side by side (never a global
    replacement); the reranker + context features decide among them."""
    base = basic_norm(entity)
    variants: list[QueryVariant] = [QueryVariant(base, "original", 1.0)] if base else []
    if not base:
        return variants
    # split letter/digit boundaries ("dm2" -> "dm 2", "t2dm" -> "t 2 dm") as a separate view
    # OCR glyph repair first (so 'f1uid' -> 'fluid' rather than being split at the digit)
    ocr_toks = [(ocr_fix(t, vocab) if not vocab.known(t) else None) or t for t in base.split()]
    base_ocr = " ".join(ocr_toks)
    split_s = _WS.sub(" ", _ALNUM_BOUNDARY.sub(" ", base_ocr)).strip()
    base_for_repair = split_s if split_s != base else base
    toks = base_for_repair.split()
    rep, kinds = repair_tokens(toks, vocab)
    if split_s != base_ocr:
        kinds = kinds + ["split"]
    if base_ocr != base:
        kinds = kinds + ["ocr"]
    rep_s = " ".join(rep)
    if rep_s != base:
        variants.append(QueryVariant(rep_s, "+".join(sorted(set(kinds))) or "typo", 0.9))

    # abbreviation hypotheses: whole-string first, then per token
    seen = {v.text for v in variants}

    def add(text: str, kind: str, w: float, meta: dict) -> None:
        if text and text not in seen and len(variants) < max_variants:
            seen.add(text)
            variants.append(QueryVariant(text, kind, w, meta))

    if base in abbr:
        for exp, conf in abbr[base][:max_abbr_expansions]:
            add(exp, "abbr", conf, {"abbr": base, "expansion": exp})
    toks = base.split() if base_for_repair == base else base_for_repair.split()
    for i, t in enumerate(toks):
        if t in abbr and len(toks) > 1:
            for exp, conf in abbr[t][:max_abbr_expansions]:
                new = toks[:i] + exp.split() + toks[i + 1:]
                add(" ".join(new), "abbr", conf, {"abbr": t, "expansion": exp})
    # combined: abbreviation on the repaired string
    if rep_s != base:
        rtoks = rep_s.split()
        for i, t in enumerate(rtoks):
            if t in abbr and len(rtoks) > 1:
                for exp, conf in abbr[t][:2]:
                    new = rtoks[:i] + exp.split() + rtoks[i + 1:]
                    add(" ".join(new), "abbr+typo", conf * 0.9, {"abbr": t, "expansion": exp})
    return variants
