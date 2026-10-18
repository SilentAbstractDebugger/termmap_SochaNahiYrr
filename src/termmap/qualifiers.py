"""Qualifier extraction + learned qualifier relevance.

Extraction uses generic English/clinical qualifier words (laterality, acuity, severity, stage,
type, encounter). It never maps to codes. RELEVANCE is learned from training: for each entity
token, P(token survives into the gold description | token in entity), smoothed toward a prior.
Tokens like "well"/"controlled" get low relevance, so a candidate that lacks them is not
penalised as under-coding.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from rapidfuzz import fuzz

from .normalize import STOPWORDS, basic_norm


def is_acronym_of(t: str, gtoks: list[str]) -> bool:
    """True if t equals the initials of a contiguous span of gtoks (len >= 2)."""
    n = len(t)
    if n < 2 or n > 6 or not t.isalpha():
        return False
    for i in range(len(gtoks) - n + 1):
        if all(gtoks[i + j][:1] == t[j] for j in range(n)):
            return True
    return False


def token_supported(t: str, gtoks: list[str]) -> bool:
    if t in gtoks:
        return True
    if len(t) >= 4 and any(fuzz.ratio(t, x) >= 80 or (len(x) > len(t) and x.startswith(t))
                           for x in gtoks):
        return True
    if is_acronym_of(t, gtoks):
        return True
    # contraction-style abbreviation: same first letter and an in-order subsequence (htn ~ hypertension)
    if 2 < len(t) <= 5 and t.isalpha():
        for x in gtoks:
            if len(x) > len(t) + 1 and x[0] == t[0] and _is_subseq(t, x):
                return True
    return False


def _is_subseq(a: str, b: str) -> bool:
    it = iter(b)
    return all(ch in it for ch in a)

LATERALITY = {
    "left": "L", "lt": "L", "l": "L", "right": "R", "rt": "R", "r": "R",
    "bilateral": "B", "bilat": "B", "both": "B", "bilaterally": "B", "b l": "B",
}
ACUITY = {"acute": "A", "chronic": "C", "subacute": "S", "acutely": "A", "chronically": "C",
          "acute on chronic": "AC"}
SEVERITY = {"mild": "1", "moderate": "2", "severe": "3", "mod": "2", "mildly": "1", "severely": "3",
            "minimal": "1", "marked": "3"}
ORIGIN = {"congenital": "C", "acquired": "A", "hereditary": "C", "familial": "C", "inherited": "C",
          "developmental": "C", "nonrheumatic": "N", "rheumatic": "R"}
ENCOUNTER = {"initial": "A", "subsequent": "D", "sequela": "S"}
_STAGE = re.compile(r"\bstage\s*([0-9]{1,2}|i{1,3}v?|iv|v)\b")
_TYPE = re.compile(r"\btype\s*([0-9]|i{1,2})\b")
_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}
_DEGREE = re.compile(r"(\d)\s*\+|\+\s*(\d)")


@dataclass
class Qual:
    lat: str | None
    acu: str | None
    sev: str | None
    stage: str | None
    typ: str | None
    enc: str | None
    org: str | None = None   # origin: congenital / acquired / rheumatic / nonrheumatic


def _first(text: str, table: dict) -> str | None:
    p = f" {text} "
    # longest phrase first
    for k in sorted(table, key=len, reverse=True):
        if f" {k} " in p:
            return table[k]
    return None


def extract(text: str) -> Qual:
    t = basic_norm(text)
    st = _STAGE.search(t)
    ty = _TYPE.search(t)
    stage = _ROMAN.get(st.group(1), st.group(1)) if st else None
    typ = _ROMAN.get(ty.group(1), ty.group(1)) if ty else None
    lat = _first(t, LATERALITY)
    if lat is None and (" left " in f" {t} " and " right " in f" {t} "):
        lat = "B"
    return Qual(lat, _first(t, ACUITY), _first(t, SEVERITY), stage, typ, _first(t, ENCOUNTER), _first(t, ORIGIN))


def compare(q: Qual, c: Qual) -> dict[str, float]:
    """Match / contradiction flags between text qualifiers q and candidate qualifiers c."""
    f = {}
    for name in ("lat", "acu", "sev", "stage", "typ", "enc", "org"):
        a, b = getattr(q, name), getattr(c, name)
        f[f"q_{name}_match"] = float(a is not None and a == b)
        f[f"q_{name}_contra"] = float(a is not None and b is not None and a != b)
        f[f"q_{name}_cand_adds"] = float(a is None and b is not None)
        f[f"q_{name}_text_only"] = float(a is not None and b is None)
    return f


class Relevance:
    """Learned token relevance: P(entity token appears in gold description tokens)."""

    def __init__(self, prior: float = 0.6, strength: float = 2.0):
        self.prior, self.strength = prior, strength
        self.kept: Counter = Counter()
        self.seen: Counter = Counter()

    def fit(self, entities, gold_desc_tokens) -> "Relevance":
        for e, gtoks in zip(entities, gold_desc_tokens):
            g = list(dict.fromkeys(gtoks))
            for t in set(basic_norm(e).split()):
                if t in STOPWORDS:
                    continue
                self.seen[t] += 1
                self.kept[t] += token_supported(t, g)
        return self

    def weight(self, t: str) -> float:
        if t in STOPWORDS:
            return 0.0
        return (self.kept[t] + self.prior * self.strength) / (self.seen[t] + self.strength)

    def table(self, min_seen: int = 3) -> list[tuple[str, float, int]]:
        rows = [(t, round(self.weight(t), 3), n) for t, n in self.seen.items() if n >= min_seen]
        return sorted(rows, key=lambda r: (r[1], -r[2], r[0]))
