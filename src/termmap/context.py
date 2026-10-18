"""Stage C: context builder from chunk_text.

Strips table markup, locates the entity mention (exact, then fuzzy alignment), returns a token
window around it and scope cues (negation / history / uncertainty) in a short window before the
mention. Cue lexicons are domain-general clinical-NLP trigger words (NegEx-style), not fitted to
any data; what gold coding DOES with these cues is learned by the reranker from training.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from .normalize import basic_norm

_MARKUP = re.compile(r"header_\d+\s*:|::|<br\s*/?>|<[^>]{1,20}>", re.I)

NEGATION = ("no", "denies", "denied", "deny", "without", "negative for", "not", "absent", "free of",
            "ruled out", "neg")
HISTORY = ("h o", "history of", "hx", "hx of", "s p", "status post", "past medical history", "pmh",
           "previous", "prior", "resolved", "remote")
UNCERTAIN = ("r o", "rule out", "possible", "possibly", "probable", "suspected", "suspect",
             "likely", "concern for", "questionable", "cannot exclude", "vs")
FAMILY = ("family history", "fh", "mother", "father", "sister", "brother")


@dataclass
class Context:
    clean: str                 # normalized cleaned chunk
    window: str                # normalized tokens around the mention (mention excluded)
    found: str                 # exact | fuzzy | none | empty
    negated: bool
    history: bool
    uncertain: bool
    family: bool
    pre: str                   # tokens immediately preceding the mention


def clean_chunk(chunk: str) -> str:
    return basic_norm(_MARKUP.sub(" ", chunk or ""))


def _has_cue(pre: str, cues) -> bool:
    p = f" {pre} "
    return any(f" {c} " in p for c in cues)


def build_context(entity: str, chunk: str, window_tokens: int = 12) -> Context:
    ent = basic_norm(entity)
    clean = clean_chunk(chunk)
    if not clean:
        return Context("", "", "empty", False, False, False, False, "")
    ctoks = clean.split()
    etoks = ent.split()
    start = end = -1
    found = "none"
    if etoks:
        n = len(etoks)
        for i in range(len(ctoks) - n + 1):
            if ctoks[i:i + n] == etoks:
                start, end, found = i, i + n, "exact"
                break
        if start < 0:
            al = fuzz.partial_ratio_alignment(ent, clean)
            if al is not None and al.score >= 75:
                # map char span to token indices
                pos = 0
                for i, t in enumerate(ctoks):
                    s, e = pos, pos + len(t)
                    if start < 0 and e > al.dest_start:
                        start = i
                    if s < al.dest_end:
                        end = i + 1
                    pos = e + 1
                found = "fuzzy"
    if start < 0:
        win = " ".join(ctoks[: 2 * window_tokens])
        return Context(clean, win, found, False, False, False, False, "")
    lo, hi = max(0, start - window_tokens), min(len(ctoks), end + window_tokens)
    win = " ".join(ctoks[lo:start] + ctoks[end:hi])
    pre = " ".join(ctoks[max(0, start - 5):start])
    # cues inside the entity string itself also count (entities like "h/o mi", "no edema")
    pre_e = (pre + " " + " ".join(etoks[:3])).strip()
    return Context(clean, win, found, _has_cue(pre_e, NEGATION), _has_cue(pre_e, HISTORY),
                   _has_cue(pre_e, UNCERTAIN), _has_cue(pre_e, FAMILY), pre)
