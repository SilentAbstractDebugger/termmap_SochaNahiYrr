"""Terminology knowledge bases built ONLY from the supplied dictionary files.

No code string, description or mapping is hardcoded. Hierarchy for ICD-10-CM is derived from
the code strings present in the dictionary. SNOMED has no IS-A in the supplied file; semantic
tags are parsed from fully specified names.
"""
from __future__ import annotations

import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .normalize import basic_norm, strip_semantic_tag
from .util import file_sha256, resolve

KB_CACHE_VERSION = "kb-v5"
_ABBR_DASH = re.compile(r"^([A-Za-z][A-Za-z0-9+/]{1,7}) - (.{3,})$")
_ABBR_PAREN = re.compile(r"^(.{3,}?)\s*\(([A-Z][A-Z0-9]{1,6})\)\s*$")


def norm_icd(code: str | None) -> str:
    """Canonical dotted ICD form: uppercase, strip spaces; insert dot after 3rd char if missing."""
    if code is None:
        return ""
    c = str(code).strip().upper().replace(" ", "")
    if c in ("", "NAN", "NONE"):
        return ""
    if "-" in c:
        return c
    if "." not in c and len(c) > 3:
        c = c[:3] + "." + c[3:]
    return c


def norm_snomed(code: str | None) -> str:
    if code is None:
        return ""
    c = str(code).strip()
    if c.lower() in ("", "nan", "none"):
        return ""
    if re.fullmatch(r"\d+\.0", c):  # float-corrupted id
        c = c[:-2]
    return c


def icd_nodot(code: str) -> str:
    return code.replace(".", "")


@dataclass
class KB:
    system: str
    codes: list[str]                                   # valid candidate codes (sorted)
    code_set: set[str]
    descs: dict[str, list[str]]                       # code -> raw descriptions (dedup, order kept)
    descs_norm: dict[str, list[str]]                  # code -> normalized descriptions (dedup)
    doc_text: list[str]                               # unique normalized description strings
    doc_codes: list[list[str]]                        # doc index -> codes having that string
    norm_to_codes: dict[str, list[str]]               # normalized description -> codes
    preferred: dict[str, str]                         # code -> display description
    meta: dict = field(default_factory=dict)
    # ICD only
    parent: dict[str, str] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=dict)
    is_leaf: dict[str, bool] = field(default_factory=dict)
    chapter: dict[str, str] = field(default_factory=dict)
    range_headers: list[tuple[str, str]] = field(default_factory=list)
    # SNOMED only
    sem_tag: dict[str, str] = field(default_factory=dict)
    # dictionary-mined abbreviation lexicon: abbr -> Counter(expansion)
    abbr: dict[str, Counter] = field(default_factory=dict)

    def valid(self, code: str) -> bool:
        return code in self.code_set

    def depth(self, code: str) -> int:
        return len(icd_nodot(code)) if self.system == "ICD10CM" else 0

    def category(self, code: str) -> str:
        return icd_nodot(code)[:3] if self.system == "ICD10CM" else code

    def ancestors(self, code: str) -> list[str]:
        out, c = [], self.parent.get(code)
        while c:
            out.append(c)
            c = self.parent.get(c)
        return out


def _mine_abbr(raw_desc: str, out: dict[str, Counter]) -> None:
    m = _ABBR_DASH.match(raw_desc)
    if m:
        a, e = basic_norm(m.group(1)), basic_norm(strip_semantic_tag(m.group(2))[0])
        if a and e and a != e and len(a) <= 8:
            out.setdefault(a, Counter())[e] += 1
        return
    m = _ABBR_PAREN.match(raw_desc)
    if m:
        a, e = basic_norm(m.group(2)), basic_norm(m.group(1))
        if a and e and a != e and len(e.split()) >= 2:
            out.setdefault(a, Counter())[e] += 1


def build_icd_kb(df: pd.DataFrame) -> KB:
    df = df.rename(columns={df.columns[0]: "Code", df.columns[1]: "Description"}).astype(str)
    df["Code"] = df["Code"].map(norm_icd)
    df = df[df["Code"] != ""]
    ranges = df[df["Code"].str.contains("-", regex=False)]
    range_headers = sorted({(r.Code, r.Description) for r in ranges.itertuples()})
    df = df[~df["Code"].str.contains("-", regex=False)]
    descs: dict[str, list[str]] = defaultdict(list)
    for c, d in zip(df["Code"], df["Description"]):
        d = d.strip()
        if d and d.lower() != "nan" and d not in descs[c]:
            descs[c].append(d)
    codes = sorted(descs)
    code_set = set(codes)
    nodot = {icd_nodot(c): c for c in codes}
    parent, children = {}, defaultdict(list)
    for c in codes:
        nd = icd_nodot(c)
        for L in range(len(nd) - 1, 2, -1):
            if nd[:L] in nodot:
                parent[c] = nodot[nd[:L]]
                children[nodot[nd[:L]]].append(c)
                break
    is_leaf = {c: c not in children for c in codes}
    # The supplied file's range headers are sparse/inconsistent (111 rows, not a full chapter
    # table), so the chapter proxy is the code's first letter (documented; D and H each span
    # two official chapters). Range headers are kept for reporting only.
    chapter = {c: c[0] for c in codes}
    return _finish_kb("ICD10CM", codes, code_set, descs, parent=parent,
                      children={k: sorted(v) for k, v in children.items()}, is_leaf=is_leaf,
                      chapter=chapter, range_headers=range_headers)


def build_snomed_kb(df: pd.DataFrame) -> KB:
    df = df.astype(str)
    order = pd.to_numeric(df["description_order"], errors="coerce").fillna(1e9)
    df = df.assign(_o=order).sort_values(["snomed_code", "_o"], kind="mergesort")
    descs: dict[str, list[str]] = defaultdict(list)
    for c, d in zip(df["snomed_code"].map(norm_snomed), df["description"]):
        d = d.strip()
        if not c or not d or d.lower() == "nan":
            continue
        if d not in descs[c]:
            descs[c].append(d)
    # Semantic tags: trailing parentheticals that recur as tags across the dictionary
    # (data-derived threshold) -- avoids treating e.g. "... (chronic obstructive ...)" as a tag.
    tag_counts: Counter = Counter()
    for ds in descs.values():
        for d in ds:
            t = strip_semantic_tag(d)[1]
            if t:
                tag_counts[t] += 1
    real_tags = {t for t, n in tag_counts.items() if n >= 100}
    sem_tag: dict[str, str] = {}
    for c, ds in descs.items():
        for d in ds:
            t = strip_semantic_tag(d)[1]
            if t in real_tags:
                sem_tag[c] = t
                break
    codes = sorted(descs)
    for c in codes:
        sem_tag.setdefault(c, "unknown")
    return _finish_kb("SNOMEDCT_US", codes, set(codes), descs, sem_tag=sem_tag)


def _finish_kb(system: str, codes, code_set, descs, **kw) -> KB:
    descs_norm: dict[str, list[str]] = {}
    norm_to_codes: dict[str, list[str]] = defaultdict(list)
    abbr: dict[str, Counter] = {}
    preferred: dict[str, str] = {}
    for c in codes:
        ns = []
        for d in descs[c]:
            _mine_abbr(d, abbr)
            n = basic_norm(strip_semantic_tag(d)[0])
            if n and n not in ns:
                ns.append(n)
        descs_norm[c] = ns
        for n in ns:
            norm_to_codes[n].append(c)
        # display: first description that is not an FSN, else the first one
        disp = next((d for d in descs[c] if strip_semantic_tag(d)[1] is None), descs[c][0])
        preferred[c] = disp
    doc_text = sorted(norm_to_codes)
    doc_codes = [sorted(norm_to_codes[t]) for t in doc_text]
    kb = KB(system=system, codes=codes, code_set=code_set, descs=dict(descs), descs_norm=descs_norm,
            doc_text=doc_text, doc_codes=doc_codes, norm_to_codes=dict(norm_to_codes),
            preferred=preferred, abbr=abbr, **kw)
    return kb


def load_kbs(cfg: dict, icd_path: str | Path | None = None, snomed_path: str | Path | None = None,
             use_cache: bool = True) -> dict[str, KB]:
    """Load (or build and cache) both KBs. Cache key = file hash + cache version."""
    rc = cfg["resources"]
    icd_path = resolve(icd_path or rc["icd_path"])
    snomed_path = resolve(snomed_path or rc["snomed_path"])
    cache_dir = resolve(rc["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for system, path, sheet, builder in (
        ("ICD10CM", icd_path, rc.get("icd_sheet", "Sheet1"), build_icd_kb),
        ("SNOMEDCT_US", snomed_path, rc.get("snomed_sheet", "SNOMED codes"), build_snomed_kb),
    ):
        h = file_sha256(path)
        cp = cache_dir / f"{system}_{KB_CACHE_VERSION}_{h[:16]}.pkl"
        if use_cache and cp.exists():
            with open(cp, "rb") as f:
                kb = pickle.load(f)
        else:
            if str(path).endswith((".csv", ".tsv")):
                df = pd.read_csv(path, dtype=str, keep_default_na=False,
                                 sep="\t" if str(path).endswith(".tsv") else ",")
            else:
                df = pd.read_excel(path, sheet_name=sheet, dtype=str)
            kb = builder(df)
            kb.meta = {"path": str(path), "sha256": h, "n_codes": len(kb.codes),
                       "n_desc_rows": int(len(df)), "n_unique_norm_desc": len(kb.doc_text)}
            if use_cache:
                with open(cp, "wb") as f:
                    pickle.dump(kb, f, protocol=pickle.HIGHEST_PROTOCOL)
        out[system] = kb
    return out
