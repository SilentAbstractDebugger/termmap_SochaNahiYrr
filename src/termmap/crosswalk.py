"""Optional SNOMED CT -> ICD-10-CM map (NLM "SNOMED CT to ICD-10-CM Map", RF2 extended map refset
or its TSV export). AUXILIARY ONLY: used for a compatibility feature / disagreement signal in
diagnostics, never to derive the ICD prediction. Not supplied in the challenge bundle, so the
submitted run has it OFF (engine runs fully without it)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from .terminology import norm_icd


class Crosswalk:
    def __init__(self, mapping: dict[str, set[str]], source: str):
        self.map, self.source = mapping, source

    @classmethod
    def load(cls, path: str | Path | None) -> "Crosswalk | None":
        if not path or not Path(path).exists():
            return None
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        cols = {c.lower(): c for c in df.columns}
        src = cols.get("referencedcomponentid") or cols.get("snomed_code") or cols.get("conceptid")
        tgt = cols.get("maptarget") or cols.get("icd10cm_code") or cols.get("icd_code")
        if src is None or tgt is None:
            raise ValueError(f"Unrecognised crosswalk columns: {list(df.columns)}")
        if "active" in cols:
            df = df[df[cols["active"]] == "1"]
        m: dict[str, set[str]] = defaultdict(set)
        for s, t in zip(df[src], df[tgt]):
            t = norm_icd(t.rstrip("?"))
            if s and t:
                m[s].add(t)
        return cls(dict(m), str(path))

    def compatible(self, snomed: str, icd: str) -> float:
        tg = self.map.get(snomed)
        if not tg:
            return -1.0  # unknown
        if icd in tg:
            return 1.0
        return 0.5 if any(t[:3] == icd[:3] for t in tg) else 0.0
