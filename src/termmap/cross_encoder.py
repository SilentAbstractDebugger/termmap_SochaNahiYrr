"""Step 4: cross-encoder reranking features (optional; OFF unless trained and enabled).

The cross-encoder reads  [entity ; context window]  x  [candidate description | ICD category
description or SNOMED semantic tag]  and outputs one relevance logit. It never sees a code id, so it
is code-agnostic by construction. It scores only the top-N candidates (by first-stage score) of a
query; its score / rank / gap become LightGBM features (LightGBM stays the decision maker).

Training is out-of-fold over grouped folds (scripts/train_cross_encoder.py) so training-table
scores never come from a model that saw that query; code-disjoint codes are excluded by the
experiment's training partition. Scores are cached on disk keyed by (model, text pair) so repeated
runs replay identical numbers (determinism on GPU).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .context import build_context
from .normalize import basic_norm
from .terminology import icd_nodot


def query_text(entity: str, chunk: str, window_tokens: int = 12) -> str:
    w = build_context(entity, chunk, window_tokens).window
    e = basic_norm(entity)
    return f"{e} ; {w}" if w else e


def doc_text(kb, system: str, code: str) -> str:
    d = (kb.descs_norm.get(code) or [""])[0]
    if system == "ICD10CM":
        cat = icd_nodot(code)[:3]
        cat_code = next((c for c in (cat,) if kb.valid(c)), None)
        extra = (kb.descs_norm.get(cat_code) or [""])[0] if cat_code and cat_code != code else ""
        return f"{d} | category: {extra}" if extra else d
    return f"{d} | {kb.sem_tag.get(code, '')}"


def top_n_pool(feats: pd.DataFrame, n: int) -> pd.DataFrame:
    d = feats[["qid", "code", "s0"] + (["label"] if "label" in feats else [])]
    d = d.sort_values(["qid", "s0", "code"], ascending=[True, False, True], kind="mergesort")
    return d.groupby("qid", sort=False).head(n).reset_index(drop=True)


class CrossEncoderScorer:
    def __init__(self, model_path, kbs: dict, top_n: int = 30, batch_size: int = 64, max_len: int = 96,
                 cache_dir: str | Path | None = None, device: str | None = None):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.path = str(model_path)
        self.kbs, self.top_n, self.bs, self.max_len = kbs, top_n, batch_size, max_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.path, num_labels=1).to(self.device)
        if self.device == "cuda":
            self.model.half()
        self.model.eval()
        self.cache_path = Path(cache_dir or Path(self.path) / "score_cache.json")
        self._cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self._mkey = hashlib.sha1(self.path.encode()).hexdigest()[:8]

    def _k(self, q, d):
        return hashlib.sha1(f"{self._mkey}|{q}|{d}".encode()).hexdigest()

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        keys = [self._k(q, d) for q, d in pairs]
        todo = [i for i, k in enumerate(keys) if k not in self._cache]
        for b in range(0, len(todo), self.bs):
            idx = todo[b:b + self.bs]
            enc = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation=True,
                           max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**enc).logits.float().view(-1).cpu().numpy()
            for i, v in zip(idx, logits):
                self._cache[keys[i]] = float(v)
        if todo:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache))
            os.replace(tmp, self.cache_path)
        return np.array([self._cache[k] for k in keys], dtype=np.float32)

    def score_feats(self, feats: pd.DataFrame, queries: pd.DataFrame, system: str) -> pd.DataFrame:
        pool = top_n_pool(feats, self.top_n)
        qt = {q: query_text(e, c) for q, e, c in zip(queries["qid"], queries["entity"], queries["chunk_text"])}
        pairs = [(qt[q], doc_text(self.kbs[system], system, c)) for q, c in zip(pool["qid"], pool["code"])]
        pool["ce_score"] = self.score_pairs(pairs) if pairs else np.zeros(0)
        return pool[["qid", "code", "ce_score"]]


def merge_scores(feats: pd.DataFrame, table: pd.DataFrame | None) -> pd.DataFrame:
    """Add ce_score / ce_rank / ce_gap_top (NaN for candidates outside the scored top-N)."""
    f = feats.drop(columns=[c for c in ("ce_score", "ce_rank", "ce_gap_top") if c in feats.columns])
    if table is None or not len(table) or not len(f):
        f["ce_score"], f["ce_rank"], f["ce_gap_top"] = np.nan, np.nan, np.nan
        return f
    t = table[["qid", "code", "ce_score"]].copy()
    t = t.sort_values(["qid", "ce_score", "code"], ascending=[True, False, True], kind="mergesort")
    t["ce_rank"] = t.groupby("qid").cumcount().astype(np.float32)
    t["ce_gap_top"] = (t.groupby("qid")["ce_score"].transform("max") - t["ce_score"]).astype(np.float32)
    out = f.merge(t, on=["qid", "code"], how="left")
    for c in ("ce_score", "ce_rank", "ce_gap_top"):
        out[c] = out[c].astype(np.float32)
    return out


def train_cross_encoder(examples: list[tuple[str, str, float]], out_dir: str | Path, base_model: str,
                        epochs: int = 1, lr: float = 2e-5, batch_size: int = 32, max_len: int = 96,
                        seed: int = 13, save_every_steps: int = 500, warmup_frac: float = 0.1,
                        time_box_minutes: float = 40) -> str:
    """Pointwise BCE fine-tuning; resumable (checkpoint dir with step counter)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    out_dir = Path(out_dir)
    done = out_dir / "DONE"
    if done.exists():
        return str(out_dir)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    ckpt = out_dir / "checkpoint"
    start_from = str(ckpt) if (ckpt / "state.json").exists() else base_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(start_from)
    model = AutoModelForSequenceClassification.from_pretrained(start_from, num_labels=1,
                                                               ignore_mismatched_sizes=True).to(dev)
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(len(examples)) for _ in range(epochs)])
    steps = (len(order) + batch_size - 1) // batch_size
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = get_linear_schedule_with_warmup(opt, int(warmup_frac * steps), steps)
    step0 = json.loads((ckpt / "state.json").read_text())["step"] if start_from == str(ckpt) else 0
    for _ in range(step0):
        sch.step()
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0 = time.time()
    for st in range(step0, steps):
        idx = order[st * batch_size:(st + 1) * batch_size]
        b = [examples[i] for i in idx]
        enc = tok([x[0] for x in b], [x[1] for x in b], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt").to(dev)
        y = torch.tensor([x[2] for x in b], dtype=torch.float32, device=dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            loss = lossf(model(**enc).logits.view(-1).float(), y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sch.step()
        if (st + 1) % save_every_steps == 0 or st == steps - 1:
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)
            (ckpt / "state.json").write_text(json.dumps({"step": st + 1, "steps": steps}))
            print(f"[cross-encoder] step {st + 1}/{steps} loss {loss.item():.4f} "
                  f"{(time.time() - t0) / 60:.1f} min (checkpointed)", flush=True)
            if time.time() - t0 > time_box_minutes * 60 and st < steps - 1:
                print("[cross-encoder] time box reached; rerun to resume", flush=True)
                return str(ckpt)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    done.write_text("ok")
    return str(out_dir)
