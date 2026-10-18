"""Stage D3: dense retrieval with LOCALLY hosted sentence encoders (gte-base, SapBERT, BioLORD,
MedCPT, bge-m3, e5, bge ...), alongside the sparse channels.

* Exact flat inner-product search (faiss.IndexFlatIP; numpy fallback) -> deterministic.
* Embeddings: fp16 on GPU when available (fp32 on CPU), fixed batch order, L2-normalised.
* EmbeddingStore: string-level, hash-keyed, append-only cache (resources/cache/emb_<model>/).
  Resumable (flushes every `save_every` batches) and INCREMENTAL: a new dictionary release only
  embeds strings not already in the store (drift).
* Dictionary-side enrichment (from the supplied dictionaries only):
    - every unique normalised description string (SNOMED FSNs are stored with the semantic tag
      stripped, synonyms included);
    - ICD "path" texts: "<description> ; category: <3-char category description>" so short leaf
      descriptions carry their block meaning.
* Query views are supplied by the caller (entity, repaired, abbreviation expansions, bridge texts,
  entity + context window).

The build sandbox of this repo had no GPU and no Hugging Face access, so this module was only
exercised with a tiny randomly initialised BERT (tests/test_dense.py); accuracy NOT RUN there.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .terminology import icd_nodot
from .util import resolve


def _key(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


class EmbeddingStore:
    def __init__(self, root: Path, dim: int | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.kp, self.ep = self.root / "keys.json", self.root / "emb.npy"
        self.keys: list[str] = json.loads(self.kp.read_text()) if self.kp.exists() else []
        self.emb = np.load(self.ep) if self.ep.exists() else np.zeros((0, dim or 0), dtype=np.float16)
        if len(self.keys) != len(self.emb):  # torn write: trust the shorter
            n = min(len(self.keys), len(self.emb))
            self.keys, self.emb = self.keys[:n], self.emb[:n]
        self.pos = {k: i for i, k in enumerate(self.keys)}

    def missing(self, texts: list[str]) -> list[str]:
        seen, out = set(), []
        for t in texts:
            k = _key(t)
            if k not in self.pos and k not in seen:
                seen.add(k)
                out.append(t)
        return out

    def add(self, texts: list[str], emb: np.ndarray) -> None:
        emb = emb.astype(np.float16)
        if self.emb.shape[1] == 0:
            self.emb = np.zeros((0, emb.shape[1]), dtype=np.float16)
        for t in texts:
            self.pos[_key(t)] = len(self.keys)
            self.keys.append(_key(t))
        self.emb = np.concatenate([self.emb, emb], axis=0)

    def flush(self) -> None:
        tmp_e, tmp_k = self.root / "emb.tmp.npy", self.root / "keys.tmp.json"
        np.save(tmp_e, self.emb)
        tmp_k.write_text(json.dumps(self.keys))
        os.replace(tmp_e, self.ep)
        os.replace(tmp_k, self.kp)

    def get(self, texts: list[str]) -> np.ndarray:
        return self.emb[[self.pos[_key(t)] for t in texts]].astype(np.float32)


class SentenceEncoder:
    def __init__(self, path: str, device: str | None = None, batch_size: int = 256, max_seq_length: int = 64,
                 pooling: str | None = None):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from sentence_transformers import SentenceTransformer, models

        torch.manual_seed(0)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if pooling:  # plain HF checkpoints (e.g. MedCPT, SapBERT use [CLS])
            word = models.Transformer(path, max_seq_length=max_seq_length)
            pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode=pooling)
            self.model = SentenceTransformer(modules=[word, pool], device=self.device)
        else:
            self.model = SentenceTransformer(path, device=self.device)
        self.model.max_seq_length = max_seq_length
        if self.device == "cuda":
            self.model.half()
        self.model.eval()
        self.batch = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            e = self.model.encode(texts, batch_size=self.batch, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(e, dtype=np.float32)


def icd_path_texts(kb) -> tuple[list[str], list[list[str]]]:
    """'<desc> ; category: <category desc>' per ICD code (codes deeper than 3 chars)."""
    texts, codes = [], []
    nodot = {icd_nodot(c): c for c in kb.codes}
    for c in kb.codes:
        nd = icd_nodot(c)
        cat = nodot.get(nd[:3])
        if len(nd) <= 3 or cat is None or not kb.descs_norm.get(cat):
            continue
        texts.append(f"{kb.descs_norm[c][0]} ; category: {kb.descs_norm[cat][0]}")
        codes.append([c])
    return texts, codes


class DenseIndex:
    """One sentence encoder over both dictionaries. search() returns (doc_idx, score) lists and
    self.doc_codes[system][doc_idx] gives the codes of a document."""

    def __init__(self, cfg: dict, kbs: dict, spec: dict | None = None, encoder=None):
        d = spec or cfg["dense"]
        self.name = d.get("name", "dense")
        mp = resolve(d.get("model_path"))
        if encoder is None:
            if mp is None or not Path(mp).exists():
                raise FileNotFoundError(f"dense model_path not found: {mp}")
            encoder = SentenceEncoder(str(mp), batch_size=int(d.get("batch_size", 256)),
                                      max_seq_length=int(d.get("max_seq_length", 64)), pooling=d.get("pooling"))
        self.enc = encoder
        # asymmetric bi-encoders (MedCPT query/article): separate document encoder
        dmp = resolve(d.get("doc_model_path"))
        self.doc_enc = (SentenceEncoder(str(dmp), batch_size=int(d.get("batch_size", 256)),
                                        max_seq_length=int(d.get("doc_max_seq_length", 96)), pooling=d.get("pooling"))
                        if dmp else self.enc)
        self.qp, self.dp = d.get("query_prefix", ""), d.get("doc_prefix", "")
        self.save_every = int(d.get("save_every_batches", 50))
        self.flush_s = float(d.get("flush_seconds", 300))
        self.time_box_s = float(d.get("time_box_minutes", 40)) * 60
        cache = resolve(cfg["resources"]["cache_dir"])
        mkey = hashlib.sha1(f"{mp}|{d.get('doc_model_path')}|{d.get('pooling')}|{self.dp}".encode()).hexdigest()[:12]
        self.store = EmbeddingStore(Path(cache) / f"emb_{self.name}_{mkey}")
        self.docs, self.doc_codes, self.index = {}, {}, {}
        for s, kb in kbs.items():
            texts = list(kb.doc_text)
            codes = [list(c) for c in kb.doc_codes]
            if s == "ICD10CM" and d.get("enrich_icd_path", True):
                pt, pc = icd_path_texts(kb)
                texts += pt
                codes += pc
            self.docs[s], self.doc_codes[s] = texts, codes
            self._ensure([self.dp + t for t in texts], label=s)
            E = self.store.get([self.dp + t for t in texts])
            self.index[s] = self._build_index(E, int(cfg.get("num_threads", 1)))

    def _ensure(self, texts: list[str], label: str) -> None:
        todo = self.store.missing(texts)
        if not todo:
            return
        t0, bs = time.time(), self.doc_enc.batch
        nb = (len(todo) + bs - 1) // bs
        print(f"[dense:{self.name}] {label}: embedding {len(todo)} new strings in {nb} batches", flush=True)
        buf_t, buf_e = [], []
        last_flush = time.time()
        for b in range(nb):
            chunk = todo[b * bs:(b + 1) * bs]
            buf_t += chunk
            buf_e.append(self.doc_enc.encode(chunk))
            last = b == nb - 1
            if (b + 1) % self.save_every == 0 or last:
                el = time.time() - t0
                eta = el / (b + 1) * (nb - b - 1)
                saved = False
                # checkpoint at most every `flush_seconds` (each flush rewrites the store file)
                if last or time.time() - last_flush > self.flush_s:
                    self.store.add(buf_t, np.concatenate(buf_e))
                    self.store.flush()
                    buf_t, buf_e, last_flush, saved = [], [], time.time(), True
                print(f"[dense:{self.name}] {label}: {b + 1}/{nb} batches, {el / 60:.1f} min, eta {eta / 60:.1f} min"
                      f"{' (checkpointed)' if saved else ''}", flush=True)
                if el > self.time_box_s and not last:
                    print(f"[dense:{self.name}] past the {self.time_box_s / 60:.0f}-min time box; checkpoints every "
                          f"{self.flush_s / 60:.0f} min, safe to interrupt and rerun", flush=True)

    @staticmethod
    def _build_index(E: np.ndarray, threads: int):
        try:
            import faiss  # type: ignore

            faiss.omp_set_num_threads(threads)
            ix = faiss.IndexFlatIP(E.shape[1])
            ix.add(np.ascontiguousarray(E, dtype=np.float32))
            return ix
        except ImportError:
            return E

    def search(self, system: str, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        if not queries:
            return []
        q = self.enc.encode([self.qp + t for t in queries]).astype(np.float32)
        ix = self.index[system]
        if isinstance(ix, np.ndarray):
            S = q @ ix.T
            kk = min(k, S.shape[1])
            I = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            D = np.take_along_axis(S, I, axis=1)
        else:
            D, I = ix.search(q, k)
        out = []
        for drow, irow in zip(D, I):
            out.append(sorted(((int(i), float(s)) for i, s in zip(irow, drow) if i >= 0), key=lambda x: (-x[1], x[0])))
        return out
