"""Shared helpers: config, hashing, seeding, deterministic sorting."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_env_file(path: Path | None = None) -> None:
    """Minimal .env loader (no third-party dependency). Never overrides existing env."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load configs/default.yaml, then overlay `path` (if given), then env overrides."""
    load_env_file()
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if p.resolve() != (ROOT / "configs" / "default.yaml").resolve():
            cfg = _merge(cfg, yaml.safe_load(p.read_text()) or {})
    env_map = {
        "TERMMAP_ICD_PATH": ("resources", "icd_path"),
        "TERMMAP_SNOMED_PATH": ("resources", "snomed_path"),
        "TERMMAP_TRAIN_PATH": ("data", "train_path"),
        "TERMMAP_CACHE_DIR": ("resources", "cache_dir"),
        "TERMMAP_CROSSWALK_PATH": ("resources", "crosswalk_path"),
        "TERMMAP_SNOMED_RELATIONSHIPS_PATH": ("resources", "snomed_relationships_path"),
        "TERMMAP_DENSE_MODEL_PATH": ("dense", "model_path"),
    }
    for env, (sec, key) in env_map.items():
        if os.environ.get(env):
            cfg.setdefault(sec, {})[key] = os.environ[env]
    return cfg


def resolve(p: str | Path | None) -> Path | None:
    if p in (None, ""):
        return None
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def text_sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # optional
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def stable_rng(seed: int, *keys: Any) -> np.random.Generator:
    """RNG derived deterministically from a seed plus string keys (independent of order of calls)."""
    h = hashlib.sha256((str(seed) + "|" + "|".join(map(str, keys))).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())
