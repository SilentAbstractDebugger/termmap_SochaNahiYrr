"""Build (or rebuild) all hashed caches from the dictionary files named in the config:
knowledge bases + lexical indexes (+ dense embeddings if dense.enabled). Idempotent."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from termmap.pipeline import load_resources  # noqa: E402
from termmap.util import load_config, resolve  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    t = time.time()
    cfg = load_config(a.config)
    R = load_resources(cfg)
    cache = resolve(cfg["resources"]["cache_dir"])
    for f in sorted(cache.iterdir()):
        print(f"{f.name}\t{f.stat().st_size / 1e6:.1f} MB")
    print({s: kb.meta for s, kb in R.kbs.items()})
    print("mode:", R.mode, f"({time.time() - t:.0f}s)")
