"""Throughput / latency / memory benchmark -> reports/benchmark.json (numbers only, no row text).

Batch throughput: all validation queries in one batch. Per-term latency: N single-query calls.
"""
import argparse
import os
import platform
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from termmap.data import diagnosis_pairs, load_train  # noqa: E402
from termmap.pipeline import Predictor, load_model, load_resources  # noqa: E402
from termmap.util import ROOT, load_config, read_json, resolve, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/evaluation.yaml")
    ap.add_argument("--single", type=int, default=60)
    a = ap.parse_args()
    cfg = load_config(a.config)
    t0 = time.time()
    R = load_resources(cfg)
    t_load = time.time() - t0
    model = load_model(ROOT / "models" / "final.pkl")
    pred = Predictor(R, model)
    pairs = diagnosis_pairs(load_train(cfg), R.kbs)
    sp = read_json(ROOT / "reports" / "splits.json")
    va = pairs[pairs["pair_id"].isin(set(sp["val_pair_ids"]))]
    q = va.rename(columns={"pair_id": "qid"})[["qid", "entity", "chunk_text"]].reset_index(drop=True)
    # fresh retrieval caches so batch timing is not flattered by earlier calls
    R.fb._bm25_cache.clear()
    R.fb._char_cache.clear()
    R.fb._dense_cache.clear()
    t = time.time()
    pred.predict(q)
    t_batch = time.time() - t
    lat = []
    R.fb._bm25_cache.clear()
    R.fb._char_cache.clear()
    R.fb._dense_cache.clear()
    for i in range(min(a.single, len(q))):
        t = time.time()
        pred.predict(q.iloc[[i]])
        lat.append(time.time() - t)
    cache = resolve(cfg["resources"]["cache_dir"])
    sizes = {f.name: round(f.stat().st_size / 1e6, 1) for f in sorted(cache.iterdir())}
    mp = ROOT / "models" / "final.pkl"
    out = {
        "hardware": {"platform": platform.platform(), "python": platform.python_version(),
                     "cpu_count": os.cpu_count(), "gpu": "none",
                     "num_threads_config": cfg["num_threads"]},
        "mode": R.mode,
        "resource_load_seconds_warm_cache": round(t_load, 1),
        "batch": {"n_terms": len(q), "seconds": round(t_batch, 2),
                  "terms_per_second_both_systems": round(len(q) / t_batch, 2)},
        "single_term_latency_seconds": {"n": len(lat), "median": round(float(np.median(lat)), 3),
                                        "p95": round(float(np.percentile(lat, 95)), 3)},
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "index_sizes_mb": sizes, "model_size_mb": round(mp.stat().st_size / 1e6, 1),
        "per_term_model_calls": {"cross_encoder_pairs": (cfg.get("cross_encoder", {}).get("top_n", 0)
                                                         if R.fb.ce is not None else 0), "llm_calls": 0,
                                 "dense_query_encodings": len(R.fb.dense or {})},
        "gpu_mode": "NOT RUN (no GPU in build environment)",
    }
    write_json(ROOT / "reports" / "benchmark.json", out)
    print(out)


if __name__ == "__main__":
    main()
