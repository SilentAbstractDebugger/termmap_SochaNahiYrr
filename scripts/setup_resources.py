"""Check / place challenge files and (optionally) download pinned dense models.

    python scripts/setup_resources.py                 # verify files + write configs/resources.yaml hashes
    python scripts/setup_resources.py --download-dense  # needs internet + `pip install sentence-transformers faiss-cpu`

Challenge files are proprietary and are NOT downloaded by this script: copy them to
  data/hackathon_train.csv
  resources/icd10cm_codes.xlsx
  resources/SNOMEDCT_US_codes_and_descriptions.xlsx
(or point TERMMAP_* variables in .env to them).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from termmap.util import ROOT, file_sha256, load_config, resolve  # noqa: E402

DENSE_MODELS = {  # Step 1/3/4 candidates; licence + exact commit are recorded at download time
    "thenlper/gte-base": "resources/models/gte-base",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext": "resources/models/sapbert",
    "FremyCompany/BioLORD-2023": "resources/models/biolord-2023",
    "ncbi/MedCPT-Query-Encoder": "resources/models/medcpt-query",
    "ncbi/MedCPT-Article-Encoder": "resources/models/medcpt-article",
    "ncbi/MedCPT-Cross-Encoder": "resources/models/medcpt-cross",
    "BAAI/bge-m3": "resources/models/bge-m3",
    "BAAI/bge-reranker-base": "resources/models/bge-reranker-base",
    "intfloat/e5-base-v2": "resources/models/e5-base-v2",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dense", action="store_true")
    ap.add_argument("--revision", default=None, help="pin a commit hash (default: current main, recorded)")
    ap.add_argument("--only", default=None, help="comma list of model ids to download")
    a = ap.parse_args()
    cfg = load_config()
    files = {"train": resolve(cfg["data"]["train_path"]), "icd": resolve(cfg["resources"]["icd_path"]),
             "snomed": resolve(cfg["resources"]["snomed_path"])}
    man = yaml.safe_load((ROOT / "configs" / "resources.yaml").read_text())
    ok = True
    for k, p in files.items():
        if p.exists():
            h = file_sha256(p)
            man["files"][k]["sha256"] = h
            print(f"OK   {k}: {p} sha256={h[:16]}...")
        else:
            ok = False
            print(f"MISSING {k}: {p}")
    (ROOT / "configs" / "resources.yaml").write_text(yaml.safe_dump(man, sort_keys=False))
    if a.download_dense:
        from huggingface_hub import HfApi, snapshot_download

        api = HfApi()
        wanted = a.only.split(",") if a.only else list(DENSE_MODELS)
        man.setdefault("models", {})["downloaded"] = man.get("models", {}).get("downloaded", {}) or {}
        for mid in wanted:
            path = DENSE_MODELS[mid]
            info = api.model_info(mid, revision=a.revision)
            lic = (getattr(info, "card_data", None) or {}).get("license") if getattr(info, "card_data", None) else None
            p = snapshot_download(mid, revision=info.sha, local_dir=str(ROOT / path))
            man["models"]["downloaded"][mid] = {"path": path, "revision": info.sha,
                                                "licence_from_model_card": lic or "not stated in card metadata"}
            print("downloaded", mid, "@", info.sha, "licence:", lic, "->", p)
        (ROOT / "configs" / "resources.yaml").write_text(yaml.safe_dump(man, sort_keys=False))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
