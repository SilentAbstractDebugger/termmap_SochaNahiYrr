# Kaggle runbook (GPU stages: dense retrieval, cross-encoder, fine-tuning)

These stages were **NOT RUN** in the build sandbox (no GPU, no Hugging Face access). The code is
complete, defaults to OFF, and was mechanically tested on a tiny random CPU BERT (`tests/test_dense.py`).

> **Approval first.** The training file is proprietary. Running it on a hosted notebook platform is a
> decision the organisers must approve. If approved: keep the dataset and notebook **private**, never
> publish outputs that contain rows (`reports/private/`), and never send data to any API.

## Notebook settings

| setting | value |
|---|---|
| Accelerator | GPU T4 x2 or P100 (one GPU is used) |
| Internet | **ON** for stage (a) only (model downloads). Everything after runs offline (`HF_HUB_OFFLINE=1`). |
| Persistence | Settings -> Persistence -> "Files only" (keeps `/kaggle/working` caches between sessions), or save a version and re-attach its output |
| Session limit | 12 h per session; stages resume from caches, so split across sessions if needed |

## Where things live

```text
/kaggle/input/termmap-data/      PRIVATE dataset: hackathon_train.csv, icd10cm_codes.xlsx, SNOMEDCT_US_codes_and_descriptions.xlsx
/kaggle/input/termmap-repo/      the repository (private dataset or git clone into /kaggle/working)
/kaggle/working/termmap/         working copy; caches: resources/cache/ (embeddings), models/cache/ (feature tables),
                                 models/cross_encoder/, models/encoders/; logs: logs/kaggle/<stage>.log
```

```bash
cp -r /kaggle/input/termmap-repo /kaggle/working/termmap && cd /kaggle/working/termmap
mkdir -p data resources
ln -s /kaggle/input/termmap-data/hackathon_train.csv data/
ln -s /kaggle/input/termmap-data/icd10cm_codes.xlsx resources/
ln -s /kaggle/input/termmap-data/SNOMEDCT_US_codes_and_descriptions.xlsx resources/
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
bash scripts/kaggle_run.sh            # or: bash scripts/kaggle_run.sh a b   (stages are resumable)
```

### If internet must stay OFF

On any machine with internet: `python scripts/setup_resources.py --download-dense --only thenlper/gte-base,cambridgeltl/SapBERT-from-PubMedBERT-fulltext,FremyCompany/BioLORD-2023,BAAI/bge-m3,ncbi/MedCPT-Cross-Encoder`
(this records the commit sha + licence in `configs/resources.yaml`), upload `resources/models/` as a private
Kaggle dataset `termmap-models`, attach it, then `ln -s /kaggle/input/termmap-models resources/models` (or
`MODELS_DIR=/kaggle/input/termmap-models bash scripts/kaggle_run.sh`). Stage (a) detects missing HF access and
checks the folders instead of downloading.

## Stages (order is fixed; each one resumable)

Times are **rough planning estimates for a single T4, not measurements** (nothing GPU-side was run).

| stage | what | output | est. time |
|---|---|---|---|
| a | GPU + HF check, deps, download pinned encoders, sparse caches | `configs/resources.yaml` (sha + licence), `resources/cache/` | 10-20 min |
| b | gte-base: embed all ~780k ICD+SNOMED strings (+ICD path texts), recall@{1..300} on validation + code-disjoint, gte top-1 = gold diagnostic | `reports/dense_benchmark.{json,md}`, `resources/cache/emb_gte_*` | 10-20 min |
| c | same for SapBERT, BioLORD, bge-m3 | appended to the benchmark | 45-90 min (bge-m3 is the slowest) |
| d | union + RRF recall over sparse + all dense models/views (embeddings reused) | `fused_this_run` section | 10-20 min |
| e | cross-encoder (MedCPT-Cross-Encoder start) on top-30 pools, 15 hard negatives, out-of-fold per experiment + final model | `models/cache/ce_tables_*`, `models/cross_encoder/` | 1.5-3 h |
| f | contrastive fine-tune of the best encoder (`BEST_ENCODER=sapbert` by default) with code-disjoint codes, val and test rows excluded; re-benchmark | `models/encoders/<best>-ft/finetune_report.json` | 20-40 min |
| g | ablations with dense + CE features, 3 seeds, bootstrap CIs; FINAL TEST once; final refit | `reports/kaggle_metrics.json`, `models/final.pkl` | 1.5-3 h (LightGBM is CPU-bound) |
| h | metrics / ablations / error analysis (val, code-disjoint, final test), benchmark, checks, tests, summary, PDF | `reports/*`, `docs/TermMap_writeup.pdf` | 15-30 min |

Before stage g, edit `configs/kaggle_dense.yaml` to keep only the encoders that added recall in stages b-d
(decision rule: keep a channel only if union recall or accuracy rises beyond noise).

## Determinism on GPU

Embeddings and cross-encoder scores are computed once in a fixed batch order and cached (string- /
pair-level, hash-keyed); every later run replays the cache, so two `termmap.predict` runs give identical
hashes once caches exist. First-time GPU encoding can differ in the last bits across GPU models; report
any residual difference rather than claiming it away (`scripts/run_checks.py` writes `reports/determinism.json`).

## Without a GPU

The same commands run on CPU (stages b-f become hours). `python -m termmap.predict` never needs a GPU: with
`configs/evaluation.yaml` it runs the sparse + static-vector mode and prints the mode it ran.
