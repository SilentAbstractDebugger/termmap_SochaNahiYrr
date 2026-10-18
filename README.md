# TermMap: clinical term mapping to ICD-10-CM and SNOMED CT US

TermMap takes one extracted clinical entity (diagnoses, conditions, signs and symptoms) and its chunk text, and returns
**one ICD-10-CM code and one SNOMED CT US concept id**. Both codes are guaranteed to exist in the supplied dictionaries.
It runs locally and offline after setup. It needs no API key and never sends data anywhere.

> **Honest status.** The 95% target on both systems was **not reached**. The numbers below are interim until the final
> test split and the final refit are filled in (cells marked **PENDING**). Write-up: [`docs/TermMap_writeup.pdf`](docs/TermMap_writeup.pdf).
> Demo video: [DEMO VIDEO LINK]

## Results

<!-- RESULTS:START -->
_Status: **interim**. Generated from `docs/results_for_pdf.json` by `python scripts/make_pdf.py`._

| split | ICD-10-CM | SNOMED CT | headline (lower) |
|---|---|---|---|
| Validation (codes seen; 3 seeds) | 0.618 (CI **PENDING**) | 0.565 (CI **PENDING**) | 0.565 |
| Code-disjoint (codes never seen; 2 seeds) | 0.436 (CI [0.389, 0.484]) | 0.535 (CI [0.487, 0.581]) | 0.436 |
| Final test (scored once) | **PENDING** (CI **PENDING**) | **PENDING** (CI **PENDING**) | **PENDING** |

Target 95% on both systems: **not reached**. Candidate recall of the fused pool (a shortlist ceiling, not accuracy): validation @200 ICD 0.964, SNOMED 0.974; code-disjoint @200 ICD 0.937, SNOMED 0.954. Final refit: **PENDING**. Demo video: [DEMO VIDEO LINK].
<!-- RESULTS:END -->

The candidate-pool recall above is a shortlist ceiling, not an accuracy figure. A single encoder's top-1 is much lower,
so ranking is the weak step. Full tables: `reports/metrics.md`, `reports/ablations.md`, `reports/error_analysis_*.md`, `reports/SUMMARY.md`.

## Architecture (summary)

```
entity + chunk_text
  -> 1 input adapter (long/wide schema auto-detect)
  -> 2 multi-view normalization (original, typo/glue/truncation repairs, every abbreviation hypothesis)
  -> 3 context builder (markup strip, mention window, negation/history cues)
  -> 4 retrieval per system: exact/alias, BM25, char 3-grams, training neighbours, synonym bridge,
       spaCy static vectors, dense encoders (gte-base, SapBERT, BioLORD-2023, bge-m3; exact flat FAISS) -> union pool + RRF
  -> 5 ICD hierarchy expansion -> 6 LightGBM LambdaRank (code-agnostic features, out-of-fold, code dropout)
  -> 7 ICD-only second pass -> 8 isotonic calibration -> 9 dictionary-validated output + diagnostics
Implemented, not in the final model: MedCPT cross-encoder, contrastive fine-tuning, LLM adjudication.
Not supplied: SNOMED IS-A relationships, SNOMED-to-ICD crosswalk.
```

Diagram: `docs/architecture.png`. Module map and leakage controls: `docs/ARCHITECTURE.md`. Repository tree: `docs/REPO_TREE.md`.

## Setup (clean environment)

```bash
git clone <private repo url> termmap && cd termmap
python -m venv .venv && source .venv/bin/activate          # Python 3.10+ (developed on 3.13)
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
cp .env.example .env                                       # optional; no API key needed
# place the proprietary challenge files (never committed):
#   data/hackathon_train.csv
#   resources/icd10cm_codes.xlsx
#   resources/SNOMEDCT_US_codes_and_descriptions.xlsx
make setup        # verify file hashes, build hash-keyed caches
make final-test   # audit -> splits -> validation / code-disjoint / final test -> models/final.pkl
```

### Environment variables (all optional; see `.env.example`)

`TERMMAP_TRAIN_PATH`, `TERMMAP_ICD_PATH`, `TERMMAP_SNOMED_PATH` (input files), `TERMMAP_CACHE_DIR` (caches),
`TERMMAP_CROSSWALK_PATH`, `TERMMAP_SNOMED_RELATIONSHIPS_PATH` (optional auxiliary resources), `TERMMAP_DENSE_MODEL_PATH`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.

## One command to reproduce predictions

```bash
PYTHONPATH=src python -m termmap.predict --input <eval file> --output predictions.csv --config configs/evaluation.yaml
```

- **Needs:** the three challenge files, `models/final.pkl` (rebuilt automatically from the training file if it is missing),
  and the hash-keyed caches (rebuilt automatically if missing).
- **Output:**
  - predictions in the **same layout as the input**;
  - a wide file with the columns `chunk_id, clinical_entity, icd10cm_code, snomed_code`, in input order, one row per input row;
  - diagnostics (confidence, top-3, agreement, review flag) in a separate file.
- **Without a GPU:** it runs on CPU in sparse + static-vector mode and prints the mode it ran in. Dense encoders are
  used only if their local snapshots and caches are present and enabled in the config.

## GPU stages (Kaggle)

`bash scripts/kaggle_run.sh` runs stages **a-h** in order:

| stage | what it does |
|---|---|
| a | check GPU and Hugging Face access, download the pinned models |
| b | gte-base recall@K |
| c | SapBERT, BioLORD and bge-m3 recall@K |
| d | union and RRF recall |
| e | cross-encoder |
| f | fine-tuning |
| g | ablations with seeds and bootstrap CIs, final test, refit |
| h | reports, error analysis, PDF |

- **Resuming:** every stage resumes from hash-keyed caches. Logs go to `logs/kaggle/<stage>.log`.
- **RAM:** feature building and the dense indexes need several GB. On small machines run one split per process
  (`scripts/train.py --splits val`, then `--splits code_disjoint`, ...).
- **Disk:** embedding caches take roughly 1-2 GB per encoder. Keep caches under `/kaggle/working` with persistence on.

Notebook settings and how to attach models when internet is off: `docs/KAGGLE_RUNBOOK.md`.

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

- **What runs without data:** unit tests. Integration tests skip unless the challenge files and the model exist.
- **What is covered:** schema, dictionary validity, determinism, normalization, qualifiers, hierarchy, leakage,
  dictionary drift, hard cases, fallback modes, the dense code path and the cross-encoder code path.
- **Hard-case failures are reported, not hidden.** They are genuine model errors, e.g. diabetic neuropathy -> E13.40.

## Terminology version drift

```bash
python scripts/drift_report.py --icd <new icd file> --snomed <new snomed file>   # reports/drift_report.md
# point TERMMAP_ICD_PATH / TERMMAP_SNOMED_PATH at the new files, then:
make index && make final-test
```

- **Caches rebuild themselves:** they are keyed by file hash, and the embedding store re-embeds only new or changed descriptions.
- **New codes need no retraining to be predicted:** reranker features are code-agnostic.
- **Retired gold codes are reported, never silently remapped.**

## Determinism

- **Same environment:** two runs produce identical output hashes. This was verified only in the documented environment
  (`reports/determinism.json`).
- **What is fixed:** seeds, exact flat indexes, tie-breaks (score desc, code asc), deterministic LightGBM with fixed
  threads, and cached embeddings / scores.
- **Across hardware:** bit-identical results are not claimed.

## Privacy

- **The training data is proprietary.** It is never committed, published or sent to any API.
- **Enforced before every push:** `.gitignore` and `scripts/check_no_data_tracked.sh` (run it before every push).
- **Shareable reports contain only aggregate numbers and public or synthetic examples.** Row-level analysis stays in
  `reports/private/`, which is git-ignored.
- **Hosted notebook:** the GPU stages ran in a private Kaggle notebook. Organizer approval for hosted notebooks is an open question.

## Limitations

- **The 95% target was not reached** (see the results table).
- **Ranking is the weak step:** union recall @200 is about 0.96-0.97 on validation, but top-1 accuracy is much lower.
- **Abbreviation collisions are only partly solved.**
  - The ICD side fails on the MS (mitral-valve context), RA and PID hard cases.
  - Diabetic neuropathy is coded E13.40 instead of E11.40.
- **Statement-vs-label conflict (HTN):** training labels map it to SNOMED 38341003, the statement shows 59621000. Training labels win.
- **Not evaluated:** the cross-encoder scores were lost in a session reset; fine-tuning and LLM adjudication were not run.
- **CPU-only speed:** about 6 terms/s, 0.34 s median per term, about 5.5 GB peak RAM.

## External resources

| resource | revision | licence |
|---|---|---|
| thenlper/gte-base | c078288308d8dee004ab72c6191778064285ec0c | MIT |
| cambridgeltl/SapBERT-from-PubMedBERT-fulltext | 090663c3ae57bf35ffe4d0d468a2a88d03051a4d | Apache-2.0 |
| FremyCompany/BioLORD-2023 | 167aab527b238a50ca65224e6319215d2ff4fc9f | other: verify on model card |
| BAAI/bge-m3 | 5617a9f61b028005a4858fdac845db406aefb181 | MIT |
| ncbi/MedCPT-Cross-Encoder (implemented, not evaluated) | 71caf65d4927987813984f54c284405a13fcca49 | other: verify on model card |
| spaCy en_core_web_lg 3.8.0 | release wheel | verify on model card |
| LightGBM, scikit-learn, RapidFuzz, FAISS (exact flat index) | pinned in requirements.txt | MIT / BSD-3 / MIT / MIT |
| Supplied ICD-10-CM export; SNOMED CT US export (Term Map gte_base_fp32 index) | file hashes in configs/resources.yaml | SNOMED CT licence applies |

## Open questions for the organizers

1. **Evaluation schema:** long or wide?
2. **Partial credit:** is there any, or exact match only?
3. **Multiple valid codes / conflicts:** how are statement-vs-label conflicts handled?
4. **Annotator agreement:** what is the inter-annotator agreement on the gold labels?
5. **Floors:** are there per-category accuracy floors?
6. **One-label rows:** how are rows with only one gold system scored?
7. **Hosted notebooks:** is running on a hosted notebook (Kaggle) approved?
