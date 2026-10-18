# TermMap: clinical term mapping to ICD-10-CM and SNOMED CT US

TermMap takes one extracted clinical entity (diagnoses, conditions, signs and symptoms) and its chunk text, and returns
**one ICD-10-CM code and one SNOMED CT US concept id**. Both codes are guaranteed to exist in the supplied dictionaries.
It runs locally and offline after setup. It needs no API key and never sends data anywhere.

> **Honest status.** The 95% target on both systems was **not reached**. The numbers below are final: the frozen
> configuration was scored once on the held-out final test split, and `models/final.pkl` was then refit on
> train+val+test. Write-up: [`Documentation`](https://drive.google.com/file/d/1Y7R2CihC5K0tMIa4llPmJRG0-p9_KYZY/view?usp=sharing).


## Results

<!-- RESULTS:START -->
_Status: **final** (iteration 2, frozen configuration `stage2_icd_only_calibrated`). Exact-match accuracy on the primary
seed with 95% bootstrap CIs (1000 resamples). Source: `reports/metrics.json`, `reports/SUMMARY.md`._

| split | ICD-10-CM | SNOMED CT | headline (lower) | n (ICD / SNOMED) |
|---|---|---|---|---|
| Validation (codes seen; 3 seeds) | 0.6085 (CI [0.5659, 0.6511]) | 0.5672 (CI [0.5261, 0.6063]) | 0.5672 | 447 / 536 |
| Code-disjoint (codes never seen; 2 seeds) | 0.4361 (CI [0.3885, 0.4837]) | 0.5349 (CI [0.4869, 0.5808]) | 0.4361 | 399 / 458 |
| Final test (scored once; 2 seeds) | 0.6402 (CI [0.5921, 0.6912]) | 0.5995 (CI [0.5522, 0.6468]) | 0.5995 | 353 / 402 |

Seed means (ICD / SNOMED): validation 0.618 ± 0.007 / 0.565 ± 0.006; code-disjoint 0.439 ± 0.003 / 0.538 ± 0.003;
final test 0.643 ± 0.003 / 0.601 ± 0.001.

Target 95% on both systems was **not reached**. The gap on the headline (lower of the two systems) is 0.383 on validation,
0.514 on code-disjoint and 0.351 on the final test. Candidate recall (a shortlist ceiling, not accuracy), per-channel
union on validation @200: ICD 0.964, SNOMED 0.974; code-disjoint @200: ICD 0.937, SNOMED 0.954. The fused pool the
ranker actually sees holds 0.911 (ICD) and 0.918 (SNOMED) of gold codes on validation.
Final refit: done (`models/final.pkl`, train+val+test, frozen configuration; the final-test numbers above are from the
model trained **without** the test split). Demo video: [[DEMO VIDEO](https://drive.google.com/file/d/1pwAeCZpRqfy7EAgY4hc2QXVSw1okTx9E/view?usp=sharing)].

Change from iteration 1 (same splits, 3 seeds): validation ICD 0.543 → 0.618 and SNOMED 0.496 → 0.565; code-disjoint
ICD 0.321 → 0.439 and SNOMED 0.447 → 0.538.
<!-- RESULTS:END -->

The candidate-pool recall above is a shortlist ceiling and not an accuracy figure. A single encoder's top-1 is much lower
(ICD 0.20-0.33, SNOMED 0.43-0.47 on validation), so ranking is the weak step. In the validation ablation, removing the dense features costs about 5 points on ICD and 4 on SNOMED, and removing the ICD hierarchy features about 4 points on ICD; the ICD-only
second pass adds about 1 point on ICD, which is within noise. Full tables: `reports/metrics.md`, `reports/ablations.md`, `reports/error_analysis_*.md`, `reports/SUMMARY.md`.

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
  and the hash-keyed caches (rebuilt automatically if missing).models/final.pkl is not in git. Without the dense snapshots and caches the run falls back to sparse + static-vector mode and will not match the submitted predictions. Bit-identical re-runs are not guaranteed
  
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
| a | check GPU and Hugging Face access, download the models |
| b | gte-base recall@K |
| c | SapBERT, BioLORD and bge-m3 recall@K |
| d | union and RRF recall |
| e | cross-encoder (implemented, not in the final model)|
| f | fine-tuning (implemented, not run)|
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
- **Latest run: 36 of 42 pass.** The six failures are all hard cases: `essential_htn_well_controlled`, `ms_mitral`,
  `ms_neuro`, `ra_joint`, `ra_atrium`, `pid_disc` (9 of 15 hard cases pass).
- **Hard-case failures are reported, not hidden.** They are genuine model errors, e.g. "MS" -> aortic stenosis (I35.0).

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

- **Latest check failed:** the two-run hash check in `reports/determinism.json` on the final configuration returned
  **MISMATCH** (CPU-only run, Python 3.13). An earlier sparse + static-vector run (iteration 1) did give identical hashes.
  Treat bit-identical output as **not currently verified** for the final configuration.
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
- **Ranking is the weak step:** union recall @200 is about 0.96-0.97 on validation, but exact match is 0.61 (ICD) and
  0.57 (SNOMED). Of validation errors, 135 (ICD) and 188 (SNOMED) are rank misses; 40 and 44 are retrieval misses.
- **Long tail:** on codes never seen in training, accuracy is 0.42 (ICD) and 0.51 (SNOMED) on validation, and 0.44 / 0.53
  on the code-disjoint fold.
- **Noisy and abbreviated terms are weaker:** on validation, noisy / out-of-vocabulary terms score 0.55 (ICD) and 0.40
  (SNOMED); history-flagged terms score 0.53 and 0.40.
- **Abbreviation collisions are only partly solved.** The hard cases for MS (mitral-valve and neurological context), RA
  and PID still fail. Diabetic neuropathy now passes (E11.40).
- **Statement-vs-label conflict (HTN):** training labels map "HTN" to SNOMED 38341003, the statement shows 59621000.
  Training labels win. The related case "essential HTN, well controlled" still fails (SNOMED returned 80394007, expected
  59621000).
- **Not evaluated:** the MedCPT cross-encoder, contrastive fine-tuning and LLM adjudication were not run for accuracy.
- **CPU-only speed (final configuration, 2 threads):** 1.48 terms/s for both systems on a batch of 576, single-term
  latency median 6.2 s (p95 8.7 s), peak RAM about 25 GB. GPU throughput was not measured. The earlier sparse +
  static-vector configuration was much faster (about 7.7 terms/s, 0.3 s median, 4.4 GB).

## External resources

| resource | revision | licence |
|---|---|---|
| thenlper/gte-base | c078288308d8dee004ab72c6191778064285ec0c | MIT |
| cambridgeltl/SapBERT-from-PubMedBERT-fulltext | 090663c3ae57bf35ffe4d0d468a2a88d03051a4d | Apache-2.0 |
| FremyCompany/BioLORD-2023 | 167aab527b238a50ca65224e6319215d2ff4fc9f | other: verify on model card |
| BAAI/bge-m3 | 5617a9f61b028005a4858fdac845db406aefb181 | MIT |
| ncbi/MedCPT-Cross-Encoder (implemented, not evaluated) | 71caf65d4927987813984f54c284405a13fcca49 | other: verify on model card |
| spaCy en_core_web_lg 3.8.0 | release wheel | MIT (vectors CC0), read from the packaged meta.json |
| LightGBM, scikit-learn, RapidFuzz, FAISS (exact flat index) | pinned in requirements.txt | MIT / BSD-3 / MIT / MIT |
| Supplied ICD-10-CM export; SNOMED CT US export (Term Map gte_base_fp32 index) | file hashes in configs/resources.yaml | SNOMED CT licence applies |

