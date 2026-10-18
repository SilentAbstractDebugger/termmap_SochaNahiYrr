# Changelog

## [2.0.0] - 2026-10-18 - iteration 2 (recall / ranking / measurement hygiene)

Environment: CPU-only sandbox (2 cores, 8 GB), **no GPU, no Hugging Face access** (HTTP 403). Steps that
need neural model weights were implemented, made resumable and tested with tiny random local models, but
NOT RUN for accuracy. All numbers: `reports/SUMMARY.md`, `reports/metrics.md`, `reports/ablations.md`,
`reports/experiment_log.md`.

### Added
- **Measurement hygiene (Step 0)**: FINAL TEST split (15% of groups, carved from non-validation groups,
  excluded from every training resource and from the code-disjoint experiment, scored once); 3 reranker
  seeds (mean +- std); 1000x bootstrap 95% CIs incl. a CI for the min-headline; experiments can run in
  separate processes (`--splits`) and merge into one report.
- **Resumable stages**: hash-keyed on-disk feature tables per OOF fold / eval set (`models/cache/`),
  string-level append-only embedding store with periodic flushes, checkpointed fine-tuning and
  cross-encoder training, cached cross-encoder scores; time-box progress logging.
- **Step 1 (code, NOT RUN)**: multi-encoder dense retrieval (`dense.models`), fp16 GPU encoding, exact
  `IndexFlatIP`, ICD "path" enrichment texts, entity+context query view, asymmetric encoders (MedCPT),
  CLS pooling; incremental re-embedding on dictionary change; `scripts/dense_benchmark.py` (per-model
  recall@{1..300}, union with the sparse pool, gte-base top-1 diagnostic); `configs/kaggle_dense.yaml`.
- **RRF fusion** of all channel ranks as a reranker feature (raw scores/ranks kept).
- **Step 2**: OCR glyph-confusion repair views (`1/l`, `0/o`, `rn/m`, ...), applied before letter/digit
  splitting; train-time noise augmentation (25% corrupted copies of training queries: typo, deletion,
  glue, truncation, OCR, punctuation).
- **Step 3 (code, NOT RUN)**: `scripts/finetune_encoder.py` (MNRL + hard negatives, synonym pairs, noisy
  anchors, strict exclusion of code-disjoint codes / validation / test rows with assertions, early stopping).
- **Step 4 (code, NOT RUN)**: `src/termmap/cross_encoder.py`, `scripts/train_cross_encoder.py` (out-of-fold
  score tables, top-N pools, merged as `ce_score/ce_rank/ce_gap_top` features; live scoring at inference).
- **Step 5**: learned conditional priors P(SNOMED tag | entity tokens), P(ICD letter / unspecified leaf /
  depth | tokens); origin qualifier (congenital / acquired / rheumatic / nonrheumatic); context-window vs
  candidate-description char cosine; **stage-2 joint reranker** (other system's first-pass top-1 text
  agreement, probability, margin; trained on OOF stage-1 scores; per-system switch).
- **Step 7**: isotonic calibration fit on OOF predictions; ECE before/after; `route_for_review` flag in
  diagnostics (low support, small margin, ICD/SNOMED disagreement).
- `scripts/kaggle_run.sh`, `scripts/make_tiny_test_models.py`, `scripts/make_experiment_log.py`,
  `tests/test_dense.py` (dense build/load offline, cached-embedding determinism, incremental drift,
  named dense features, cross-encoder train/score/cache/merge, calibration monotonicity, fine-tune leakage).

### Changed
- Static-vector index loaded into RAM (memory-mapped matmul was the bottleneck).
- Retrieval result caches are cleared between folds to bound memory.
- Hard case `htn` marked as a statement-vs-data conflict (training labels win; SNOMED checked for
  plausibility); listed in the open questions.
- Licence of spaCy `en_core_web_lg` confirmed from the packaged metadata (MIT; vectors CC0).
- Reports: `metrics.md` with CIs, new `ablations.md/json`, `experiment_log.md`, three error-analysis files
  with before/after counts; PDF rebuilt with a large legible architecture figure.

### Rejected (measured)
- Training-table negative subsampling to the top-60 first-stage candidates (ICD val 0.54 -> 0.32).
- See `reports/experiment_log.md` for components kept/rejected by their 3-seed numbers.
