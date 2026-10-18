# Repository tree (annotated)

```text
termmap/
├── README.md                 overview, results block (generated), setup, one command, privacy, resources
├── CHANGELOG.md              what changed per iteration
├── Makefile                  setup / train / final-test / errors / test / bench / report / kaggle
├── requirements.txt          pinned CPU deps (+ commented optional GPU deps)
├── pyproject.toml            package metadata, pytest config
├── .gitignore                blocks data, models, caches, secrets, private reports
├── .env.example              optional overrides only; NO API key needed
├── demo.py                   CLI demo: both codes + descriptions, retrieval views, qualifiers, top-3, agreement
├── configs/
│   ├── default.yaml          all parameters (seeds, retrieval, reranker, dense/CE flags default OFF)
│   ├── evaluation.yaml       frozen overlay used by the one-command prediction
│   ├── kaggle_dense.yaml     GPU overlay: dense encoders + optional cross-encoder
│   └── resources.yaml        file hashes, model ids, revisions, licences
├── src/termmap/
│   ├── input_schema.py       stage 1: auto-detect long / wide input
│   ├── io_adapter.py         write predictions in the input's layout + wide file
│   ├── normalize.py          stage 2: multi-view normalization (typo, glue, truncation, OCR, abbreviations)
│   ├── context.py            stage 3: markup strip, mention window, negation / history / uncertainty cues
│   ├── qualifiers.py         laterality, acuity, severity, stage, type, origin; learned token relevance
│   ├── terminology.py        knowledge bases from the supplied dictionaries (ICD hierarchy from code strings)
│   ├── retrieve_lexical.py   exact, BM25, char 3-gram TF-IDF (exact sparse products)
│   ├── retrieve_static.py    spaCy static-vector channel
│   ├── retrieve_dense.py     sentence encoders, resumable embedding store, exact flat FAISS
│   ├── retrieve_training.py  training-partition resources: aliases, neighbours, mined acronyms, priors
│   ├── features.py           candidate pool (union + RRF), hierarchy expansion, code-agnostic features
│   ├── rerank.py             LightGBM LambdaRank + baselines
│   ├── stages.py             stage 1, ICD-only second pass, isotonic calibration
│   ├── cross_encoder.py      implemented, not in final model (OFF by default)
│   ├── specificity.py        specificity profile, dictionary-safe decoding
│   ├── joint.py              ICD/SNOMED agreement signals (never overwrite)
│   ├── crosswalk.py          optional SNOMED->ICD map loader (not supplied)
│   ├── output_validation.py  schema / order / dictionary membership checks
│   ├── diagnostics.py        confidence, top-3, review routing (separate file)
│   ├── evaluation.py         metrics, recall@K, bootstrap CIs, calibration
│   ├── pipeline.py           resources, resumable feature tables, Predictor
│   ├── splits.py             grouped validation, code-disjoint fold, final test split
│   ├── data.py, data_audit.py, util.py
│   └── predict.py            `python -m termmap.predict` (the one command)
├── scripts/
│   ├── audit.py, make_splits.py, build_index.py, setup_resources.py
│   ├── train.py              experiments with seeds, CIs, ablations, refit (per-split processes)
│   ├── evaluate.py           render metrics / ablations; score a prediction file
│   ├── error_analysis.py     error categories + challenge failure modes (counts only)
│   ├── dense_benchmark.py    per-encoder recall@K, union / RRF recall
│   ├── train_cross_encoder.py, finetune_encoder.py   GPU stages (implemented)
│   ├── kaggle_run.sh         GPU stages a-h, resumable
│   ├── benchmark_speed.py, run_checks.py, drift_report.py
│   ├── make_pdf.py           PDF + README results block from docs/results_for_pdf.json
│   ├── make_summary.py, make_experiment_log.py, make_tiny_test_models.py, make_synthetic_eval.py
│   └── check_no_data_tracked.sh   fails if data / models / secrets are tracked
├── tests/                    unit + integration tests (integration skips without data), hard_cases.yaml (public / synthetic)
├── docs/
│   ├── TermMap_writeup.pdf   the 2-3 page documentation
│   ├── results_for_pdf.json  the ONLY place to edit numbers for the PDF / README
│   ├── architecture.png, ARCHITECTURE.md, RESOURCES.md, KAGGLE_RUNBOOK.md
│   ├── demo_script.md, REPO_TREE.md, GITHUB_SETUP.md, FILL_AFTER_RESULTS.md
└── reports/                  shareable aggregate reports (*.json, *.md); reports/private/ is NEVER committed
```

## Must NOT be committed (enforced by .gitignore and scripts/check_no_data_tracked.sh)

- `data/`, `resources/` (challenge files, model snapshots, `resources/cache/` embeddings and indexes)
- `models/` (incl. `models/final.pkl`, feature-table caches, cross-encoder and fine-tuned checkpoints)
- `reports/private/`, `reports/diagnostics.csv`, `outputs/`, `logs/`, any `predictions*` file
- `.env` and any `.env.*` except `.env.example`
- any `*.csv`, `*.tsv`, `*.xlsx`, `*.xls`, `*.pkl`, `*.pickle`, `*.npy`, `*.npz`, `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `*.onnx`, `*.faiss`
