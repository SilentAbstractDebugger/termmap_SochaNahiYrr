# Fill-after-results checklist

Every number in the PDF and in the README results block comes from **`docs/results_for_pdf.json`**.
Updating them is one edit and one command:

```bash
# edit docs/results_for_pdf.json, then:
python scripts/make_pdf.py      # rebuilds docs/TermMap_writeup.pdf, docs/architecture.png and the README results block
```

The script prints how many PENDING fields remain. PENDING values show as yellow markers in the PDF and **bold
PENDING** in the README.

## 1. Copy from the Kaggle run (shareable files only)

Copy into this repo's `reports/`:

- **Copy:** `reports/*.json` and `reports/*.md` (e.g. `kaggle_metrics.json` / `metrics.json`, `metrics.md`,
  `ablations.*`, `error_analysis_{val,code_disjoint,final_test}.*`, `dense_benchmark.*`, `benchmark.json`,
  `determinism.json`, `hard_cases.json`, `SUMMARY.md`, `experiment_log.md`, `pytest.xml`).
- **Never copy:** `reports/private/`, `reports/diagnostics.csv`, `models/`, `resources/`, `data/`, `logs/`,
  anything `*.csv` / `*.pkl`.

## 2. Fields to fill in `docs/results_for_pdf.json`

| field | source |
|---|---|
| `status` | set to `final` |
| `repo_link`, `demo_video_link` | private repo URL, internal-drive video link |
| `results.validation.ci_icd`, `ci_snomed`, `n_icd`, `n_snomed` | `reports/metrics.json` -> `val.metrics.<system>.ci95` |
| `results.final_test.*` (icd, snomed, headline, seeds, ci_icd, ci_snomed, n_icd, n_snomed) | `reports/metrics.json` -> `final_test` |
| `results.final_refit` | e.g. "refit on all pairs, frozen config" once `models/final.pkl` is rebuilt |
| `recall.retrieval_miss_vs_rank_miss` | `reports/metrics.json` -> `val.metrics.<system>.errors` |
| `slices_val_snomed.by_train_freq`, `slices_val_snomed.abbreviation_terms` | `reports/metrics.md` (validation section) |
| `calibration.ece_val_icd`, `ece_val_snomed`, `coverage_at_conf_0.7` | `reports/metrics.md` (ECE and coverage vs accuracy rows) |
| `failure_modes_counts` | `reports/error_analysis_val.md` (failure-mode table, short form "FM1 x / y; ...") |
| `engineering.gpu_terms_per_second`, `engineering.tests` | `reports/benchmark.json` (GPU run), `reports/pytest.xml` summary |
| `hard_cases` rows still marked PENDING | `reports/hard_cases.json` from the final model |
| any changed interim value (`results.validation.*`, `ablations_val`, `recall.*`) | the final reports; keep the earlier CPU-only row as is |

## 3. README cells

- **Results block:** do not edit it by hand. `make_pdf.py` regenerates it between `<!-- RESULTS:START -->` and
  `<!-- RESULTS:END -->`.
- **By hand:** replace `[DEMO VIDEO LINK]` in the README header, and update the Limitations bullets if final
  numbers change any statement.

## 4. Final checks

```bash
python scripts/make_pdf.py                         # "PENDING fields left: 0" when complete
pdfinfo docs/TermMap_writeup.pdf | grep Pages      # must be 2 or 3
bash scripts/check_no_data_tracked.sh              # must print OK
git add -A && git commit -m "Final results" && git push
```
