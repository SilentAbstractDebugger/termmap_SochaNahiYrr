# Fill-after-results checklist

Every number in the PDF and in the README results block comes from **`docs/results_for_pdf.json`**.
Updating them is one edit and one command:

```bash
# edit docs/results_for_pdf.json, then:
python scripts/make_pdf.py      # rebuilds docs/TermMap_writeup.pdf, docs/architecture.png and the README results block
```

The script prints how many PENDING fields remain. PENDING values show as yellow markers in the PDF and **bold
PENDING** in the README.

**Status after the Kaggle run:** `results_for_pdf.json` is already filled from `termmap_results.zip` and
`status` is `final`. Only `repo_link` is still PENDING (plus `[DEMO VIDEO LINK]` in the README).

## 1. Copy from the Kaggle run (shareable files only)

Copy into this repo's `reports/`:

- **Copy:** `reports/*.json` and `reports/*.md` (e.g. `metrics.json`, `metrics.md`, `ablations.*`,
  `error_analysis_{val,code_disjoint,final_test}.*`, `dense_benchmark.*`, `benchmark.json`, `determinism.json`,
  `hard_cases.json`, `SUMMARY.md`, `experiment_log.md`, `pytest.xml`).
- **Never copy:** `reports/private/`, `reports/diagnostics.csv`, `models/`, `resources/`, `data/`, `logs/`,
  anything `*.csv` / `*.pkl`.
- Before committing, open `splits.json` and `data_audit.*` and check they contain no real entity or chunk text.

## 2. Fields to fill in `docs/results_for_pdf.json`

| field | source | status |
|---|---|---|
| `status` | set to `final` | done |
| `repo_link`, `demo_video_link` | private repo URL, internal-drive video link | **still PENDING** |
| `results.validation.ci_*`, `n_*` | `reports/metrics.json` -> `val.metrics.<system>.ci95` | done |
| `results.final_test.*` | `reports/metrics.json` -> `final_test` | done (ICD 0.640, SNOMED 0.600, headline 0.600) |
| `results.final_refit` | `models/final.pkl` refit on train+val+test | done |
| `recall.retrieval_miss_vs_rank_miss` | `reports/metrics.json` errors | done |
| `slices_val_snomed.*` | `reports/metrics.md` | done |
| `calibration.*` | `reports/metrics.md` | done |
| `failure_modes_counts` | `reports/error_analysis_val.md` | done |
| `engineering.*` | `reports/benchmark.json`, `reports/pytest.xml`, `reports/determinism.json` | done (GPU speed: not measured) |
| `hard_cases` | `reports/hard_cases.json` | done (9 of 15 pass) |
| changed interim values | final reports; keep earlier CPU-only row as is | done; recall "@200" renamed to "@300" |

## 3. README cells

- **Results block:** do not edit by hand. `make_pdf.py` regenerates it between `<!-- RESULTS:START -->` and
  `<!-- RESULTS:END -->`.
- **By hand:** replace `[DEMO VIDEO LINK]` in the README header, and update the Limitations bullets (see section 4).

## 4. Fix before submitting (new findings from the run)

1. **Determinism is MISMATCH** (`determinism.json`, two different hashes). Do not write "determinism verified";
   the JSON says "NOT confirmed". Fix the README determinism statement and the PDF accordingly.
2. **Speed and RAM:** with the four dense encoders on CPU it is 1.48 terms/s, 6.19 s median, about 25 GB RAM.
   The "6 terms/s, 5.5 GB" figure belongs to the earlier no-dense run. Say so in the README mode notes.
3. **Recall labels:** the fused-union figures are @100 and @300, not @200.
4. **Hard cases:** diabetic neuropathy now passes (E11.40). New failure: "essential HTN, well controlled" (SNOMED
   returned Hyperglycemia). MS, RA and PID (disc) still fail. Update the PDF error analysis and the demo script;
   use MS, RA or essential HTN as the deliberate failing case, not diabetic neuropathy.
5. **`reports/SUMMARY.md`** still says "no GPU, no Hugging Face access, dense encoders NOT RUN". That is stale
   wording from `scripts/make_summary.py`; fix it before shipping the file.
6. **Dense encoders ran "unpinned"**: pin the revisions in the config.
7. **Tests:** 36 of 42 pass; the 6 failures are hard-case assertions. State this or mark them xfail with a reason.

## 5. Final checks

```bash
python scripts/make_pdf.py                         # "PENDING fields left: 0" when complete (1 now: repo_link)
pdfinfo docs/TermMap_writeup.pdf | grep Pages      # must be 2 or 3
bash scripts/check_no_data_tracked.sh              # must print OK
git add -A && git commit -m "Final results" && git push
```

Then paste the repo URL into `repo_link`, the video link into the README and JSON, and re-run `make_pdf.py`.
