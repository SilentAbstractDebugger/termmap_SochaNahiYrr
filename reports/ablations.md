# Ablations (iteration 2)

Mean +- std over reranker seeds; same candidate tables and splits per experiment. Differences < 2 points are noise unless bootstrap CIs separate (CIs in metrics.md).

## val

| configuration | ICD mean +- std | SNOMED mean +- std | headline (mean) | ICD unseen | SNOMED unseen |
|---|---|---|---|---|---|
| stage1 | 0.5414 +- 0.0064 | 0.5044 +- 0.0017 | 0.5044 | 0.375 | 0.4152 |
| stage2_icd_only_calibrated | 0.5667 +- 0.0074 | 0.5044 +- 0.0017 | 0.5044 | 0.401 | 0.4152 |
| heuristic: bm25_only | 0.1723 | 0.3545 | 0.1723 | 0.1875 | 0.3737 |
| heuristic: char_only | 0.226 | 0.403 | 0.226 | 0.2188 | 0.3875 |
| heuristic: dense_only | 0.1857 | 0.2892 | 0.1857 | 0.1484 | 0.2872 |
| heuristic: exact_fuzzy | 0.1767 | 0.2705 | 0.1767 | 0.1719 | 0.2803 |
| heuristic: hybrid_sum | 0.3289 | 0.3246 | 0.3246 | 0.1797 | 0.2457 |
| heuristic: rrf_only | 0.2438 | 0.3209 | 0.2438 | 0.2266 | 0.301 |

## code_disjoint

| configuration | ICD mean +- std | SNOMED mean +- std | headline (mean) | ICD unseen | SNOMED unseen |
|---|---|---|---|---|---|
| stage1 | 0.3275 +- 0.0072 | 0.4505 +- 0.0037 | 0.3275 | 0.3275 | 0.4505 |
| stage2_icd_only_calibrated | 0.3266 +- 0.0042 | 0.4505 +- 0.0037 | 0.3266 | 0.3266 | 0.4505 |
| heuristic: bm25_only | 0.1654 | 0.3341 | 0.1654 | 0.1654 | 0.3341 |
| heuristic: char_only | 0.2105 | 0.4039 | 0.2105 | 0.2105 | 0.4039 |
| heuristic: dense_only | 0.1855 | 0.2642 | 0.1855 | 0.1855 | 0.2642 |
| heuristic: exact_fuzzy | 0.1604 | 0.2249 | 0.1604 | 0.1604 | 0.2249 |
| heuristic: hybrid_sum | 0.1955 | 0.2293 | 0.1955 | 0.1955 | 0.2293 |
| heuristic: rrf_only | 0.208 | 0.321 | 0.208 | 0.208 | 0.321 |

## final_test

| configuration | ICD mean +- std | SNOMED mean +- std | headline (mean) | ICD unseen | SNOMED unseen |
|---|---|---|---|---|---|
| stage2_icd_only_calibrated | 0.6034 +- 0.0028 | 0.5373 +- 0.01 | 0.5373 | 0.3904 | 0.4278 |
| heuristic: bm25_only | 0.2011 | 0.3383 | 0.2011 | 0.2466 | 0.3369 |
| heuristic: char_only | 0.2635 | 0.3905 | 0.2635 | 0.2603 | 0.3583 |
| heuristic: dense_only | 0.1756 | 0.2687 | 0.1756 | 0.1781 | 0.262 |
| heuristic: exact_fuzzy | 0.2096 | 0.2264 | 0.2096 | 0.2329 | 0.2299 |
| heuristic: hybrid_sum | 0.4051 | 0.3507 | 0.3507 | 0.2055 | 0.2086 |
| heuristic: rrf_only | 0.2408 | 0.3109 | 0.2408 | 0.2055 | 0.246 |

Not run in this environment (no GPU / no Hugging Face access): +gte-base, +SapBERT, +BioLORD, +MedCPT, +bge-m3 dense channels, +fine-tuned encoder, +cross-encoder, +LLM. Rows appear here after `scripts/kaggle_run.sh`.

