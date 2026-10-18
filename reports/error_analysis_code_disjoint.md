# Error analysis (code_disjoint split)

Automatic classification of every error (rules in scripts/error_analysis.py). Counts only;
row-level detail with proprietary text is in reports/private/ (gitignored).

## ICD10CM: 268 errors / 399 rows (before this iteration: 317 / 476, previous split)

| category | n | before |
|---|---|---|
| dictionary/version mismatch | 0 | 0 |
| gold absent from candidates (missing synonym) | 88 | 102 |
| gold absent from candidates (lexical) | 26 | 29 |
| wrong specificity: over-coding | 2 | 2 |
| wrong specificity: under-coding | 7 | 11 |
| hierarchy confusion (same category) | 54 | 66 |
| abbreviation misinterpreted | 8 | 11 |
| qualifier ignored | 3 | 1 |
| negation/history mishandled | 0 | 2 |
| context ignored | 11 | 15 |
| label noise suspicion | 0 | 0 |
| retrieved but misranked | 69 | 78 |

| challenge failure mode (errors touching it) | n |
|---|---|
| FM1 nearest string is the wrong concept | 71 |
| FM2 abbreviation (term has a token in the abbreviation/translation lexicon) | 141 |
| FM3 specificity calibration | 63 |
| FM5 long tail (code unseen in training) | 268 |
| FM6 upstream noise (OOV token in term) | 24 |

## SNOMEDCT_US: 250 errors / 458 rows (before this iteration: 288 / 539, previous split)

| category | n | before |
|---|---|---|
| dictionary/version mismatch | 0 | 0 |
| gold absent from candidates (missing synonym) | 65 | 75 |
| gold absent from candidates (lexical) | 32 | 36 |
| wrong specificity: over-coding | 0 | 0 |
| wrong specificity: under-coding | 0 | 0 |
| hierarchy confusion (same category) | 0 | 0 |
| abbreviation misinterpreted | 13 | 15 |
| qualifier ignored | 1 | 1 |
| negation/history mishandled | 4 | 5 |
| context ignored | 15 | 17 |
| label noise suspicion | 2 | 2 |
| retrieved but misranked | 118 | 137 |

| challenge failure mode (errors touching it) | n |
|---|---|
| FM1 nearest string is the wrong concept | 84 |
| FM2 abbreviation (term has a token in the abbreviation/translation lexicon) | 137 |
| FM5 long tail (code unseen in training) | 250 |
| FM6 upstream noise (OOV token in term) | 29 |

## FM4 granularity (pairs with both labels)

{'pairs_with_both_labels': 83, 'icd_right_snomed_wrong': 8, 'snomed_right_icd_wrong': 18, 'both_wrong': 41, 'both_right': 16}

## Recurring general error patterns

- Retrieval miss without lexical overlap (lay paraphrase vs clinical description): the 'missing synonym' rows; needs semantic (dense) retrieval.
- Near-synonym misranking where the gold follows a coder convention seen once or never (e.g. finding vs disorder concept, 'other' vs 'unspecified' leaf).
- Abbreviation sense chosen from string evidence when the context word is outside the window.
- Specificity within the right 3-character category (hierarchy confusion).

Per-step effects are in reports/experiment_log.md; iteration-1 fixes in reports/dev_log.json.

