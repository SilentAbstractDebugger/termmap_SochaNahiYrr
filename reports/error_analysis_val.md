# Error analysis (val split)

Automatic classification of every error (rules in scripts/error_analysis.py). Counts only;
row-level detail with proprietary text is in reports/private/ (gitignored).

## ICD10CM: 193 errors / 447 rows (before this iteration: 208 / 447, previous split)

| category | n | before |
|---|---|---|
| dictionary/version mismatch | 0 | 0 |
| gold absent from candidates (missing synonym) | 65 | 64 |
| gold absent from candidates (lexical) | 11 | 13 |
| wrong specificity: over-coding | 4 | 5 |
| wrong specificity: under-coding | 1 | 1 |
| hierarchy confusion (same category) | 23 | 26 |
| abbreviation misinterpreted | 11 | 10 |
| qualifier ignored | 0 | 1 |
| negation/history mishandled | 3 | 2 |
| context ignored | 9 | 14 |
| label noise suspicion | 7 | 5 |
| retrieved but misranked | 59 | 67 |

| challenge failure mode (errors touching it) | n |
|---|---|
| FM1 nearest string is the wrong concept | 63 |
| FM2 abbreviation (term has a token in the abbreviation/translation lexicon) | 89 |
| FM3 specificity calibration | 28 |
| FM5 long tail (code unseen in training) | 76 |
| FM6 upstream noise (OOV token in term) | 18 |

## SNOMEDCT_US: 265 errors / 536 rows (before this iteration: 262 / 536, previous split)

| category | n | before |
|---|---|---|
| dictionary/version mismatch | 0 | 0 |
| gold absent from candidates (missing synonym) | 53 | 53 |
| gold absent from candidates (lexical) | 34 | 31 |
| wrong specificity: over-coding | 0 | 0 |
| wrong specificity: under-coding | 0 | 0 |
| hierarchy confusion (same category) | 0 | 0 |
| abbreviation misinterpreted | 17 | 16 |
| qualifier ignored | 1 | 0 |
| negation/history mishandled | 6 | 4 |
| context ignored | 17 | 18 |
| label noise suspicion | 3 | 4 |
| retrieved but misranked | 134 | 136 |

| challenge failure mode (errors touching it) | n |
|---|---|
| FM1 nearest string is the wrong concept | 102 |
| FM2 abbreviation (term has a token in the abbreviation/translation lexicon) | 135 |
| FM5 long tail (code unseen in training) | 168 |
| FM6 upstream noise (OOV token in term) | 33 |

## FM4 granularity (pairs with both labels)

{'pairs_with_both_labels': 407, 'icd_right_snomed_wrong': 67, 'snomed_right_icd_wrong': 64, 'both_wrong': 108, 'both_right': 168}

## Recurring general error patterns

- Retrieval miss without lexical overlap (lay paraphrase vs clinical description): the 'missing synonym' rows; needs semantic (dense) retrieval.
- Near-synonym misranking where the gold follows a coder convention seen once or never (e.g. finding vs disorder concept, 'other' vs 'unspecified' leaf).
- Abbreviation sense chosen from string evidence when the context word is outside the window.
- Specificity within the right 3-character category (hierarchy confusion).

Per-step effects are in reports/experiment_log.md; iteration-1 fixes in reports/dev_log.json.

