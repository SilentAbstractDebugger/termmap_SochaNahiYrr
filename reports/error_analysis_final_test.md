# Error analysis (final_test split)

Automatic classification of every error (rules in scripts/error_analysis.py). Counts only;
row-level detail with proprietary text is in reports/private/ (gitignored).

## ICD10CM: 141 errors / 353 rows

| category | n | before |
|---|---|---|
| dictionary/version mismatch | 0 |  |
| gold absent from candidates (missing synonym) | 45 |  |
| gold absent from candidates (lexical) | 11 |  |
| wrong specificity: over-coding | 0 |  |
| wrong specificity: under-coding | 0 |  |
| hierarchy confusion (same category) | 26 |  |
| abbreviation misinterpreted | 10 |  |
| qualifier ignored | 1 |  |
| negation/history mishandled | 0 |  |
| context ignored | 10 |  |
| label noise suspicion | 4 |  |
| retrieved but misranked | 34 |  |

| challenge failure mode (errors touching it) | n |
|---|---|
| FM1 nearest string is the wrong concept | 40 |
| FM2 abbreviation (term has a token in the abbreviation/translation lexicon) | 84 |
| FM3 specificity calibration | 26 |
| FM5 long tail (code unseen in training) | 44 |
| FM6 upstream noise (OOV token in term) | 10 |

## SNOMEDCT_US: 190 errors / 402 rows

| category | n | before |
|---|---|---|
| dictionary/version mismatch | 0 |  |
| gold absent from candidates (missing synonym) | 36 |  |
| gold absent from candidates (lexical) | 26 |  |
| wrong specificity: over-coding | 0 |  |
| wrong specificity: under-coding | 0 |  |
| hierarchy confusion (same category) | 0 |  |
| abbreviation misinterpreted | 11 |  |
| qualifier ignored | 1 |  |
| negation/history mishandled | 6 |  |
| context ignored | 15 |  |
| label noise suspicion | 0 |  |
| retrieved but misranked | 95 |  |

| challenge failure mode (errors touching it) | n |
|---|---|
| FM1 nearest string is the wrong concept | 65 |
| FM2 abbreviation (term has a token in the abbreviation/translation lexicon) | 110 |
| FM5 long tail (code unseen in training) | 108 |
| FM6 upstream noise (OOV token in term) | 21 |

## FM4 granularity (pairs with both labels)

{'pairs_with_both_labels': 323, 'icd_right_snomed_wrong': 66, 'snomed_right_icd_wrong': 58, 'both_wrong': 66, 'both_right': 133}

## Recurring general error patterns

- Retrieval miss without lexical overlap (lay paraphrase vs clinical description): the 'missing synonym' rows; needs semantic (dense) retrieval.
- Near-synonym misranking where the gold follows a coder convention seen once or never (e.g. finding vs disorder concept, 'other' vs 'unspecified' leaf).
- Abbreviation sense chosen from string evidence when the context word is outside the window.
- Specificity within the right 3-character category (hierarchy confusion).

Per-step effects are in reports/experiment_log.md; iteration-1 fixes in reports/dev_log.json.

