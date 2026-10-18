# Data audit (Phase 0)

Aggregate statistics only; no proprietary row text.

- Columns: `['entity', 'chunk_text', 'coding_system', 'code', 'display', 'subconcept']`; rows: **9646**; full duplicate rows: 0
- subconcept: {'DiagnosisCode': 5005, 'ProcedureCode': 2778, 'LabService': 858, 'EncounterType': 567, 'Device': 437, 'MedicationProfileMedication': 1}
- Diagnosis rows by system: {'SNOMEDCT_US': 2700, 'ICD10CM': 2305}; duplicate (entity,chunk,system): 0
- Diagnosis entity+chunk pairs: 2882 (both labels 2123, ICD only 182, SNOMED only 577)
- Empty chunk_text pairs: 3; chunk length quantiles: {'min': 0.0, 'p25': 82.0, 'median': 156.0, 'p75': 404.0, 'p95': 1065.699999999999, 'max': 2112.0}
- Chunks with `header_` markup: 229, with `<br>`: 157
- Distinct entities: 2778 (normalized 2766); entities appearing in >1 chunk: 104

## ICD10CM

- buckets: {'distinct': 866, 'codes_freq_1': 465, 'codes_freq_2_5': 297, 'codes_freq_6p': 104, 'share_codes_singleton': 0.537, 'share_rows_singleton_code': 0.2017}
- entities_with_multiple_codes: 0
- norm_entities_with_multiple_codes: 0
- duplicate_entity_strings: 0
- gold_in_dict_exact: 2303
- gold_in_dict_normalized: 2303
- gold_not_in_dict: [('L6.0', 2)]
- gold_is_category_3char: 117
- gold_is_leaf: 2260
- gold_non_leaf: 43
- gold_is_range: 0
- gold_depth_counts: [(4, 1068), (5, 775), (6, 235), (3, 117), (7, 108)]
- gold_chapter_letter_counts: [('R', 690), ('M', 293), ('I', 292), ('J', 143), ('E', 131), ('K', 102), ('L', 84), ('Z', 79), ('G', 79), ('N', 68), ('H', 62), ('C', 50), ('S', 50), ('F', 46), ('T', 37), ('B', 30), ('D', 28), ('W', 14), ('A', 8), ('U', 7), ('Q', 6), ('O', 2), ('X', 2)]
- dict_codes_with_multiple_desc: 53874
- dict_range_headers: 111
- display_matches_gold_code_desc: 2200
- display_not_in_gold_code_desc: 103
- display_not_anywhere_in_dict: 73
- entity_exact_desc_of_gold: 102
- entity_exact_desc_any_code: 155

## SNOMEDCT_US

- buckets: {'distinct': 1669, 'codes_freq_1': 1199, 'codes_freq_2_5': 427, 'codes_freq_6p': 43, 'share_codes_singleton': 0.7184, 'share_rows_singleton_code': 0.4441}
- entities_with_multiple_codes: 0
- norm_entities_with_multiple_codes: 1
- duplicate_entity_strings: 0
- gold_in_dict_exact: 2700
- gold_in_dict_normalized: 2700
- gold_not_in_dict: []
- gold_semantic_tags: [('disorder', 1299), ('finding', 1055), ('morphologic abnormality', 72), ('situation', 65), ('observable entity', 52), ('procedure', 35), ('body structure', 34), ('qualifier value', 30), ('substance', 11), ('regime/therapy', 10), ('event', 10), ('organism', 7), ('physical object', 6), ('occupation', 3), ('environment', 2), ('assessment scale', 2), ('tumor staging', 1), ('navigational concept', 1), ('record artifact', 1), ('person', 1), ('physical force', 1), ('attribute', 1), ('clinical drug', 1)]
- dict_semantic_tags_top: [('disorder', 92887), ('procedure', 57843), ('finding', 37342), ('body structure', 36349), ('organism', 34952), ('substance', 27983), ('physical object', 14051), ('qualifier value', 11281), ('observable entity', 11001), ('medicinal product', 8565), ('clinical drug', 8355), ('medicinal product form', 7272), ('situation', 5577), ('morphologic abnormality', 5155), ('occupation', 3526), ('event', 3403), ('regime/therapy', 3153), ('specimen', 1867), ('assessment scale', 1539), ('product', 1520), ('environment', 1244), ('attribute', 1193), ('cell', 716), ('geographic location', 659), ('navigational concept', 617)]
- dict_concepts_with_multiple_desc: 382122
- dict_concepts_without_fsn_tag: 363
- display_matches_gold_code_desc: 2700
- display_not_in_gold_code_desc: 0
- display_not_anywhere_in_dict: 0
- entity_exact_desc_of_gold: 512
- entity_exact_desc_any_code: 590

## Discrepancies vs the brief (section 2)

- none

Eval-split signals in train: {'split_like_columns': [], 'id_like_columns': []}
