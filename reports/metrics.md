# Measured results (iteration 2)

Run mode: `{'cross_encoder': 'disabled', 'crosswalk': 'not supplied', 'dense': 'static_vectors:en_core_web_lg', 'lexical': True, 'reranker': True, 'snomed_hierarchy': 'not supplied (no IS-A features)', 'training_neighbours': True}`  
Reranker seeds: {'code_disjoint': [13, 14, 15], 'final_test': [13, 14], 'val': [13, 14, 15]}

## val (main variant: stage2_icd_only_calibrated, primary seed)

| metric | ICD-10-CM | SNOMED CT |
|---|---|---|
| exact match [95% bootstrap CI] | 0.5682 [0.5235, 0.6175] | 0.5056 [0.4646, 0.5504] |
| headline (min) [CI] | 0.5056 [0.4627, 0.5459]; gap to 95%: 0.4444 | |
| acc, code train-freq 0 | {'acc': 0.4062, 'n': 128} | {'acc': 0.4187, 'n': 289} |
| acc, code train-freq 1 | {'acc': 0.5926, 'n': 81} | {'acc': 0.5983, 'n': 117} |
| acc, code train-freq 2-5 | {'acc': 0.6352, 'n': 159} | {'acc': 0.6042, 'n': 96} |
| acc, code train-freq 6+ | {'acc': 0.6709, 'n': 79} | {'acc': 0.6471, 'n': 34} |
| candidate pool recall | 0.83 | 0.8377 |
| errors: retrieval / rank miss | {'rank_miss': 117, 'retrieval_miss': 76, 'total': 193} | {'rank_miss': 178, 'retrieval_miss': 87, 'total': 265} |
| first-stage recall@K | {'@1': 0.3289, '@10': 0.604, '@100': 0.7808, '@20': 0.6711, '@200': 0.83, '@5': 0.5414, '@50': 0.7271} | {'@1': 0.3246, '@10': 0.6175, '@100': 0.8134, '@20': 0.6679, '@200': 0.8377, '@5': 0.5541, '@50': 0.7537} |
| ECE (reported prob) | 0.1061 | 0.043 |
| ECE before calibration | 0.1571 | 0.1619 |
| coverage vs acc | {'0.0': {'acc': 0.5682, 'coverage': 1.0}, '0.3': {'acc': 0.7074, 'coverage': 0.6957}, '0.5': {'acc': 0.7982, 'coverage': 0.4877}, '0.7': {'acc': 0.8913, 'coverage': 0.3087}, '0.9': {'acc': 0.9348, 'coverage': 0.1029}} | {'0.0': {'acc': 0.5056, 'coverage': 1.0}, '0.3': {'acc': 0.6829, 'coverage': 0.653}, '0.5': {'acc': 0.8099, 'coverage': 0.4907}, '0.7': {'acc': 0.9346, 'coverage': 0.2854}, '0.9': {'acc': 0.9798, 'coverage': 0.1847}} |
| abbreviation_terms | {'acc': 0.5762, 'n': 210} | {'acc': 0.4375, 'n': 240} |
| noisy_oov_terms | {'acc': 0.4545, 'n': 33} | {'acc': 0.3654, 'n': 52} |
| negated_terms | {'acc': 0.6133, 'n': 75} | {'acc': 0.4712, 'n': 104} |
| history_terms | {'acc': 0.4211, 'n': 19} | {'acc': 0.3, 'n': 20} |
| empty_context | {'acc': None, 'n': 0} | {'acc': 0.0, 'n': 1} |
| ICD same category / same letter (partial credit, separate) | 0.651 / 0.7875 | n/a |
| ICD over / under coding rate | 0.0089 / 0.0022 | n/a |
| ICD letters below overall | ['C', 'F', 'H', 'I', 'M', 'Z'] | |
| SNOMED hierarchy distance | n/a | not computed: no SNOMED relationships supplied |

## code_disjoint (main variant: stage2_icd_only_calibrated, primary seed)

| metric | ICD-10-CM | SNOMED CT |
|---|---|---|
| exact match [95% bootstrap CI] | 0.3283 [0.2857, 0.3759] | 0.4541 [0.4105, 0.5022] |
| headline (min) [CI] | 0.3283 [0.2857, 0.3759]; gap to 95%: 0.6217 | |
| acc, code train-freq 0 | {'acc': 0.3283, 'n': 399} | {'acc': 0.4541, 'n': 458} |
| acc, code train-freq 1 | None | None |
| acc, code train-freq 2-5 | None | None |
| acc, code train-freq 6+ | None | None |
| candidate pool recall | 0.7143 | 0.7882 |
| errors: retrieval / rank miss | {'rank_miss': 154, 'retrieval_miss': 114, 'total': 268} | {'rank_miss': 153, 'retrieval_miss': 97, 'total': 250} |
| first-stage recall@K | {'@1': 0.1955, '@10': 0.4386, '@100': 0.6617, '@20': 0.5263, '@200': 0.7143, '@5': 0.3684, '@50': 0.6065} | {'@1': 0.2293, '@10': 0.5502, '@100': 0.7664, '@20': 0.6288, '@200': 0.7882, '@5': 0.4585, '@50': 0.7227} |
| ECE (reported prob) | 0.0506 | 0.0656 |
| ECE before calibration | 0.2829 | 0.1748 |
| coverage vs acc | {'0.0': {'acc': 0.3283, 'coverage': 1.0}, '0.3': {'acc': 0.4808, 'coverage': 0.5213}, '0.5': {'acc': 0.5619, 'coverage': 0.2632}, '0.7': {'acc': 0.7692, 'coverage': 0.0977}, '0.9': {'acc': 0.9231, 'coverage': 0.0326}} | {'0.0': {'acc': 0.4541, 'coverage': 1.0}, '0.3': {'acc': 0.6854, 'coverage': 0.583}, '0.5': {'acc': 0.8476, 'coverage': 0.3581}, '0.7': {'acc': 0.9052, 'coverage': 0.2533}, '0.9': {'acc': 0.963, 'coverage': 0.059}} |
| abbreviation_terms | {'acc': 0.2843, 'n': 197} | {'acc': 0.4195, 'n': 236} |
| noisy_oov_terms | {'acc': 0.04, 'n': 25} | {'acc': 0.383, 'n': 47} |
| negated_terms | {'acc': 0.4412, 'n': 68} | {'acc': 0.4884, 'n': 86} |
| history_terms | {'acc': 0.3846, 'n': 13} | {'acc': 0.5263, 'n': 19} |
| empty_context | {'acc': None, 'n': 0} | {'acc': None, 'n': 0} |
| ICD same category / same letter (partial credit, separate) | 0.5288 / 0.7193 | n/a |
| ICD over / under coding rate | 0.005 / 0.0226 | n/a |
| ICD letters below overall | ['I', 'K', 'M', 'T'] | |
| SNOMED hierarchy distance | n/a | not computed: no SNOMED relationships supplied |

## final_test (main variant: stage2_icd_only_calibrated, primary seed)

| metric | ICD-10-CM | SNOMED CT |
|---|---|---|
| exact match [95% bootstrap CI] | 0.6006 [0.5523, 0.6544] | 0.5274 [0.4751, 0.5721] |
| headline (min) [CI] | 0.5274 [0.4776, 0.5751]; gap to 95%: 0.4226 | |
| acc, code train-freq 0 | {'acc': 0.3973, 'n': 73} | {'acc': 0.4225, 'n': 187} |
| acc, code train-freq 1 | {'acc': 0.6875, 'n': 48} | {'acc': 0.5926, 'n': 81} |
| acc, code train-freq 2-5 | {'acc': 0.6058, 'n': 104} | {'acc': 0.6321, 'n': 106} |
| acc, code train-freq 6+ | {'acc': 0.6797, 'n': 128} | {'acc': 0.6429, 'n': 28} |
| candidate pool recall | 0.8414 | 0.8458 |
| errors: retrieval / rank miss | {'rank_miss': 85, 'retrieval_miss': 56, 'total': 141} | {'rank_miss': 128, 'retrieval_miss': 62, 'total': 190} |
| first-stage recall@K | {'@1': 0.4051, '@10': 0.6856, '@100': 0.796, '@20': 0.728, '@200': 0.8414, '@5': 0.6346, '@50': 0.762} | {'@1': 0.3507, '@10': 0.6592, '@100': 0.8159, '@20': 0.704, '@200': 0.8458, '@5': 0.5896, '@50': 0.7637} |
| ECE (reported prob) | 0.0713 | 0.0508 |
| ECE before calibration | 0.1289 | 0.0992 |
| coverage vs acc | {'0.0': {'acc': 0.6006, 'coverage': 1.0}, '0.3': {'acc': 0.7052, 'coverage': 0.7592}, '0.5': {'acc': 0.8187, 'coverage': 0.5156}, '0.7': {'acc': 0.9048, 'coverage': 0.2975}, '0.9': {'acc': 0.9531, 'coverage': 0.1813}} | {'0.0': {'acc': 0.5274, 'coverage': 1.0}, '0.3': {'acc': 0.6606, 'coverage': 0.6891}, '0.5': {'acc': 0.8194, 'coverage': 0.3856}, '0.7': {'acc': 0.886, 'coverage': 0.2836}, '0.9': {'acc': 0.913, 'coverage': 0.1716}} |
| abbreviation_terms | {'acc': 0.5922, 'n': 206} | {'acc': 0.5111, 'n': 225} |
| noisy_oov_terms | {'acc': 0.6, 'n': 25} | {'acc': 0.4, 'n': 35} |
| negated_terms | {'acc': 0.55, 'n': 60} | {'acc': 0.4805, 'n': 77} |
| history_terms | {'acc': 0.6154, 'n': 13} | {'acc': 0.4167, 'n': 12} |
| empty_context | {'acc': 0.0, 'n': 1} | {'acc': None, 'n': 0} |
| ICD same category / same letter (partial credit, separate) | 0.7167 / 0.7734 | n/a |
| ICD over / under coding rate | 0.0 / 0.0 | n/a |
| ICD letters below overall | ['H', 'J', 'K', 'L', 'M', 'Z'] | |
| SNOMED hierarchy distance | n/a | not computed: no SNOMED relationships supplied |

