# External resources and licences

| resource | id / release | used for | status (iteration 2 build) | licence |
|---|---|---|---|---|
| ICD-10-CM dictionary | `resources/icd10cm_codes.xlsx` (supplied; sha256 in `configs/resources.yaml`) | candidate space, code-string hierarchy | used | ICD-10-CM published by CDC/NCHS (public domain); file supplied by organisers |
| SNOMED CT US dictionary | `resources/SNOMEDCT_US_codes_and_descriptions.xlsx`, Term Map export from index `gte_base_fp32`, exported 24 Sep 2026 | candidate space, semantic tags, abbreviations | used | SNOMED CT: UMLS / SNOMED International affiliate licence |
| Training file | `data/hackathon_train.csv` (proprietary) | supervision only; never published or uploaded | used | challenge terms |
| spaCy `en_core_web_lg` 3.8.0 | official GitHub release wheel, sha256 `293e9547a655b25499198ab15a525b05b9407a75f10255e405e8c3854329ab63` | static 300-d word vectors (dense channel `dense_*`) | used | **MIT** (package metadata and `meta.json`); vectors source "Explosion Vectors" listed as **CC0** in `meta.json` |
| scikit-learn 1.9.1, LightGBM 4.7.0, RapidFuzz 3.14.6, numpy, pandas, scipy | PyPI | BM25/TF-IDF, reranker, fuzzy matching, isotonic calibration | used | BSD-3 / MIT / MIT / BSD |
| PyTorch 2.7.1, transformers 4.57.6, sentence-transformers 5.1.1, faiss-cpu 1.15.1, datasets, accelerate | PyPI | dense / fine-tune / cross-encoder code paths | installed; exercised only with tiny random local models | BSD-3 / Apache-2.0 / Apache-2.0 / MIT / Apache-2.0 / Apache-2.0 |
| thenlper/gte-base | Hugging Face | mandatory dense baseline; top-1 diagnostic | **NOT RUN** (HF blocked: HTTP 403) | recorded from the model card at download |
| cambridgeltl/SapBERT-from-PubMedBERT-fulltext | Hugging Face | dense candidate (CLS pooling) | **NOT RUN** | recorded at download |
| FremyCompany/BioLORD-2023 | Hugging Face | dense candidate | **NOT RUN** | recorded at download (UMLS-derived training data: check terms) |
| ncbi/MedCPT-Query-Encoder + Article-Encoder, ncbi/MedCPT-Cross-Encoder | Hugging Face | asymmetric dense candidate; cross-encoder start point | **NOT RUN** | recorded at download |
| BAAI/bge-m3, BAAI/bge-reranker-base, intfloat/e5-base-v2 | Hugging Face | dense / reranker candidates | **NOT RUN** | recorded at download |
| SNOMED CT -> ICD-10-CM map (NLM) | not supplied | auxiliary compatibility feature only | not used | UMLS licence |
| SNOMED CT US RF2 relationships | not supplied | IS-A features / hierarchy distance | not used; hierarchy distance not computed | SNOMED licence |

Model revisions: `scripts/setup_resources.py --download-dense` resolves each model's current commit
(or `--revision <sha>`), downloads exactly that commit and writes `revision` and
`licence_from_model_card` into `configs/resources.yaml`. No revision is claimed here because no
download was possible in this build.

## GPU runbook (Kaggle)

See `scripts/kaggle_run.sh` (every step resumable). In short:

```bash
python scripts/setup_resources.py --download-dense --only thenlper/gte-base,cambridgeltl/SapBERT-from-PubMedBERT-fulltext,FremyCompany/BioLORD-2023,BAAI/bge-m3,ncbi/MedCPT-Cross-Encoder
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/dense_benchmark.py --model gte=resources/models/gte-base --model sapbert=resources/models/sapbert \
       --model biolord=resources/models/biolord-2023 --model bgem3=resources/models/bge-m3
python scripts/train.py --config configs/kaggle_dense.yaml --tag kaggle_dense --no-final
python scripts/finetune_encoder.py --base resources/models/sapbert --out models/encoders/sapbert-ft --experiment code_disjoint
for EXP in val code_disjoint final_test final; do
  python scripts/train_cross_encoder.py --config configs/kaggle_dense.yaml --experiment $EXP --base-model resources/models/medcpt-cross
done
# set cross_encoder.enabled: true in configs/kaggle_dense.yaml, then
python scripts/train.py --config configs/kaggle_dense.yaml --tag kaggle_ce --final-test
```

Cost notes: ~780k dictionary strings (+ ~70k ICD path texts) per encoder; fp16 on GPU, cached in a
string-level store (`resources/cache/emb_<name>_<hash>/`, ~1.3 GB per 768-d model), so a new
dictionary release only embeds new/changed strings. Inference encodes only the query views.
