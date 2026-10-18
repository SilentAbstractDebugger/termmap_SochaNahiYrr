"""Create TINY randomly initialised BERT models locally (no download) to exercise the dense,
fine-tuning and cross-encoder code paths in environments without model access. They carry NO
semantic knowledge; numbers obtained with them are meaningless and never reported as results.

  python scripts/make_tiny_test_models.py   ->  models/test/tiny-bi, models/test/tiny-ce
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(out=ROOT / "models" / "test"):
    import torch
    from sentence_transformers import SentenceTransformer, models
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizerFast

    torch.manual_seed(0)
    out = Path(out)
    tokdir = out / "tok"
    tokdir.mkdir(parents=True, exist_ok=True)
    corpus = ["chronic kidney disease stage 3", "essential hypertension", "mitral valve stenosis",
              "multiple sclerosis", "rheumatoid arthritis", "pelvic inflammatory disease", "chest pain",
              "type 2 diabetes mellitus with diabetic neuropathy", "right atrial abnormality", "edema"] * 20
    (tokdir / "corpus.txt").write_text("\n".join(corpus))
    t = BertWordPieceTokenizer(lowercase=True)
    t.train([str(tokdir / "corpus.txt")], vocab_size=500, min_frequency=1)
    t.save_model(str(tokdir))
    tok = BertTokenizerFast(vocab_file=str(tokdir / "vocab.txt"), do_lower_case=True)
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=128)
    bi = out / "tiny-bi-base"
    BertModel(cfg).save_pretrained(bi)
    tok.save_pretrained(bi)
    word = models.Transformer(str(bi), max_seq_length=64)
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
    SentenceTransformer(modules=[word, pool], device="cpu").save(str(out / "tiny-bi"))
    ce = out / "tiny-ce"
    BertForSequenceClassification(BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, max_position_embeddings=128, num_labels=1)).save_pretrained(ce)
    tok.save_pretrained(ce)
    print("tiny models in", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "test")
