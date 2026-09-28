"""Tests for the streaming n-gram contamination/decontamination tool."""
import json
from evaluation.benchmark_contamination import build_eval_index, scan_corpus, norm_words


def test_norm_words_strips_punctuation_and_case():
    assert norm_words("Hello, World!!") == ["hello", "world"]


def test_planted_leak_is_detected_and_removed(tmp_path):
    leak = "the quick brown fox jumps over the lazy dog again and again all day long"
    clean = "completely different sentence about farming tomatoes in the countryside every single summer season"
    eval_dir = tmp_path / "eval"; eval_dir.mkdir()
    (eval_dir / "bench.jsonl").write_text(json.dumps({"id": 0, "texts": [leak]}) + "\n" +
                                           json.dumps({"id": 1, "texts": ["too short"]}) + "\n")
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"text": "prefix " + leak + " suffix"}) + "\n" +
                      json.dumps({"text": clean}) + "\n")

    index, meta, stats = build_eval_index(str(eval_dir), n=13)
    assert stats["bench"]["too_short"] == 1  # "too short" has < 13 words

    out = tmp_path / "clean.jsonl"
    hits, n_docs, n_flagged = scan_corpus(str(train), index, n=13, out_path=str(out))
    assert n_docs == 2 and n_flagged == 1
    kept = out.read_text().strip().splitlines()
    assert len(kept) == 1
    assert json.loads(kept[0])["text"] == clean
