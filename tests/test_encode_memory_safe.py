"""Regression test: the batched tokenizer/encode.py must produce byte-identical
shards to a naive one-doc-at-a-time reference encoding."""
import json
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tokenizer"))
from encode import encode_file

TOKENIZER = Path(__file__).resolve().parent.parent / "tokenizer/artifacts/nova50_tokenizer.model"


def test_batched_encoding_matches_naive_reference(tmp_path):
    if not TOKENIZER.exists():
        return  # tokenizer not trained yet in this environment; skip gracefully
    sp = spm.SentencePieceProcessor(); sp.load(str(TOKENIZER))
    texts = ["Question: What is 2 plus 2? Answer: 4.", "Hello world.", "def f(x):\n    return x"]
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps({"text": t}) for t in texts) + "\n")
    out = tmp_path / "out.npy"
    encode_file(str(src), str(TOKENIZER), str(out), batch_docs=1)  # force multiple flush() calls

    ref = []
    for t in texts:
        ref.extend(sp.encode(t, out_type=int) + [sp.eos_id()])
    assert np.array_equal(np.load(out), np.array(ref, dtype=np.uint16))
