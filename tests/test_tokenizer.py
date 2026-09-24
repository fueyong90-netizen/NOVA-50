"""Tests du tokenizer entraîné (si présent) : encode/decode roundtrip."""
from pathlib import Path

import pytest
import sentencepiece as spm

TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "tokenizer/artifacts/nova50_tokenizer.model"


@pytest.mark.skipif(not TOKENIZER_PATH.exists(), reason="Tokenizer pas encore entraîné (voir tokenizer/train_tokenizer.py)")
def test_encode_decode_roundtrip():
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    text = "Question: What is 4 plus 5? Answer: 9."
    ids = sp.encode(text, out_type=int)
    decoded = sp.decode(ids)
    # SentencePiece peut légèrement changer l'espacement -> on compare sans espaces
    assert decoded.replace(" ", "") == text.replace(" ", "")


@pytest.mark.skipif(not TOKENIZER_PATH.exists(), reason="Tokenizer pas encore entraîné")
def test_special_tokens_present():
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    assert sp.pad_id() == 0
    assert sp.unk_id() == 1
    assert sp.bos_id() == 2
    assert sp.eos_id() == 3
