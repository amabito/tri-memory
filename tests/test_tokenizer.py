"""Tests for CharTokenizer — happy path + adversarial."""
from __future__ import annotations

import pytest

from trimemory.tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# Happy path tests (8)
# ---------------------------------------------------------------------------


def test_fit_builds_vocab() -> None:
    tok = CharTokenizer()
    tok.fit("abc")
    # 4 special tokens + 3 unique chars = 7
    assert tok.vocab_size == 7


def test_encode_basic() -> None:
    tok = CharTokenizer()
    tok.fit("abc")
    ids = tok.encode("abc")
    assert ids == [4, 5, 6]


def test_encode_bos_eos() -> None:
    tok = CharTokenizer()
    tok.fit("a")
    ids = tok.encode("a", add_bos=True, add_eos=True)
    assert ids == [CharTokenizer.BOS_ID, 4, CharTokenizer.EOS_ID]
    assert ids == [2, 4, 3]


def test_decode_basic() -> None:
    tok = CharTokenizer()
    tok.fit("abc")
    text = tok.decode([4, 5, 6])
    assert text == "abc"


def test_decode_skip_special() -> None:
    tok = CharTokenizer()
    tok.fit("a")
    # [BOS, 'a', EOS] → skip BOS and EOS → "a"
    text = tok.decode([CharTokenizer.BOS_ID, 4, CharTokenizer.EOS_ID])
    assert text == "a"


def test_roundtrip() -> None:
    tok = CharTokenizer()
    original = "hello world"
    tok.fit(original)
    ids = tok.encode(original)
    decoded = tok.decode(ids)
    assert decoded == original


def test_save_load_roundtrip(tmp_path) -> None:
    tok = CharTokenizer()
    tok.fit("abc")
    path = tmp_path / "vocab.json"
    tok.save(path)

    tok2 = CharTokenizer.load(path)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.encode("abc") == [4, 5, 6]
    assert tok2.decode([4, 5, 6]) == "abc"


def test_encode_batch() -> None:
    tok = CharTokenizer()
    tok.fit("abc")
    result = tok.encode_batch(["ab", "c"])
    assert result == [[4, 5], [6]]


# ---------------------------------------------------------------------------
# Adversarial tests (10)
# ---------------------------------------------------------------------------


def test_encode_unknown_char() -> None:
    tok = CharTokenizer()
    tok.fit("abc")
    # 'z' not in vocab → UNK_ID
    ids = tok.encode("z")
    assert ids == [CharTokenizer.UNK_ID]
    assert ids == [1]


def test_encode_empty_string() -> None:
    tok = CharTokenizer()
    tok.fit("abc")
    assert tok.encode("") == []


def test_decode_empty_list() -> None:
    tok = CharTokenizer()
    tok.fit("abc")
    assert tok.decode([]) == ""


def test_fit_idempotent() -> None:
    tok = CharTokenizer()
    tok.fit("aaa")
    # "aaa" has only one unique char 'a'
    assert tok.vocab_size == 5  # 4 special + 1 char

    tok2 = CharTokenizer()
    tok2.fit("a")
    assert tok2.vocab_size == 5


def test_fit_merge() -> None:
    tok = CharTokenizer()
    tok.fit("ab")
    tok.fit("bc")
    # Chars: a, b (already in), c — b must not be duplicated
    assert tok.vocab_size == 4 + 3  # a, b, c
    ids = tok.encode("abc")
    assert len(ids) == 3
    assert len(set(ids)) == 3  # all distinct IDs


def test_save_load_unknown(tmp_path) -> None:
    tok = CharTokenizer()
    tok.fit("abc")
    path = tmp_path / "vocab.json"
    tok.save(path)

    tok2 = CharTokenizer.load(path)
    # 'z' not in saved vocab → UNK_ID
    assert tok2.encode("z") == [CharTokenizer.UNK_ID]


def test_load_nonexistent() -> None:
    with pytest.raises(FileNotFoundError):
        CharTokenizer.load("nonexistent_path_xyz.json")


def test_vocab_size_property() -> None:
    tok = CharTokenizer()
    tok.fit("hello")
    unique_chars = len(set("hello"))  # h, e, l, o = 4
    assert tok.vocab_size == 4 + unique_chars


def test_unicode() -> None:
    tok = CharTokenizer()
    text = "こんにちは"
    tok.fit(text)
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_pad_unk_bos_eos_ids() -> None:
    assert CharTokenizer.PAD_ID == 0
    assert CharTokenizer.UNK_ID == 1
    assert CharTokenizer.BOS_ID == 2
    assert CharTokenizer.EOS_ID == 3
