"""Character-level tokenizer for TRN language models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union


class CharTokenizer:
    """Character-level tokenizer with special tokens.

    Special token IDs (always reserved):
        PAD = 0
        UNK = 1
        BOS = 2
        EOS = 3
    Vocabulary: special tokens (0-3) + sorted unique chars (4+)
    """

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        # 4 special tokens always present
        self._vocab_size: int = 4

    def fit(self, text: str) -> "CharTokenizer":
        """Build vocabulary from text. Can be called multiple times (merges)."""
        chars = sorted(set(text) - set(self._char_to_id.keys()))
        for ch in chars:
            idx = self._vocab_size
            self._char_to_id[ch] = idx
            self._id_to_char[idx] = ch
            self._vocab_size += 1
        return self

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        """Encode text to token IDs. Unknown chars map to UNK_ID."""
        ids = [self._char_to_id.get(ch, self.UNK_ID) for ch in text]
        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        special = {self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID}
        chars = []
        for i in ids:
            if skip_special and i in special:
                continue
            chars.append(self._id_to_char.get(i, ""))
        return "".join(chars)

    def encode_batch(
        self,
        texts: list[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]:
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "char_to_id": self._char_to_id,
            "vocab_size": self._vocab_size,
        }
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CharTokenizer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        tok = cls()
        tok._char_to_id = data["char_to_id"]
        tok._id_to_char = {int(v): k for k, v in data["char_to_id"].items()}
        tok._vocab_size = data["vocab_size"]
        return tok
