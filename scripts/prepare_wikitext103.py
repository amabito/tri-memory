"""Prepare WikiText-103 as packed binary for TRN training.

Downloads WikiText-103 via HuggingFace datasets, tokenizes with GPT-2 BPE
(same tokenizer as current WikiText-2 setup), and saves as packed uint16
binary files compatible with PackedDataset.

Token IDs fit in uint16 since GPT-2 vocab_size=50257 < 65535.
"""

from datasets import load_dataset
from transformers import GPT2TokenizerFast
import numpy as np
from pathlib import Path


def main() -> None:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    print("Downloading WikiText-103 (Salesforce/wikitext, wikitext-103-raw-v1)...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    out_dir = Path("D:/work/Projects/Tri-Memory/data/wikitext103")
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    for split_name, split_key in split_map.items():
        print(f"\nProcessing split: {split_name} ...")
        texts = ds[split_key]["text"]

        # Join all documents with newline separator -- same approach as wikitext-2 setup.
        # The tokenizer warning about sequence length > 1024 is expected and harmless;
        # we are tokenizing the full concatenated corpus, not individual sequences.
        text = "\n".join(texts)
        ids = tokenizer.encode(text)

        arr = np.array(ids, dtype=np.uint16)
        assert arr.max() < 65535, f"Token id {arr.max()} exceeds uint16 max"

        path = out_dir / f"{split_name}.bin"
        arr.tofile(str(path))

        size_mb = path.stat().st_size / 1e6
        print(f"  {split_name}: {len(ids):,} tokens, {size_mb:.1f} MB -> {path}")

    print("\nDone. Files written to:", out_dir)


if __name__ == "__main__":
    main()
