#!/usr/bin/env python
"""TRN end-to-end demo: train on a small text corpus, then generate.

Usage:
    uv run python demo.py
    uv run python demo.py --compare   # also compare with Transformer baseline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch

from trimemory.config import TRNConfig
from trimemory.generate import GenerationConfig, generate
from trimemory.model import TRNModel
from trimemory.tokenizer import CharTokenizer


CORPUS = """\
The quick brown fox jumps over the lazy dog.
A journey of a thousand miles begins with a single step.
To be or not to be, that is the question.
All that glitters is not gold.
The only way to do great work is to love what you do.
In the middle of difficulty lies opportunity.
Life is what happens when you are busy making other plans.
""" * 10  # repeat for more training data


def main() -> None:
    parser = argparse.ArgumentParser(description="TRN demo")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--compare", action="store_true", help="Compare with Transformer"
    )
    parser.add_argument("--prompt", type=str, default="The quick")
    args = parser.parse_args()

    # Tokenize corpus
    print("Building tokenizer...")
    tok = CharTokenizer()
    tok.fit(CORPUS)
    print(f"Vocab size: {tok.vocab_size} characters")

    # Build config matching tokenizer vocab
    cfg = TRNConfig(
        vocab_size=tok.vocab_size,
        d_model=128,
        n_layers=2,
        n_oscillators=64,
        d_ff=512,
        max_seq_len=128,
        dropout=0.0,
        tie_weights=True,
        use_parallel_scan=False,
    )

    # Train TRN
    print(f"\nTraining TRN for {args.steps} steps...")

    # Encode corpus to token ids
    ids = tok.encode(CORPUS)

    def make_batch(batch_size: int = 8, seq_len: int = 64) -> torch.Tensor:
        starts = np.random.randint(
            0, max(1, len(ids) - seq_len - 1), size=batch_size
        )
        return torch.tensor(
            [ids[s : s + seq_len + 1] for s in starts], dtype=torch.long
        )

    model = TRNModel(cfg)
    optimizer = torch.optim.AdamW(
        model.configure_optimizer_param_groups(weight_decay=0.01), lr=3e-3
    )

    losses = []
    for step in range(args.steps):
        batch = make_batch()
        input_ids = batch[:, :-1]
        optimizer.zero_grad()
        out = model(input_ids, labels=input_ids)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            print(f"  step {step+1:4d}/{args.steps} | loss {loss.item():.4f}")

    print(f"\nTraining complete. Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Generate
    print(f"\nGenerating from prompt: '{args.prompt}'")
    prompt_ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    gen_cfg = GenerationConfig(
        max_new_tokens=100, temperature=0.8, top_p=0.9, do_sample=True
    )
    model.eval()
    with torch.no_grad():
        generated = generate(model, prompt_ids, gen_cfg)

    text = args.prompt + tok.decode(generated[0].tolist())
    print(f"\n{'-'*50}")
    print(text)
    print(f"{'-'*50}")

    # Optional comparison
    if args.compare:
        print("\nRunning TRN vs Transformer comparison...")
        from trimemory.compare import print_comparison_report, run_comparison

        result = run_comparison(
            cfg, n_train_steps=30, n_bench_steps=10, device="cpu"
        )
        print_comparison_report(result)


if __name__ == "__main__":
    main()
