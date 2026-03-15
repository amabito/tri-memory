"""Long-context evaluation for Tri-Memory best model (Phase 2C: 44M hybrid).

Three evaluations:
    1. Needle in a Haystack -- retrieval accuracy at various positions
    2. Long-Range Dependency -- per-position PPL within a 256-token window
    3. Context Length Extrapolation -- val PPL at seq_len 64 / 128 / 256

Usage:
    python scripts/run_longctx_eval.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trimemory.block import CausalAttnBlock
from trimemory.config import TRNConfig
from trimemory.model import TRNModel

# ---------------------------------------------------------------------------
# Model + data helpers (reuse Phase 3 patterns)
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_OUT = Path(__file__).resolve().parent.parent / "data" / "longctx_eval.json"


def make_best_model(device: str = DEVICE) -> tuple[TRNModel, TRNConfig]:
    """Build Phase 2C best config (44M hybrid, dropout=0.3)."""
    cfg = TRNConfig(
        vocab_size=50257, d_model=512, n_oscillators=256,
        n_layers=8, d_ff=1024, max_seq_len=256,
        dropout=0.3, gate_bias_init=0.65, state_norm=True, phase_mode="log",
    )
    torch.manual_seed(42)
    model = TRNModel(cfg).to(device)
    model.blocks[2] = CausalAttnBlock(cfg, n_heads=8).to(device)
    model.blocks[6] = CausalAttnBlock(cfg, n_heads=8).to(device)
    return model, cfg


def load_wikitext2(split: str = "train") -> list[int]:
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    return tokenizer.encode("\n".join(ds[split]["text"]))


def train_quick(model: TRNModel, device: str = DEVICE, n_epochs: int = 22) -> TRNModel:
    """Re-train to reproduce Phase 2C model state (same loop as run_phase3_eval.py)."""
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_data = torch.tensor(
        tokenizer.encode("\n".join(ds["train"]["text"])), dtype=torch.long
    )

    param_groups = model.configure_optimizer_param_groups(weight_decay=0.1)
    optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95))
    seq_len, bs = 256, 16
    n_tokens = len(train_data)
    n_examples = (n_tokens - 1) // seq_len

    model.train()
    for ep in range(n_epochs):
        warmup = 3
        if ep < warmup:
            lr = 3e-4 * (ep + 1) / warmup
        else:
            p = (ep - warmup) / max(1, n_epochs - warmup)
            lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(p * math.pi))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        indices = torch.randperm(n_examples)
        for i in range(0, len(indices) - bs, bs):
            batch = []
            for idx in indices[i : i + bs]:
                off = idx.item() * seq_len
                if off + seq_len + 1 > n_tokens:
                    continue
                batch.append(train_data[off : off + seq_len].unsqueeze(0))
            if len(batch) < bs:
                continue
            ids = torch.cat(batch).to(device)
            optimizer.zero_grad()
            loss = model(ids, labels=ids)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if (ep + 1) % 5 == 0:
            print(f"  retrain epoch {ep+1}/{n_epochs}")

    return model


# ---------------------------------------------------------------------------
# Evaluation 1: Needle in a Haystack
# ---------------------------------------------------------------------------

def encode_phrase(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


@torch.inference_mode()
def run_needle_haystack(model: TRNModel, device: str = DEVICE) -> dict:
    """Insert 'The secret number is 42' at various positions; measure next-token accuracy."""
    from transformers import GPT2TokenizerFast
    print("\n" + "=" * 60)
    print("Evaluation 1: Needle in a Haystack")
    print("=" * 60)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    val_ids = load_wikitext2("validation")

    # Encode the needle and query
    needle_ids  = encode_phrase(tokenizer, " The secret number is 42")   # needle to insert
    query_ids   = encode_phrase(tokenizer, " The secret number is")       # query prefix
    target_id   = encode_phrase(tokenizer, " 42")[0]                      # expected next token

    needle_len = len(needle_ids)
    query_len  = len(query_ids)

    context_lengths = [64, 128, 256]
    # Fractional positions within the context where the needle is inserted
    insert_fracs = [0.10, 0.25, 0.50, 0.75, 0.90]

    results: dict[str, dict] = {}
    model.eval()

    for ctx_len in context_lengths:
        print(f"\n  Context length = {ctx_len}")
        row: dict[str, float] = {}
        for frac in insert_fracs:
            # Position where the needle starts (must leave room for needle + query)
            insert_pos = max(1, int(frac * (ctx_len - needle_len - query_len)))
            insert_pos = min(insert_pos, ctx_len - needle_len - query_len - 1)

            correct = 0
            n_trials = 100

            for trial in range(n_trials):
                # Sample a background context from val set
                start = (trial * ctx_len) % max(1, len(val_ids) - ctx_len - needle_len - query_len)
                bg = val_ids[start : start + ctx_len]

                # Build: bg[:insert_pos] + needle + bg[insert_pos:...] + query
                pre  = bg[:insert_pos]
                post_len = ctx_len - insert_pos - needle_len - query_len
                post = bg[insert_pos : insert_pos + max(0, post_len)]
                seq  = pre + needle_ids + post + query_ids

                # Trim or pad to ctx_len (should already be ctx_len)
                seq = seq[:ctx_len]
                if len(seq) < ctx_len:
                    seq = seq + [tokenizer.eos_token_id] * (ctx_len - len(seq))

                input_ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
                out = model(input_ids)
                pred = out["logits"][0, -1].argmax().item()
                if pred == target_id:
                    correct += 1

            acc = correct / n_trials
            label = f"{int(frac * 100)}%"
            row[label] = round(acc, 4)
            print(f"    insert @ {label:>4s}: acc = {acc:.4f}  ({correct}/{n_trials})")

        results[f"ctx{ctx_len}"] = row

    return results


# ---------------------------------------------------------------------------
# Evaluation 2: Long-Range Dependency (per-position PPL)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_position_ppl(model: TRNModel, device: str = DEVICE) -> dict:
    """Compute per-position NLL in 256-token windows; group into 5 bands."""
    print("\n" + "=" * 60)
    print("Evaluation 2: Long-Range Dependency (per-position PPL)")
    print("=" * 60)

    val_ids = torch.tensor(load_wikitext2("validation"), dtype=torch.long)
    seq_len = 256
    bs = 8

    # Accumulators per position
    pos_nll  = torch.zeros(seq_len, dtype=torch.float64)
    pos_cnt  = torch.zeros(seq_len, dtype=torch.long)

    model.eval()
    n_batches = 0

    for s in range(0, len(val_ids) - seq_len - 1, seq_len * bs):
        batch = []
        for b in range(bs):
            off = s + b * seq_len
            if off + seq_len + 1 > len(val_ids):
                break
            batch.append(val_ids[off : off + seq_len].unsqueeze(0))
        if not batch:
            break

        ids = torch.cat(batch).to(device)           # (B, 256)
        out = model(ids)
        logits = out["logits"]                        # (B, 256, V)

        # Per-position NLL: predict position t using t-1 context
        # shift: logits[:, :-1] predicts labels[:, 1:]
        shift_logits = logits[:, :-1]                # (B, 255, V)
        shift_labels = ids[:, 1:]                    # (B, 255)

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        # Gather NLL at each target position
        nll = -log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)  # (B, 255)

        # Accumulate by position index (position 0 predicts token at 1, etc.)
        for pos in range(nll.shape[1]):
            pos_nll[pos]  += nll[:, pos].sum().item()
            pos_cnt[pos]  += nll.shape[0]

        n_batches += 1

    print(f"  Processed {n_batches} batches")

    # Compute PPL per position band
    bands = [
        ("0-32",    range(0, 32)),
        ("32-64",   range(32, 64)),
        ("64-128",  range(64, 128)),
        ("128-192", range(128, 192)),
        ("192-255", range(192, 255)),
    ]

    band_results: dict[str, float] = {}
    print(f"\n  {'Band':<12} {'Avg NLL':>10} {'PPL':>10} {'Tokens':>10}")
    print("  " + "-" * 45)
    for band_name, pos_range in bands:
        indices = list(pos_range)
        total_nll = pos_nll[indices].sum().item()
        total_cnt = pos_cnt[indices].sum().item()
        avg_nll   = total_nll / max(total_cnt, 1)
        ppl       = math.exp(avg_nll)
        band_results[band_name] = round(ppl, 2)
        print(f"  {band_name:<12} {avg_nll:>10.4f} {ppl:>10.2f} {int(total_cnt):>10d}")

    return band_results


# ---------------------------------------------------------------------------
# Evaluation 3: Context Length Extrapolation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_ctx_extrapolation(model: TRNModel, device: str = DEVICE) -> dict:
    """Evaluate val PPL at seq_len = 64, 128, 256 (in-distribution: 256)."""
    print("\n" + "=" * 60)
    print("Evaluation 3: Context Length Extrapolation")
    print("=" * 60)

    val_ids = torch.tensor(load_wikitext2("validation"), dtype=torch.long)
    bs = 16
    model.eval()

    results: dict[str, float] = {}
    print(f"\n  {'seq_len':<12} {'PPL':>10} {'Batches':>10}")
    print("  " + "-" * 35)

    for seq_len in [64, 128, 256]:
        total_loss = 0.0
        n = 0
        for s in range(0, len(val_ids) - seq_len - 1, seq_len * bs):
            batch = []
            for b in range(bs):
                off = s + b * seq_len
                if off + seq_len + 1 > len(val_ids):
                    break
                batch.append(val_ids[off : off + seq_len].unsqueeze(0))
            if not batch:
                break
            ids = torch.cat(batch).to(device)
            loss = model(ids, labels=ids)["loss"].item()
            total_loss += loss
            n += 1

        ppl = math.exp(total_loss / max(n, 1))
        results[f"seq{seq_len}"] = round(ppl, 2)
        label = "in-dist" if seq_len == 256 else "shorter"
        print(f"  {seq_len:<12} {ppl:>10.2f} {n:>10}  ({label})")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Long-Context Evaluation -- Tri-Memory Phase 2C (44M hybrid)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Build model
    print("\nBuilding model...")
    model, cfg = make_best_model(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    # Retrain
    print("\nRetraining (22 epochs, same as Phase 3)...")
    model = train_quick(model, DEVICE, n_epochs=22)
    model.eval()

    # Run evaluations
    needle_results    = run_needle_haystack(model, DEVICE)
    position_results  = run_position_ppl(model, DEVICE)
    extrap_results    = run_ctx_extrapolation(model, DEVICE)

    # Compile summary
    all_results = {
        "model": "Phase2C_44M_hybrid",
        "config": {
            "d_model": cfg.d_model,
            "n_oscillators": cfg.n_oscillators,
            "n_layers": cfg.n_layers,
            "dropout": cfg.dropout,
            "hybrid_attn_layers": [2, 6],
            "max_seq_len": cfg.max_seq_len,
        },
        "needle_in_haystack": needle_results,
        "position_ppl": position_results,
        "ctx_extrapolation": extrap_results,
    }

    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_OUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {DATA_OUT}")

    # Final summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n--- Needle in a Haystack (accuracy) ---")
    print(f"{'Ctx':>6} | {'10%':>6} {'25%':>6} {'50%':>6} {'75%':>6} {'90%':>6}")
    print("-" * 48)
    for ctx_key, row in needle_results.items():
        vals = [f"{row.get(f'{p}%', 0.0):.3f}" for p in [10, 25, 50, 75, 90]]
        print(f"{ctx_key:>6} | " + " ".join(f"{v:>6}" for v in vals))

    print("\n--- Per-Position PPL (256-token window) ---")
    for band, ppl in position_results.items():
        print(f"  {band:<12}: {ppl:.2f}")

    print("\n--- Context Length Extrapolation ---")
    for key, ppl in extrap_results.items():
        note = " (training length)" if key == "seq256" else ""
        print(f"  {key}: {ppl:.2f}{note}")


if __name__ == "__main__":
    main()
