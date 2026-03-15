"""Phase 3: Memory capability evaluation + ablation.

3D: TRN zero-out (res_scale=0) -- measure TRN contribution
3E: Attn zero-out -- measure Attn contribution
3A: Associative recall (synthetic)
3B: Copy task (synthetic)
3C: Induction head (synthetic)

Usage:
    python scripts/run_phase3_eval.py --task ablation  # 3D+3E
    python scripts/run_phase3_eval.py --task recall     # 3A
    python scripts/run_phase3_eval.py --task copy       # 3B
    python scripts/run_phase3_eval.py --task induction  # 3C
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trimemory.block import CausalAttnBlock
from trimemory.config import TRNConfig
from trimemory.model import TRNModel


def make_best_model(device: str = "cuda") -> tuple[TRNModel, TRNConfig]:
    """Build Phase 2 best config (2C: dropout=0.3, 44M hybrid)."""
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


def prepare_wikitext2_val() -> torch.Tensor:
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    ids = tokenizer.encode("\n".join(ds["validation"]["text"]))
    return torch.tensor(ids, dtype=torch.long)


@torch.inference_mode()
def evaluate_ppl(model, data, seq_len=256, bs=16, device="cuda"):
    model.eval()
    total, n = 0.0, 0
    for s in range(0, len(data) - seq_len - 1, seq_len * bs):
        batch = []
        for b in range(bs):
            off = s + b * seq_len
            if off + seq_len + 1 > len(data):
                break
            batch.append(data[off : off + seq_len].unsqueeze(0))
        if not batch:
            break
        ids = torch.cat(batch).to(device)
        total += model(ids, labels=ids)["loss"].item()
        n += 1
    return math.exp(total / max(n, 1))


def train_quick(model, cfg, device, n_epochs=22):
    """Quick re-train to reproduce Phase 2C best model state."""
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
# 3D/3E: Ablation
# ---------------------------------------------------------------------------

def run_ablation(device: str = "cuda"):
    print("=" * 60)
    print("Phase 3D/3E: Ablation (TRN zero-out, Attn zero-out)")
    print("=" * 60)

    model, cfg = make_best_model(device)
    val_data = prepare_wikitext2_val()

    print("\nRetraining model to reproduce Phase 2C state...")
    model = train_quick(model, cfg, device, n_epochs=22)

    # Baseline PPL
    base_ppl = evaluate_ppl(model, val_data, device=device)
    print(f"\nBaseline Val PPL: {base_ppl:.2f}")

    # 3D: TRN zero-out (set res_scale=0 for all TRN blocks)
    saved_scales = {}
    for i, block in enumerate(model.blocks):
        if hasattr(block, "resonance"):
            saved_scales[i] = block.resonance.res_scale.data.clone()
            block.resonance.res_scale.data.fill_(0.0)

    trn_zero_ppl = evaluate_ppl(model, val_data, device=device)
    print(f"3D TRN zero-out PPL: {trn_zero_ppl:.2f} (delta: +{trn_zero_ppl - base_ppl:.2f})")

    # Restore
    for i, scale in saved_scales.items():
        model.blocks[i].resonance.res_scale.data.copy_(scale)

    # 3E: Attn zero-out (zero QKV weights)
    saved_attn = {}
    for i, block in enumerate(model.blocks):
        if hasattr(block, "qkv"):
            saved_attn[i] = {
                "qkv": block.qkv.weight.data.clone(),
                "proj": block.proj.weight.data.clone(),
            }
            block.qkv.weight.data.fill_(0.0)
            block.proj.weight.data.fill_(0.0)

    attn_zero_ppl = evaluate_ppl(model, val_data, device=device)
    print(f"3E Attn zero-out PPL: {attn_zero_ppl:.2f} (delta: +{attn_zero_ppl - base_ppl:.2f})")

    # Restore
    for i, weights in saved_attn.items():
        model.blocks[i].qkv.weight.data.copy_(weights["qkv"])
        model.blocks[i].proj.weight.data.copy_(weights["proj"])

    results = {
        "baseline_ppl": round(base_ppl, 2),
        "trn_zero_ppl": round(trn_zero_ppl, 2),
        "trn_contribution": round(trn_zero_ppl - base_ppl, 2),
        "attn_zero_ppl": round(attn_zero_ppl, 2),
        "attn_contribution": round(attn_zero_ppl - base_ppl, 2),
    }

    print(f"\n{'Component':<20} {'PPL':>10} {'Delta':>10} {'% of total':>12}")
    print("-" * 55)
    total_delta = (trn_zero_ppl - base_ppl) + (attn_zero_ppl - base_ppl)
    for name, zppl in [("TRN (6 layers)", trn_zero_ppl), ("Attn (2 layers)", attn_zero_ppl)]:
        delta = zppl - base_ppl
        pct = delta / total_delta * 100 if total_delta > 0 else 0
        print(f"{name:<20} {zppl:>10.2f} {delta:>+10.2f} {pct:>11.1f}%")

    out = Path(__file__).parent.parent / "data" / "phase3_ablation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
    return results


# ---------------------------------------------------------------------------
# 3A: Associative Recall
# ---------------------------------------------------------------------------

def run_associative_recall(device: str = "cuda"):
    print("=" * 60)
    print("Phase 3A: Associative Recall (synthetic)")
    print("=" * 60)

    model, cfg = make_best_model(device)

    print("\nRetraining model...")
    model = train_quick(model, cfg, device, n_epochs=22)
    model.eval()

    results = {}
    for vocab_size in [64, 256]:
        for n_pairs in [8, 16, 32]:
            seq_len = n_pairs * 2 + 2  # KV pairs + separator + query
            if seq_len > cfg.max_seq_len:
                continue

            correct, total = 0, 0
            n_trials = 200

            for _ in range(n_trials):
                # Generate KV pairs: K1 V1 K2 V2 ... Kn Vn SEP Kq
                keys = torch.randint(0, vocab_size, (n_pairs,))
                vals = torch.randint(0, vocab_size, (n_pairs,))
                # Ensure unique keys
                keys = torch.arange(n_pairs) % vocab_size

                seq = []
                for k, v in zip(keys, vals):
                    seq.extend([k.item(), v.item()])
                query_idx = torch.randint(0, n_pairs, (1,)).item()
                seq.append(vocab_size)  # separator token
                seq.append(keys[query_idx].item())

                input_ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
                with torch.inference_mode():
                    out = model(input_ids)
                    logits = out["logits"][0, -1, :vocab_size]
                    pred = logits.argmax().item()
                    target = vals[query_idx].item()
                    if pred == target:
                        correct += 1
                    total += 1

            acc = correct / total
            key = f"v{vocab_size}_p{n_pairs}"
            results[key] = round(acc, 4)
            print(f"  vocab={vocab_size:3d}, pairs={n_pairs:2d}: acc={acc:.4f} ({correct}/{total})")

    out = Path(__file__).parent.parent / "data" / "phase3_recall.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ablation",
                        choices=["ablation", "recall", "all"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.task in ("ablation", "all"):
        run_ablation(args.device)
    if args.task in ("recall", "all"):
        run_associative_recall(args.device)


if __name__ == "__main__":
    main()
