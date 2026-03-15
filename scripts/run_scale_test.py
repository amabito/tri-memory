"""Scale test: Phase 0 (smoke) + Phase 1 (scale-up) experiments.

Config 1A: d=512, K=256, L=8, d_ff=1024, hybrid [TRN,TRN,Attn,TRN,TRN,TRN,Attn,TRN]
Config 1B: same but pure TRN (no attn)
Config 1C: d=384, K=192, L=4, hybrid 1:1 ratio

Usage:
    python scripts/run_scale_test.py --config 1A --seed 42 [--max-epochs 30] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trimemory.block import CausalAttnBlock
from trimemory.config import TRNConfig
from trimemory.model import TRNModel


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def make_config(name: str, vocab: int = 50257) -> tuple[TRNConfig, list[int]]:
    """Return (TRNConfig, attn_positions) for named config."""
    if name == "1A":
        cfg = TRNConfig(
            vocab_size=vocab, d_model=512, n_oscillators=256,
            n_layers=8, d_ff=1024, max_seq_len=256,
            dropout=0.1, gate_bias_init=0.65, state_norm=True, phase_mode="log",
        )
        attn_pos = [2, 6]
    elif name == "1B":
        cfg = TRNConfig(
            vocab_size=vocab, d_model=512, n_oscillators=256,
            n_layers=8, d_ff=1024, max_seq_len=256,
            dropout=0.1, gate_bias_init=0.65, state_norm=True, phase_mode="log",
        )
        attn_pos = []  # pure TRN
    elif name == "1C":
        cfg = TRNConfig(
            vocab_size=vocab, d_model=384, n_oscillators=192,
            n_layers=4, d_ff=1536, max_seq_len=256,
            dropout=0.1, gate_bias_init=0.65, state_norm=True, phase_mode="log",
        )
        attn_pos = [1, 3]  # 1:1 ratio
    elif name == "2A":
        # Phase 2: same as 1A but dropout=0.2
        cfg = TRNConfig(
            vocab_size=vocab, d_model=512, n_oscillators=256,
            n_layers=8, d_ff=1024, max_seq_len=256,
            dropout=0.2, gate_bias_init=0.65, state_norm=True, phase_mode="log",
        )
        attn_pos = [2, 6]
    elif name == "2C":
        # Phase 2: same as 1A but dropout=0.3
        cfg = TRNConfig(
            vocab_size=vocab, d_model=512, n_oscillators=256,
            n_layers=8, d_ff=1024, max_seq_len=256,
            dropout=0.3, gate_bias_init=0.65, state_norm=True, phase_mode="log",
        )
        attn_pos = [2, 6]
    else:
        raise ValueError(f"Unknown config: {name}")
    return cfg, attn_pos


def build_model(cfg: TRNConfig, attn_pos: list[int], device: str) -> TRNModel:
    model = TRNModel(cfg).to(device)
    n_heads = max(4, cfg.d_model // 64)
    for pos in attn_pos:
        model.blocks[pos] = CausalAttnBlock(cfg, n_heads=n_heads).to(device)
    return model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare_data() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (train_ids, val_ids, vocab_size). No test -- reserved for final eval."""
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def enc(split: str) -> torch.Tensor:
        return torch.tensor(
            tokenizer.encode("\n".join(ds[split]["text"])), dtype=torch.long
        )

    return enc("train"), enc("validation"), tokenizer.vocab_size


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_val(model, data, seq_len, bs, device):
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


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def collect_diagnostics(model) -> dict:
    """Collect alpha mean, res_scale, gate weights from model."""
    diag = {"alpha_mean": [], "res_scale": [], "layer_type": []}
    for i, block in enumerate(model.blocks):
        if hasattr(block, "resonance"):
            # TRN block
            diag["layer_type"].append("TRN")
            # res_scale
            rs = block.resonance.res_scale.item()
            diag["res_scale"].append(round(rs, 6))
            # alpha: run a dummy forward to get alpha? Too expensive.
            # Instead read gate bias -> sigmoid
            bias = block.resonance.proj.proj.bias.data
            K = block.resonance.K
            gate_bias = bias[3 * K : 5 * K]
            alpha_mean = torch.sigmoid(gate_bias).mean().item()
            diag["alpha_mean"].append(round(alpha_mean, 4))
        else:
            diag["layer_type"].append("Attn")
            diag["res_scale"].append(None)
            diag["alpha_mean"].append(None)
    return diag


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, data, seq_len, bs, optimizer, device, grad_clip=1.0):
    model.train()
    n_tokens = len(data)
    total_loss, n_steps = 0.0, 0
    n_examples = (n_tokens - 1) // seq_len
    indices = torch.randperm(n_examples)

    for i in range(0, len(indices) - bs, bs):
        batch_idx = indices[i : i + bs]
        batch = []
        for idx in batch_idx:
            off = idx.item() * seq_len
            if off + seq_len + 1 > n_tokens:
                continue
            batch.append(data[off : off + seq_len].unsqueeze(0))
        if len(batch) < bs:
            continue

        ids = torch.cat(batch).to(device)
        optimizer.zero_grad()
        loss = model(ids, labels=ids)["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        optimizer.step()
        total_loss += loss.item()
        n_steps += 1

    return total_loss / max(n_steps, 1), grad_norm


def run_experiment(
    config_name: str,
    seed: int,
    max_epochs: int = 30,
    smoke: bool = False,
    device: str = "cuda",
    schedule: str = "cosine",  # "cosine" or "wsd"
) -> dict:
    print(f"\n{'='*60}")
    print(f"Config {config_name}, seed={seed}, max_epochs={max_epochs}, smoke={smoke}")
    print(f"{'='*60}")

    # Data
    train_data, val_data, vocab = prepare_data()

    # Model
    cfg, attn_pos = make_config(config_name, vocab)
    torch.manual_seed(seed)
    model = build_model(cfg, attn_pos, device)
    n_params = sum(p.numel() for p in model.parameters())

    layout = "".join("A" if i in attn_pos else "T" for i in range(cfg.n_layers))
    print(f"Params: {n_params:,}, d={cfg.d_model}, K={cfg.n_oscillators}, "
          f"L={cfg.n_layers}, layout=[{layout}]")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Optimizer
    param_groups = model.configure_optimizer_param_groups(weight_decay=0.1)
    optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95))

    seq_len, bs = 256, 16
    actual_epochs = 2 if smoke else max_epochs
    results = {
        "config": config_name,
        "seed": seed,
        "n_params": n_params,
        "layout": layout,
        "d_model": cfg.d_model,
        "n_oscillators": cfg.n_oscillators,
        "n_layers": cfg.n_layers,
        "epochs": [],
    }

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val PPL':>10} | "
          f"{'Grad Norm':>10} | {'Time':>6} | {'VRAM MB':>8}")
    print("-" * 70)

    best_val_ppl = float("inf")
    no_improve_count = 0
    t0 = time.perf_counter()

    for ep in range(actual_epochs):
        # LR schedule
        base_lr, min_lr = 3e-4, 3e-5
        warmup = 3
        if schedule == "wsd":
            # Warmup-Stable-Decay: 10% warmup, 75% stable, 15% decay
            warmup_end = int(actual_epochs * 0.10)
            decay_start = int(actual_epochs * 0.85)
            if ep < warmup_end:
                lr = base_lr * (ep + 1) / max(warmup_end, 1)
            elif ep < decay_start:
                lr = base_lr
            else:
                p = (ep - decay_start) / max(1, actual_epochs - decay_start)
                lr = min_lr + (base_lr - min_lr) * (1 - p)
        else:
            # Cosine with warmup
            if ep < warmup:
                lr = base_lr * (ep + 1) / warmup
            else:
                p = (ep - warmup) / max(1, actual_epochs - warmup)
                lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(p * math.pi))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        ep_start = time.perf_counter()
        train_loss, grad_norm = train_epoch(
            model, train_data, seq_len, bs, optimizer, device
        )
        val_ppl = evaluate_val(model, val_data, seq_len, bs, device)
        ep_time = time.perf_counter() - ep_start

        vram = 0.0
        if device == "cuda":
            vram = torch.cuda.max_memory_allocated() / 1e6

        # Early stop check
        if val_ppl < best_val_ppl:
            improvement = (best_val_ppl - val_ppl) / best_val_ppl
            if improvement < 0.01 and ep >= 20:
                no_improve_count += 1
            else:
                no_improve_count = 0
            best_val_ppl = val_ppl
            marker = " *"
        else:
            no_improve_count += 1
            marker = ""

        results["epochs"].append({
            "epoch": ep,
            "train_loss": round(train_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "grad_norm": round(grad_norm, 4),
            "lr": round(lr, 6),
            "time_sec": round(ep_time, 1),
            "vram_mb": round(vram, 0),
        })

        print(f"{ep:5d} | {train_loss:10.4f} | {val_ppl:10.2f} | "
              f"{grad_norm:10.4f} | {ep_time:5.1f}s | {vram:7.0f}{marker}")

        # Early stop: 3 consecutive epochs < 1% improvement after epoch 20
        if no_improve_count >= 3 and ep >= 20 and not smoke:
            print(f"Early stop at epoch {ep} (3 epochs < 1% improvement)")
            break

    total_time = time.perf_counter() - t0

    # Diagnostics
    diag = collect_diagnostics(model)

    results["final"] = {
        "best_val_ppl": round(best_val_ppl, 2),
        "total_time_sec": round(total_time, 1),
        "peak_vram_mb": round(vram, 0),
    }
    results["diagnostics"] = diag

    print(f"\nBest Val PPL: {best_val_ppl:.2f}")
    print(f"Total time: {total_time / 60:.1f} min")
    print(f"Peak VRAM: {vram:.0f} MB")
    print(f"Diagnostics: {json.dumps(diag, indent=2)}")

    # Smoke checks
    if smoke:
        checks = {
            "vram_ok": vram < 28000,
            "loss_decreasing": (
                len(results["epochs"]) >= 2
                and results["epochs"][1]["train_loss"] < results["epochs"][0]["train_loss"]
            ),
            "alpha_ok": all(
                0.3 <= a <= 0.9 for a in diag["alpha_mean"] if a is not None
            ),
            "res_scale_ok": all(
                abs(r) < 1.0 for r in diag["res_scale"] if r is not None
            ),
            "grad_norm_ok": all(
                e["grad_norm"] < 100.0 for e in results["epochs"]
            ),
        }
        results["smoke_checks"] = checks
        all_pass = all(checks.values())
        print(f"\nSmoke checks: {'ALL PASS' if all_pass else 'FAIL'}")
        for k, v in checks.items():
            status = "PASS" if v else "FAIL"
            print(f"  {k}: {status}")

    # Save
    tag = f"{'smoke' if smoke else 'full'}_{config_name}_seed{seed}"
    out_path = Path(__file__).parent.parent / "data" / f"scale_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="1A", choices=["1A", "1B", "1C", "2A", "2C"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"])
    args = parser.parse_args()

    run_experiment(
        config_name=args.config,
        seed=args.seed,
        max_epochs=args.max_epochs,
        smoke=args.smoke,
        device=args.device,
        schedule=args.schedule,
    )


if __name__ == "__main__":
    main()
