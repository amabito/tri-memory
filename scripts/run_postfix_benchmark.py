"""Post-fix benchmark: train toy TRN model, measure loss and throughput.

Compares training dynamics with the mathematical fixes applied:
- H2: complex modulus normalization
- M1: softplus+clamp amplitude
- C1: omega Nyquist clamp
- H5: gate_bias_init=0.65

Runs on CUDA (RTX 5090) with synthetic data for reproducibility.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trimemory.config import TRNConfig
from trimemory.model import TRNModel
from trimemory.benchmark import benchmark_forward, benchmark_step_single


def train_and_evaluate(
    cfg: TRNConfig,
    n_steps: int = 2000,
    batch_size: int = 16,
    seq_len: int = 256,
    lr: float = 3e-4,
    warmup_steps: int = 200,
    device: str = "cuda",
    seed: int = 42,
    label: str = "default",
) -> dict:
    """Train a model and return loss curve + metrics."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = TRNModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[{label}] params={n_params:,}, device={device}")
    print(f"  config: d_model={cfg.d_model}, K={cfg.n_oscillators}, "
          f"n_layers={cfg.n_layers}, gate_bias_init={cfg.gate_bias_init}")

    # Optimizer with weight decay separation
    param_groups = model.configure_optimizer_param_groups(weight_decay=0.1)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

    # Cosine LR schedule
    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return lr * step / warmup_steps
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return lr * 0.1 + 0.5 * (lr - lr * 0.1) * (1 + __import__("math").cos(progress * 3.14159))

    losses = []
    start_time = time.perf_counter()

    model.train()
    for step in range(n_steps):
        # Set LR
        current_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Synthetic data
        input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

        optimizer.zero_grad()
        out = model(input_ids, labels=input_ids)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if (step + 1) % 200 == 0 or step == 0:
            elapsed = time.perf_counter() - start_time
            tok_per_sec = (step + 1) * batch_size * seq_len / elapsed
            print(f"  step {step+1:5d}/{n_steps} | loss={loss_val:.4f} | "
                  f"lr={current_lr:.2e} | {tok_per_sec:.0f} tok/s")

    total_time = time.perf_counter() - start_time
    avg_tok_per_sec = n_steps * batch_size * seq_len / total_time

    # Final metrics
    final_loss = sum(losses[-50:]) / 50
    min_loss = min(losses)
    loss_at_500 = sum(losses[490:510]) / 20 if len(losses) > 510 else losses[-1]
    loss_at_1000 = sum(losses[990:1010]) / 20 if len(losses) > 1010 else losses[-1]

    result = {
        "label": label,
        "n_params": n_params,
        "n_steps": n_steps,
        "final_loss_avg50": round(final_loss, 4),
        "min_loss": round(min_loss, 4),
        "loss_at_500": round(loss_at_500, 4),
        "loss_at_1000": round(loss_at_1000, 4),
        "total_time_sec": round(total_time, 1),
        "avg_tok_per_sec": round(avg_tok_per_sec, 0),
        "losses_every_10": [round(losses[i], 4) for i in range(0, len(losses), 10)],
    }

    print(f"\n  [RESULT] final_loss={final_loss:.4f} | min_loss={min_loss:.4f} | "
          f"{avg_tok_per_sec:.0f} tok/s | {total_time:.1f}s")

    return result


def run_throughput_bench(cfg: TRNConfig, device: str = "cuda") -> dict:
    """Run throughput benchmarks."""
    model = TRNModel(cfg)

    fwd = benchmark_forward(model, batch_size=16, seq_len=256, n_steps=50, device=device)
    step = benchmark_step_single(model, batch_size=1, n_steps=200, device=device)

    return {
        "forward_tok_per_sec": round(fwd.tokens_per_second, 0),
        "forward_ms_per_step": round(fwd.ms_per_step, 2),
        "step_single_tok_per_sec": round(step.tokens_per_second, 0),
        "step_single_ms": round(step.ms_per_step, 2),
        "peak_memory_mb": round(fwd.peak_memory_mb, 1),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA: {torch.version.cuda}")

    results = {}

    # --- Toy config (fast, for validation) ---
    print("\n" + "=" * 60)
    print("=== Toy Config (d=128, K=64, L=2) ===")
    print("=" * 60)
    cfg_toy = TRNConfig.toy()
    results["toy_train"] = train_and_evaluate(
        cfg_toy, n_steps=2000, batch_size=32, seq_len=128,
        device=device, seed=42, label="toy",
    )
    results["toy_bench"] = run_throughput_bench(cfg_toy, device=device)

    # --- 100M config (realistic) ---
    print("\n" + "=" * 60)
    print("=== TRN-100M Config (d=512, K=256, L=8) ===")
    print("=" * 60)
    cfg_100m = TRNConfig.trn_100m()
    results["100m_train"] = train_and_evaluate(
        cfg_100m, n_steps=1000, batch_size=8, seq_len=512,
        device=device, seed=42, label="trn-100m",
    )
    results["100m_bench"] = run_throughput_bench(cfg_100m, device=device)

    # --- Multi-seed stability test (toy) ---
    print("\n" + "=" * 60)
    print("=== Stability Test: 3 seeds (toy config) ===")
    print("=" * 60)
    seed_losses = []
    for seed in [1, 2, 3]:
        r = train_and_evaluate(
            cfg_toy, n_steps=1000, batch_size=32, seq_len=128,
            device=device, seed=seed, label=f"toy_seed{seed}",
        )
        seed_losses.append(r["final_loss_avg50"])
    results["stability"] = {
        "seed_final_losses": seed_losses,
        "mean": round(sum(seed_losses) / len(seed_losses), 4),
        "std": round(
            (sum((x - sum(seed_losses)/3)**2 for x in seed_losses) / 3) ** 0.5, 4
        ),
    }

    # --- Summary ---
    print("\n" + "=" * 60)
    print("=== SUMMARY ===")
    print("=" * 60)

    print(f"\nToy (d=128, K=64, L=2, 2K steps):")
    t = results["toy_train"]
    print(f"  Final loss:  {t['final_loss_avg50']}")
    print(f"  Min loss:    {t['min_loss']}")
    print(f"  @500 steps:  {t['loss_at_500']}")
    print(f"  @1000 steps: {t['loss_at_1000']}")
    print(f"  Throughput:  {t['avg_tok_per_sec']:.0f} tok/s")
    b = results["toy_bench"]
    print(f"  Forward:     {b['forward_tok_per_sec']:.0f} tok/s, {b['forward_ms_per_step']:.2f} ms/step")
    print(f"  StepSingle:  {b['step_single_tok_per_sec']:.0f} tok/s, {b['step_single_ms']:.2f} ms/step")

    print(f"\nTRN-100M (d=512, K=256, L=8, 1K steps):")
    t = results["100m_train"]
    print(f"  Final loss:  {t['final_loss_avg50']}")
    print(f"  Min loss:    {t['min_loss']}")
    print(f"  @500 steps:  {t['loss_at_500']}")
    print(f"  Throughput:  {t['avg_tok_per_sec']:.0f} tok/s")
    b = results["100m_bench"]
    print(f"  Forward:     {b['forward_tok_per_sec']:.0f} tok/s, {b['forward_ms_per_step']:.2f} ms/step")
    print(f"  StepSingle:  {b['step_single_tok_per_sec']:.0f} tok/s, {b['step_single_ms']:.2f} ms/step")
    if b['peak_memory_mb'] > 0:
        print(f"  Peak VRAM:   {b['peak_memory_mb']:.0f} MB")

    print(f"\nStability (3 seeds, toy 1K steps):")
    s = results["stability"]
    print(f"  Losses: {s['seed_final_losses']}")
    print(f"  Mean:   {s['mean']}  Std: {s['std']}")

    # Save results
    out_path = Path(__file__).parent.parent / "data" / "postfix_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
