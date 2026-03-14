"""Fair comparison benchmark: TRNModel vs TransformerModel.

Compares on:
1. Loss after N training steps (same optimizer, LR, data)
2. Forward throughput (tokens/sec)
3. Generation throughput (step_single vs transformer autoregressive)
4. Parameter count
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from .baseline import TransformerModel
from .benchmark import benchmark_forward, benchmark_step_single
from .config import TRNConfig
from .model import TRNModel


@dataclass
class ComparisonResult:
    trn_loss: float
    transformer_loss: float
    trn_params: int
    transformer_params: int
    trn_forward_tps: float       # tokens/sec forward
    transformer_forward_tps: float
    trn_gen_tps: float           # tokens/sec autoregressive
    transformer_gen_tps: float
    n_steps: int


def _train_n_steps(
    model: nn.Module,
    n_steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    lr: float,
    device: str,
    seed: int = 42,
) -> float:
    """Train model for n_steps on fixed synthetic data. Returns final loss."""
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(
        model.configure_optimizer_param_groups(weight_decay=0.1), lr=lr
    )
    model.to(device).train()

    # Fixed data (same for both models — fair comparison)
    torch.manual_seed(seed)
    data = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
    input_ids = data[:, :-1]

    if n_steps == 0:
        # No training — return initial loss
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
            return out["loss"].item() if "loss" in out else float("inf")

    loss_val = float("inf")
    for _ in range(n_steps):
        optimizer.zero_grad()
        out = model(input_ids, labels=input_ids)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_val = loss.item()

    return loss_val


def _transformer_gen_tps(
    model: TransformerModel,
    batch_size: int,
    n_steps: int,
    device: str,
) -> float:
    """Measure Transformer autoregressive generation throughput (tokens/sec).

    Generates one token at a time using the full context (O(n^2) attention).
    """
    model.eval()
    vocab = model.cfg.vocab_size
    seq = torch.randint(0, vocab, (batch_size, 1), device=device)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_steps):
            out = model(seq)
            next_tok = out["logits"][:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=1)
    elapsed = time.perf_counter() - start
    return (batch_size * n_steps) / elapsed


def run_comparison(
    cfg: TRNConfig,
    n_train_steps: int = 50,
    n_bench_steps: int = 20,
    batch_size: int = 4,
    seq_len: int = 64,
    lr: float = 3e-4,
    device: str = "cpu",
) -> ComparisonResult:
    """Run full TRN vs Transformer comparison."""
    trn = TRNModel(cfg)
    transformer = TransformerModel(cfg)

    # Training comparison
    trn_loss = _train_n_steps(
        trn, n_train_steps, batch_size, seq_len, cfg.vocab_size, lr, device
    )
    # Reset transformer to same init seed for fair comparison
    transformer = TransformerModel(cfg)
    xfmr_loss = _train_n_steps(
        transformer, n_train_steps, batch_size, seq_len, cfg.vocab_size, lr, device
    )

    # Throughput benchmarks
    trn_fwd = benchmark_forward(
        TRNModel(cfg),
        batch_size=batch_size,
        seq_len=seq_len,
        n_steps=n_bench_steps,
        device=device,
    )
    xfmr_fwd = benchmark_forward(
        TransformerModel(cfg),
        batch_size=batch_size,
        seq_len=seq_len,
        n_steps=n_bench_steps,
        device=device,
    )

    trn_gen = benchmark_step_single(
        TRNModel(cfg), batch_size=1, n_steps=n_bench_steps, device=device
    )
    xfmr_gen_tps = _transformer_gen_tps(
        TransformerModel(cfg), batch_size=1, n_steps=n_bench_steps, device=device
    )

    return ComparisonResult(
        trn_loss=trn_loss,
        transformer_loss=xfmr_loss,
        trn_params=TRNModel(cfg).num_parameters(),
        transformer_params=TransformerModel(cfg).num_parameters(),
        trn_forward_tps=trn_fwd.tokens_per_second,
        transformer_forward_tps=xfmr_fwd.tokens_per_second,
        trn_gen_tps=trn_gen.tokens_per_second,
        transformer_gen_tps=xfmr_gen_tps,
        n_steps=n_train_steps,
    )


def print_comparison_report(r: ComparisonResult) -> None:
    """Print formatted comparison table."""
    print(f"\n{'='*60}")
    print(f"TRN vs Transformer Comparison ({r.n_steps} training steps)")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'TRN':>12} {'Transformer':>12}")
    print(f"{'-'*60}")
    print(
        f"{'Parameters (M)':<30} "
        f"{r.trn_params/1e6:>12.2f} {r.transformer_params/1e6:>12.2f}"
    )
    print(
        f"{'Final loss':<30} "
        f"{r.trn_loss:>12.4f} {r.transformer_loss:>12.4f}"
    )
    print(
        f"{'Forward tps':<30} "
        f"{r.trn_forward_tps:>12.1f} {r.transformer_forward_tps:>12.1f}"
    )
    print(
        f"{'Generation tps':<30} "
        f"{r.trn_gen_tps:>12.1f} {r.transformer_gen_tps:>12.1f}"
    )
    print(
        f"{'TRN gen advantage':<30} "
        f"{r.trn_gen_tps / max(r.transformer_gen_tps, 1):>11.1f}x"
    )
    print(f"{'='*60}\n")
