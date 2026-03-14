#!/usr/bin/env python3
"""Tri-Memory Performance Profiler & Optimizer.

Phase 1: Baseline profiling
Phase 2: Batch size sweep
Phase 3: Mixed precision comparison
Phase 4: Python overhead reduction
Phase 5: Retrieval batching
Phase 6: Compile / kernel optimization

Usage:
    # Phase 1: baseline only
    python scripts/profile_trimemory_perf.py --phase 1 --device cuda --output-dir artifacts/perf/

    # All phases
    python scripts/profile_trimemory_perf.py --phase all --device cuda --output-dir artifacts/perf/

    # Specific phases
    python scripts/profile_trimemory_perf.py --phase 1,2,3 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

_src = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, _src)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trimemory.bench_data import seed_everything
from trimemory.config import TRNConfig
from trimemory.integrations.vllm_backend import DualMemoryEngine
from trimemory.tri_memory import TriMemoryEngine

# ---------------------------------------------------------------------------
# Constants (match streaming eval exactly)
# ---------------------------------------------------------------------------
VOCAB_SIZE = 256
D_MODEL = 128
N_LAYERS = 4
N_OSC = 64
D_FF = 512
DEFAULT_WINDOW = 64
DEFAULT_CHUNK = 32
DEFAULT_MAX_ARCHIVE = 128
SEED = 0

# Accuracy tolerance for optimization adoption
ACCURACY_TOLERANCE = 0.005

# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

class Timer:
    """Accumulating timer for named sections."""

    def __init__(self):
        self._timers: dict[str, list[float]] = {}
        self._starts: dict[str, float] = {}

    def start(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._starts.pop(name)
        self._timers.setdefault(name, []).append(elapsed)
        return elapsed

    @contextmanager
    def section(self, name: str):
        self.start(name)
        yield
        self.stop(name)

    def mean_ms(self, name: str) -> float:
        vals = self._timers.get(name, [0])
        return float(np.mean(vals)) * 1000.0

    def std_ms(self, name: str) -> float:
        vals = self._timers.get(name, [0])
        return float(np.std(vals)) * 1000.0

    def total_ms(self, name: str) -> float:
        vals = self._timers.get(name, [0])
        return float(np.sum(vals)) * 1000.0

    def summary(self) -> dict[str, dict]:
        return {
            name: {
                "mean_ms": self.mean_ms(name),
                "std_ms": self.std_ms(name),
                "total_ms": self.total_ms(name),
                "count": len(vals),
            }
            for name, vals in self._timers.items()
        }


def get_gpu_stats() -> dict:
    """Get current GPU memory and utilization."""
    if not torch.cuda.is_available():
        return {"gpu_mem_peak_mb": 0, "gpu_mem_allocated_mb": 0}
    return {
        "gpu_mem_peak_mb": torch.cuda.max_memory_allocated() / 1e6,
        "gpu_mem_allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "gpu_mem_reserved_mb": torch.cuda.memory_reserved() / 1e6,
    }


def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


# ---------------------------------------------------------------------------
# Data generation (deterministic, same as streaming eval)
# ---------------------------------------------------------------------------

FILLER_RANGE = (130, 200)
FACT_RANGE = (220, 245)
PATTERN_RANGE = (10, 60)
SALIENT_RANGE = (245, 256)
NEUTRAL_RANGE = (100, 130)
RECENT_RANGE = (200, 220)


def generate_training_data(
    n_samples: int, seq_len: int, seed: int = 0,
) -> Tensor:
    """Generate deterministic training data matching episode token distribution."""
    rng = np.random.default_rng(seed)
    data = np.zeros((n_samples, seq_len), dtype=np.int64)
    for i in range(n_samples):
        for j in range(seq_len):
            data[i, j] = rng.integers(0, VOCAB_SIZE)
    return torch.tensor(data, dtype=torch.long)


def generate_streaming_episode(rng, kv_window=64, chunk_size=32) -> list[int]:
    """Generate one streaming episode (480 tokens)."""
    tokens = []
    fact_positions = [5, 15, 30, 50]
    for fp in fact_positions:
        while len(tokens) < fp:
            tokens.append(int(rng.integers(*FILLER_RANGE)))
        tokens.append(int(rng.integers(*FACT_RANGE)))
    while len(tokens) < 64:
        tokens.append(int(rng.integers(*FILLER_RANGE)))
    period = int(rng.integers(3, 8))
    pattern_val = int(rng.integers(*PATTERN_RANGE))
    for i in range(128):
        tokens.append(pattern_val if i % period == 0 else int(rng.integers(*FILLER_RANGE)))
    for i in range(128):
        if i in (10, 40, 70, 100):
            tokens.append(int(rng.integers(*SALIENT_RANGE)))
        elif i in (20, 55, 85, 115):
            tokens.append(int(rng.integers(*NEUTRAL_RANGE)))
        else:
            tokens.append(int(rng.integers(*FILLER_RANGE)))
    for _ in range(64):
        tokens.append(int(rng.integers(*FILLER_RANGE)))
    recent_val = int(rng.integers(*RECENT_RANGE))
    for i in range(64):
        tokens.append(recent_val if i % 4 == 0 else int(rng.integers(*FILLER_RANGE)))
    while len(tokens) < 480:
        tokens.append(int(rng.integers(*FILLER_RANGE)))
    return tokens


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def make_cfg(max_seq_len: int = 512) -> TRNConfig:
    return TRNConfig(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_oscillators=N_OSC,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=max_seq_len,
    )


def build_trimemory(cfg, window=DEFAULT_WINDOW, chunk=DEFAULT_CHUNK,
                     max_archive=DEFAULT_MAX_ARCHIVE) -> TriMemoryEngine:
    return TriMemoryEngine(
        cfg, window_size=window, chunk_size=chunk,
        max_retrieval_chunks=max_archive,
        enable_trn=True, enable_retrieval=True,
        saliency_threshold=0.15,
    )


# ---------------------------------------------------------------------------
# Phase 1: Baseline profiling
# ---------------------------------------------------------------------------

@dataclass
class BaselineProfile:
    step_time_ms: float = 0.0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    optimizer_time_ms: float = 0.0
    dataloader_wait_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    archive_update_time_ms: float = 0.0
    router_time_ms: float = 0.0
    logging_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    gpu_mem_peak_mb: float = 0.0
    cpu_util_percent: float = 0.0
    gpu_util_percent: float = 0.0
    window_mask_time_ms: float = 0.0
    trn_scan_time_ms: float = 0.0
    streaming_step_time_ms: float = 0.0
    streaming_tokens_per_second: float = 0.0
    streaming_eviction_time_ms: float = 0.0
    streaming_retrieval_time_ms: float = 0.0
    streaming_trn_state_time_ms: float = 0.0


def profile_baseline(
    device: torch.device,
    steps: int = 100,
    batch_size: int = 16,
    seq_len: int = 480,
    warmup: int = 10,
) -> BaselineProfile:
    """Phase 1: Profile all components of the training + streaming pipeline."""
    print("\n=== Phase 1: Baseline Profiling ===")

    seed_everything(SEED)
    cfg = make_cfg()
    model = build_trimemory(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    data = generate_training_data(max(batch_size * 4, 64), seq_len, seed=SEED).to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
    )

    timer = Timer()
    reset_gpu_stats()

    # --- Training profiling ---
    model.train()
    loader_it = iter(loader)
    total_tokens = 0

    for step in range(1, steps + warmup + 1):
        try:
            (batch_ids,) = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            (batch_ids,) = next(loader_it)

        batch_ids = batch_ids.to(device)
        is_measured = step > warmup

        if is_measured:
            timer.start("step")

        if is_measured:
            timer.start("forward")
        out = model(batch_ids, labels=batch_ids)
        loss = out["loss"]
        if is_measured:
            timer.stop("forward")

        if torch.isfinite(loss):
            if is_measured:
                timer.start("backward")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if is_measured:
                timer.stop("backward")

            if is_measured:
                timer.start("optimizer")
            optimizer.step()
            if is_measured:
                timer.stop("optimizer")

        if is_measured:
            timer.stop("step")
            total_tokens += batch_ids.numel()

    train_gpu = get_gpu_stats()

    # --- Profile window mask creation ---
    model.eval()
    with torch.inference_mode():
        for _ in range(50):
            with timer.section("window_mask"):
                model.blocks[0]._make_window_mask(seq_len, device)

    # --- Profile TRN scan ---
    with torch.inference_mode():
        dummy_input = torch.randint(0, VOCAB_SIZE, (1, seq_len), device=device)
        x = model.drop_emb(model.embedding(dummy_input))
        pe_len = min(seq_len, model.pe.size(0))
        x[:, :pe_len] += model.pe[:pe_len]
        h = model.blocks[0].norm1(x)
        for _ in range(50):
            with timer.section("trn_scan"):
                model.blocks[0].trn(h)

    # --- Streaming profiling ---
    rng = np.random.default_rng(SEED)
    stream_episodes = [generate_streaming_episode(rng) for _ in range(8)]

    streaming_timer = Timer()
    model.eval()

    for ep_tokens in stream_episodes:
        model.reset_memory()
        T = len(ep_tokens)
        K = model.cfg.n_oscillators
        states_r = [torch.zeros(1, K, device=device, dtype=torch.float32) for _ in range(N_LAYERS)]
        states_i = [torch.zeros(1, K, device=device, dtype=torch.float32) for _ in range(N_LAYERS)]

        pos = 0
        chunk_size = DEFAULT_CHUNK
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk_ids = torch.tensor(
                ep_tokens[start:end], dtype=torch.long, device=device,
            ).unsqueeze(0)

            with streaming_timer.section("streaming_step"):
                with torch.inference_mode():
                    result, states_r, states_i, _ = model.forward_with_memory(
                        chunk_ids, states_r, states_i, pos,
                    )
            pos += (end - start)

    # --- Profile individual streaming components ---
    model.reset_memory()
    states_r = [torch.zeros(1, K, device=device, dtype=torch.float32) for _ in range(N_LAYERS)]
    states_i = [torch.zeros(1, K, device=device, dtype=torch.float32) for _ in range(N_LAYERS)]

    # Profile eviction
    for _ in range(50):
        evicted = list(range(DEFAULT_CHUNK))
        with streaming_timer.section("eviction"):
            with torch.no_grad():
                evicted_tensor = torch.tensor(evicted, device=device).unsqueeze(0)
                hidden = model.embedding(evicted_tensor).mean(dim=1).squeeze(0)
            model._process_eviction(evicted, hidden, 0)

    # Profile retrieval search (need archive first)
    for _ in range(20):
        query = list(range(32))
        with streaming_timer.section("retrieval_search"):
            model._get_retrieval_context(query, device)

    # Profile TRN step_single
    dummy_tok = model.embedding(torch.tensor([[0]], device=device)).squeeze(1)
    dummy_tok = model.drop_emb(dummy_tok)
    for _ in range(200):
        with streaming_timer.section("trn_step_single"):
            with torch.no_grad():
                h = model.blocks[0].norm1(dummy_tok)
                _, states_r[0], states_i[0] = model.blocks[0].trn.step_single(
                    h, states_r[0], states_i[0], 0,
                )

    # --- Build profile ---
    total_time_s = timer.total_ms("step") / 1000.0
    profile = BaselineProfile(
        step_time_ms=timer.mean_ms("step"),
        forward_time_ms=timer.mean_ms("forward"),
        backward_time_ms=timer.mean_ms("backward"),
        optimizer_time_ms=timer.mean_ms("optimizer"),
        window_mask_time_ms=timer.mean_ms("window_mask"),
        trn_scan_time_ms=timer.mean_ms("trn_scan"),
        tokens_per_second=total_tokens / max(total_time_s, 1e-6),
        samples_per_second=(steps * batch_size) / max(total_time_s, 1e-6),
        gpu_mem_peak_mb=train_gpu.get("gpu_mem_peak_mb", 0),
        streaming_step_time_ms=streaming_timer.mean_ms("streaming_step"),
        streaming_tokens_per_second=(
            480.0 / (streaming_timer.mean_ms("streaming_step") * (480 / DEFAULT_CHUNK) / 1000.0)
            if streaming_timer.mean_ms("streaming_step") > 0 else 0
        ),
        streaming_eviction_time_ms=streaming_timer.mean_ms("eviction"),
        streaming_retrieval_time_ms=streaming_timer.mean_ms("retrieval_search"),
        streaming_trn_state_time_ms=streaming_timer.mean_ms("trn_step_single"),
    )

    # Print breakdown
    print(f"\n  Training step breakdown (batch_size={batch_size}, seq_len={seq_len}):")
    print(f"    step_time:      {profile.step_time_ms:8.2f} ms")
    print(f"    forward:        {profile.forward_time_ms:8.2f} ms")
    print(f"    backward:       {profile.backward_time_ms:8.2f} ms")
    print(f"    optimizer:      {profile.optimizer_time_ms:8.2f} ms")
    print(f"    window_mask:    {profile.window_mask_time_ms:8.2f} ms")
    print(f"    trn_scan:       {profile.trn_scan_time_ms:8.2f} ms")
    print(f"    tokens/sec:     {profile.tokens_per_second:8.0f}")
    print(f"    samples/sec:    {profile.samples_per_second:8.1f}")
    print(f"    GPU mem peak:   {profile.gpu_mem_peak_mb:8.1f} MB")
    print(f"\n  Streaming eval breakdown (chunk_size={DEFAULT_CHUNK}):")
    print(f"    chunk_step:     {profile.streaming_step_time_ms:8.2f} ms")
    print(f"    eviction:       {profile.streaming_eviction_time_ms:8.2f} ms")
    print(f"    retrieval:      {profile.streaming_retrieval_time_ms:8.2f} ms")
    print(f"    trn_step_single:{profile.streaming_trn_state_time_ms:8.2f} ms")
    print(f"    tokens/sec:     {profile.streaming_tokens_per_second:8.0f}")

    return profile


# ---------------------------------------------------------------------------
# Phase 2: Batch size sweep
# ---------------------------------------------------------------------------

@dataclass
class BatchSweepResult:
    batch_size: int = 0
    step_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    gpu_mem_peak_mb: float = 0.0
    oom: bool = False
    loss_mean: float = 0.0


def batch_sweep(
    device: torch.device,
    steps: int = 50,
    warmup: int = 5,
    seq_len: int = 480,
) -> list[BatchSweepResult]:
    """Phase 2: Find optimal batch size."""
    print("\n=== Phase 2: Batch Size Sweep ===")

    candidates = [4, 8, 16, 32, 64, 128]
    results = []

    for bs in candidates:
        print(f"\n  batch_size={bs}...", end=" ", flush=True)
        seed_everything(SEED)
        cfg = make_cfg()
        reset_gpu_stats()

        try:
            model = build_trimemory(cfg).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            data = generate_training_data(max(bs * 4, 64), seq_len, seed=SEED).to(device)
            dataset = torch.utils.data.TensorDataset(data)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=bs, shuffle=True, drop_last=True,
            )

            model.train()
            loader_it = iter(loader)
            losses = []
            timer = Timer()

            for step in range(1, steps + warmup + 1):
                try:
                    (batch_ids,) = next(loader_it)
                except StopIteration:
                    loader_it = iter(loader)
                    (batch_ids,) = next(loader_it)
                batch_ids = batch_ids.to(device)

                is_measured = step > warmup
                if is_measured:
                    timer.start("step")

                out = model(batch_ids, labels=batch_ids)
                loss = out["loss"]
                if torch.isfinite(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if is_measured:
                        losses.append(loss.item())

                if is_measured:
                    timer.stop("step")

            gpu = get_gpu_stats()
            total_time_s = timer.total_ms("step") / 1000.0
            total_tokens = steps * bs * seq_len

            r = BatchSweepResult(
                batch_size=bs,
                step_time_ms=timer.mean_ms("step"),
                tokens_per_second=total_tokens / max(total_time_s, 1e-6),
                samples_per_second=(steps * bs) / max(total_time_s, 1e-6),
                gpu_mem_peak_mb=gpu.get("gpu_mem_peak_mb", 0),
                oom=False,
                loss_mean=float(np.mean(losses)) if losses else float("inf"),
            )
            print(f"step={r.step_time_ms:.1f}ms tok/s={r.tokens_per_second:.0f} mem={r.gpu_mem_peak_mb:.0f}MB loss={r.loss_mean:.3f}")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"OOM at batch_size={bs}")
                r = BatchSweepResult(batch_size=bs, oom=True)
                reset_gpu_stats()
            else:
                raise

        results.append(r)
        del model, optimizer
        reset_gpu_stats()

    # Find best
    valid = [r for r in results if not r.oom]
    if valid:
        best = max(valid, key=lambda r: r.tokens_per_second)
        print(f"\n  Best: batch_size={best.batch_size} ({best.tokens_per_second:.0f} tok/s)")

    return results


# ---------------------------------------------------------------------------
# Phase 3: Mixed precision comparison
# ---------------------------------------------------------------------------

@dataclass
class PrecisionResult:
    precision_mode: str = ""
    step_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    gpu_mem_peak_mb: float = 0.0
    nan_count: int = 0
    inf_count: int = 0
    loss_mean: float = 0.0
    loss_delta: float = 0.0


def precision_compare(
    device: torch.device,
    batch_size: int = 16,
    steps: int = 50,
    warmup: int = 5,
    seq_len: int = 480,
) -> list[PrecisionResult]:
    """Phase 3: Compare fp32 / bf16 / fp16."""
    print("\n=== Phase 3: Mixed Precision Comparison ===")

    modes = ["fp32", "bf16", "fp16"]
    results = []
    fp32_loss = None

    for mode in modes:
        print(f"\n  {mode}...", end=" ", flush=True)
        seed_everything(SEED)
        cfg = make_cfg()
        reset_gpu_stats()

        model = build_trimemory(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        data = generate_training_data(max(batch_size * 4, 64), seq_len, seed=SEED).to(device)
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        )

        use_amp = mode != "fp32"
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mode, torch.float32)
        scaler = torch.amp.GradScaler("cuda", enabled=(mode == "fp16")) if device.type == "cuda" else None

        model.train()
        loader_it = iter(loader)
        timer = Timer()
        losses = []
        nan_count = 0
        inf_count = 0

        for step in range(1, steps + warmup + 1):
            try:
                (batch_ids,) = next(loader_it)
            except StopIteration:
                loader_it = iter(loader)
                (batch_ids,) = next(loader_it)
            batch_ids = batch_ids.to(device)
            is_measured = step > warmup

            if is_measured:
                timer.start("step")

            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = model(batch_ids, labels=batch_ids)
                    loss = out["loss"]
            else:
                out = model(batch_ids, labels=batch_ids)
                loss = out["loss"]

            if torch.isnan(loss):
                nan_count += 1
            elif torch.isinf(loss):
                inf_count += 1
            elif torch.isfinite(loss):
                optimizer.zero_grad()
                if scaler and mode == "fp16":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if is_measured:
                    losses.append(loss.item())

            if is_measured:
                timer.stop("step")

        gpu = get_gpu_stats()
        total_time_s = timer.total_ms("step") / 1000.0
        total_tokens = steps * batch_size * seq_len
        loss_mean = float(np.mean(losses)) if losses else float("inf")

        if mode == "fp32":
            fp32_loss = loss_mean

        r = PrecisionResult(
            precision_mode=mode,
            step_time_ms=timer.mean_ms("step"),
            tokens_per_second=total_tokens / max(total_time_s, 1e-6),
            samples_per_second=(steps * batch_size) / max(total_time_s, 1e-6),
            gpu_mem_peak_mb=gpu.get("gpu_mem_peak_mb", 0),
            nan_count=nan_count,
            inf_count=inf_count,
            loss_mean=loss_mean,
            loss_delta=loss_mean - fp32_loss if fp32_loss is not None else 0.0,
        )
        print(f"step={r.step_time_ms:.1f}ms tok/s={r.tokens_per_second:.0f} "
              f"mem={r.gpu_mem_peak_mb:.0f}MB nan={nan_count} loss={loss_mean:.3f}")
        results.append(r)
        del model, optimizer
        reset_gpu_stats()

    return results


# ---------------------------------------------------------------------------
# Phase 4: Python overhead measurement + vectorized mask
# ---------------------------------------------------------------------------

@dataclass
class PythonOverheadResult:
    variant: str = ""
    step_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    python_overhead_ms: float = 0.0
    window_mask_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    saliency_time_ms: float = 0.0


def _vectorized_window_mask(T: int, W: int, device: torch.device) -> Tensor:
    """Vectorized banded causal mask -- no Python for-loop."""
    row = torch.arange(T, device=device).unsqueeze(1)
    col = torch.arange(T, device=device).unsqueeze(0)
    mask = torch.where(
        (col <= row) & (col >= row - W + 1),
        torch.tensor(0.0, device=device),
        torch.tensor(float("-inf"), device=device),
    )
    return mask


def _vectorized_token_bag(token_ids: list[int], vocab_size: int) -> Tensor:
    """Vectorized bag-of-tokens -- no Python for-loop."""
    ids = torch.tensor(token_ids, dtype=torch.long)
    ids = ids.clamp(0, vocab_size - 1)
    bag = torch.zeros(vocab_size, dtype=torch.float32)
    bag.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
    norm = bag.norm()
    if norm > 0:
        bag = bag / norm
    return bag


def _vectorized_saliency_score(
    token_ids: list[int], vocab_size: int = 256,
) -> float:
    """Vectorized saliency scoring -- minimal Python loops."""
    t = torch.tensor(token_ids, dtype=torch.float32)
    n = len(token_ids)

    # Number score
    high_count = (t >= vocab_size * 3 / 4).sum().item()
    number_score = high_count / n

    # Entity score (max consecutive run of high tokens)
    high_mask = (t >= vocab_size / 2).int()
    # Use diff-based run detection
    if high_mask.sum() > 0:
        padded = torch.cat([torch.zeros(1, dtype=torch.int32), high_mask, torch.zeros(1, dtype=torch.int32)])
        diffs = padded.diff()
        starts = (diffs == 1).nonzero(as_tuple=True)[0]
        ends = (diffs == -1).nonzero(as_tuple=True)[0]
        if len(starts) > 0 and len(ends) > 0:
            runs = ends[:len(starts)] - starts[:len(ends)]
            max_run = runs.max().item() if len(runs) > 0 else 0
        else:
            max_run = 0
    else:
        max_run = 0
    entity_score = min(1.0, max_run / 4.0)

    # Variance score
    variance_score = min(1.0, t.std().item() / (vocab_size / 4)) if n > 1 else 0.0

    # Rare token score
    rare_threshold = vocab_size * 7 // 8
    rare_count = (t >= rare_threshold).sum().item()
    rare_score = min(1.0, rare_count / max(n * 0.1, 1))

    total = 0.3 * number_score + 0.2 * entity_score + 0.15 * 0.0 + 0.15 * variance_score + 0.1 * rare_score
    return total


def python_overhead_check(
    device: torch.device,
    steps: int = 50,
    warmup: int = 5,
    batch_size: int = 16,
    seq_len: int = 480,
) -> list[PythonOverheadResult]:
    """Phase 4: Measure Python-side overhead vs vectorized alternatives."""
    print("\n=== Phase 4: Python Overhead Analysis ===")

    results = []
    timer = Timer()

    # --- Measure window mask: original vs vectorized ---
    print("  Window mask comparison:")
    W = DEFAULT_WINDOW
    for _ in range(100):
        with timer.section("mask_original"):
            mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
            for i in range(seq_len):
                start = max(0, i - W + 1)
                mask[i, start: i + 1] = 0.0

    for _ in range(100):
        with timer.section("mask_vectorized"):
            mask = _vectorized_window_mask(seq_len, W, device)

    print(f"    Original:    {timer.mean_ms('mask_original'):.3f} ms")
    print(f"    Vectorized:  {timer.mean_ms('mask_vectorized'):.3f} ms")
    print(f"    Speedup:     {timer.mean_ms('mask_original') / max(timer.mean_ms('mask_vectorized'), 0.001):.1f}x")

    # --- Measure token bag: original vs vectorized ---
    print("  Token bag comparison:")
    test_tokens = list(range(32))
    for _ in range(500):
        with timer.section("bag_original"):
            bag = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
            for t in test_tokens:
                if 0 <= t < VOCAB_SIZE:
                    bag[t] += 1.0
            norm = bag.norm()
            if norm > 0:
                bag = bag / norm

    for _ in range(500):
        with timer.section("bag_vectorized"):
            bag = _vectorized_token_bag(test_tokens, VOCAB_SIZE)

    print(f"    Original:    {timer.mean_ms('bag_original'):.3f} ms")
    print(f"    Vectorized:  {timer.mean_ms('bag_vectorized'):.3f} ms")
    print(f"    Speedup:     {timer.mean_ms('bag_original') / max(timer.mean_ms('bag_vectorized'), 0.001):.1f}x")

    # --- Measure saliency scoring ---
    print("  Saliency scoring comparison:")
    from trimemory.tri_memory import SaliencyArchiver
    archiver = SaliencyArchiver(threshold=0.15, vocab_size=VOCAB_SIZE)

    for _ in range(500):
        with timer.section("saliency_original"):
            archiver.score(test_tokens)

    for _ in range(500):
        with timer.section("saliency_vectorized"):
            _vectorized_saliency_score(test_tokens, VOCAB_SIZE)

    print(f"    Original:    {timer.mean_ms('saliency_original'):.3f} ms")
    print(f"    Vectorized:  {timer.mean_ms('saliency_vectorized'):.3f} ms")
    print(f"    Speedup:     {timer.mean_ms('saliency_original') / max(timer.mean_ms('saliency_vectorized'), 0.001):.1f}x")

    # --- Training with optimized mask ---
    print("\n  Training comparison: original vs cached mask:")

    for variant_name, use_cached_mask in [("original", False), ("cached_mask", True)]:
        seed_everything(SEED)
        cfg = make_cfg()
        reset_gpu_stats()
        model = build_trimemory(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        data = generate_training_data(max(batch_size * 4, 64), seq_len, seed=SEED).to(device)
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        )

        # Patch mask if using cached version
        cached_masks = {}
        if use_cached_mask:
            for block in model.blocks:
                orig_make = block._make_window_mask
                def make_cached_mask(T, device, _orig=orig_make, _cache=cached_masks):
                    key = (T, str(device))
                    if key not in _cache:
                        _cache[key] = _vectorized_window_mask(T, block.window_size, device)
                    return _cache[key]
                block._make_window_mask = make_cached_mask

        model.train()
        loader_it = iter(loader)
        step_timer = Timer()

        for step in range(1, steps + warmup + 1):
            try:
                (batch_ids,) = next(loader_it)
            except StopIteration:
                loader_it = iter(loader)
                (batch_ids,) = next(loader_it)
            batch_ids = batch_ids.to(device)
            is_measured = step > warmup

            if is_measured:
                step_timer.start("step")
            out = model(batch_ids, labels=batch_ids)
            loss = out["loss"]
            if torch.isfinite(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if is_measured:
                step_timer.stop("step")

        total_time_s = step_timer.total_ms("step") / 1000.0
        total_tokens = steps * batch_size * seq_len

        r = PythonOverheadResult(
            variant=variant_name,
            step_time_ms=step_timer.mean_ms("step"),
            tokens_per_second=total_tokens / max(total_time_s, 1e-6),
            window_mask_time_ms=timer.mean_ms(f"mask_{variant_name.split('_')[0] if 'cached' not in variant_name else 'vectorized'}"),
        )
        print(f"    {variant_name:15s}: step={r.step_time_ms:.2f}ms tok/s={r.tokens_per_second:.0f}")
        results.append(r)
        del model, optimizer
        reset_gpu_stats()

    return results


# ---------------------------------------------------------------------------
# Phase 5: Retrieval batching (streaming path)
# ---------------------------------------------------------------------------

@dataclass
class RetrievalBatchResult:
    variant: str = ""
    retrieval_calls: int = 0
    avg_retrieved_chunks: float = 0.0
    streaming_step_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    total_time_ms: float = 0.0


def retrieval_batching_check(
    device: torch.device,
    n_episodes: int = 8,
) -> list[RetrievalBatchResult]:
    """Phase 5: Compare per-chunk vs batched retrieval calls."""
    print("\n=== Phase 5: Retrieval Batching ===")

    results = []
    rng = np.random.default_rng(SEED)
    episodes = [generate_streaming_episode(rng) for _ in range(n_episodes)]

    for variant in ["per_chunk", "per_2chunk"]:
        seed_everything(SEED)
        cfg = make_cfg()
        model = build_trimemory(cfg).to(device)
        model.eval()
        timer = Timer()
        total_ret_calls = 0

        for ep_tokens in episodes:
            model.reset_memory()
            T = len(ep_tokens)
            K = model.cfg.n_oscillators
            states_r = [torch.zeros(1, K, device=device, dtype=torch.float32) for _ in range(N_LAYERS)]
            states_i = [torch.zeros(1, K, device=device, dtype=torch.float32) for _ in range(N_LAYERS)]

            ret_call_count = [0]
            original_get_ret = model._get_retrieval_context
            def counting_get_ret(q, d, _orig=original_get_ret, _c=ret_call_count):
                _c[0] += 1
                return _orig(q, d)
            model._get_retrieval_context = counting_get_ret

            chunk_size = DEFAULT_CHUNK
            if variant == "per_2chunk":
                chunk_size = DEFAULT_CHUNK * 2

            pos = 0
            with torch.inference_mode():
                for start in range(0, T, chunk_size):
                    end = min(start + chunk_size, T)
                    chunk_ids = torch.tensor(
                        ep_tokens[start:end], dtype=torch.long, device=device,
                    ).unsqueeze(0)
                    with timer.section("step"):
                        result, states_r, states_i, _ = model.forward_with_memory(
                            chunk_ids, states_r, states_i, pos,
                        )
                    pos += (end - start)

            total_ret_calls += ret_call_count[0]
            model._get_retrieval_context = original_get_ret

        total_time = timer.total_ms("step")
        total_tokens = sum(len(e) for e in episodes)

        r = RetrievalBatchResult(
            variant=variant,
            retrieval_calls=total_ret_calls,
            streaming_step_time_ms=timer.mean_ms("step"),
            tokens_per_second=total_tokens / (total_time / 1000.0) if total_time > 0 else 0,
            total_time_ms=total_time,
        )
        print(f"    {variant:15s}: ret_calls={r.retrieval_calls} "
              f"step={r.streaming_step_time_ms:.2f}ms "
              f"tok/s={r.tokens_per_second:.0f}")
        results.append(r)
        del model
        reset_gpu_stats()

    return results


# ---------------------------------------------------------------------------
# Phase 6: Compile / kernel optimizations
# ---------------------------------------------------------------------------

@dataclass
class CompileResult:
    variant: str = ""
    step_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    startup_overhead_s: float = 0.0
    gpu_mem_peak_mb: float = 0.0


def compile_check(
    device: torch.device,
    batch_size: int = 16,
    steps: int = 50,
    warmup: int = 10,
    seq_len: int = 480,
) -> list[CompileResult]:
    """Phase 6: torch.compile, cudnn.benchmark, pinned memory."""
    print("\n=== Phase 6: Compile / Kernel Optimization ===")

    variants = [
        ("baseline", False, False, False),
        ("cudnn_benchmark", True, False, False),
        ("pin_memory", False, True, False),
    ]

    # Only try torch.compile on CUDA with PyTorch >= 2.1
    if device.type == "cuda" and hasattr(torch, "compile"):
        variants.append(("torch_compile", False, False, True))

    results = []

    for name, use_cudnn_bench, use_pin, use_compile in variants:
        print(f"\n  {name}...", end=" ", flush=True)
        seed_everything(SEED)
        cfg = make_cfg()
        reset_gpu_stats()

        if use_cudnn_bench:
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.benchmark = False

        model = build_trimemory(cfg).to(device)

        compile_start = time.perf_counter()
        if use_compile:
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                print(f"compile failed: {e}")
                results.append(CompileResult(variant=name, step_time_ms=float("inf")))
                del model
                reset_gpu_stats()
                continue
        compile_time = time.perf_counter() - compile_start

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        data = generate_training_data(max(batch_size * 4, 64), seq_len, seed=SEED)
        if use_pin:
            data = data.pin_memory()
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            pin_memory=use_pin and not data.is_pinned(),
        )

        model.train()
        loader_it = iter(loader)
        timer = Timer()

        try:
            for step in range(1, steps + warmup + 1):
                try:
                    (batch_ids,) = next(loader_it)
                except StopIteration:
                    loader_it = iter(loader)
                    (batch_ids,) = next(loader_it)
                batch_ids = batch_ids.to(device, non_blocking=use_pin)
                is_measured = step > warmup

                if is_measured:
                    timer.start("step")
                out = model(batch_ids, labels=batch_ids)
                loss = out["loss"]
                if torch.isfinite(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if is_measured:
                    timer.stop("step")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(CompileResult(variant=name, step_time_ms=float("inf")))
            del model, optimizer
            torch.backends.cudnn.benchmark = False
            reset_gpu_stats()
            continue

        gpu = get_gpu_stats()
        total_time_s = timer.total_ms("step") / 1000.0
        total_tokens = steps * batch_size * seq_len

        r = CompileResult(
            variant=name,
            step_time_ms=timer.mean_ms("step"),
            tokens_per_second=total_tokens / max(total_time_s, 1e-6),
            startup_overhead_s=compile_time,
            gpu_mem_peak_mb=gpu.get("gpu_mem_peak_mb", 0),
        )
        print(f"step={r.step_time_ms:.1f}ms tok/s={r.tokens_per_second:.0f} mem={r.gpu_mem_peak_mb:.0f}MB")
        results.append(r)

        del model, optimizer
        torch.backends.cudnn.benchmark = False
        reset_gpu_stats()

    return results


# ---------------------------------------------------------------------------
# Final report generation
# ---------------------------------------------------------------------------

def generate_final_report(
    baseline: BaselineProfile,
    batch_results: list[BatchSweepResult],
    precision_results: list[PrecisionResult],
    overhead_results: list[PythonOverheadResult],
    retrieval_results: list[RetrievalBatchResult],
    compile_results: list[CompileResult],
    out_dir: Path,
):
    """Generate final comparison CSV, markdown, and plots."""
    print("\n=== Generating Final Report ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save all phase results as JSON
    all_data = {
        "baseline": asdict(baseline),
        "batch_sweep": [asdict(r) for r in batch_results],
        "precision": [asdict(r) for r in precision_results],
        "python_overhead": [asdict(r) for r in overhead_results],
        "retrieval_batching": [asdict(r) for r in retrieval_results],
        "compile": [asdict(r) for r in compile_results],
    }
    with open(out_dir / "all_phase_data.json", "w") as f:
        json.dump(all_data, f, indent=2, default=str)

    # --- Phase summaries ---

    # Phase 2: batch sweep CSV
    if batch_results:
        with open(out_dir / "batch_sweep.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(batch_results[0]).keys()))
            w.writeheader()
            for r in batch_results:
                w.writerow(asdict(r))

    # Phase 3: precision CSV
    if precision_results:
        with open(out_dir / "precision_compare.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(precision_results[0]).keys()))
            w.writeheader()
            for r in precision_results:
                w.writerow(asdict(r))

    # --- Final optimization report CSV ---
    rows = []
    base_step = baseline.step_time_ms
    base_tok = baseline.tokens_per_second

    # Best batch size
    valid_batch = [r for r in batch_results if not r.oom]
    if valid_batch:
        best_batch = max(valid_batch, key=lambda r: r.tokens_per_second)
        adopt_batch = best_batch.tokens_per_second > base_tok * 1.05
        rows.append({
            "optimization": f"batch_size_{best_batch.batch_size}",
            "step_time_ms_before": base_step,
            "step_time_ms_after": best_batch.step_time_ms,
            "tokens_per_second_before": base_tok,
            "tokens_per_second_after": best_batch.tokens_per_second,
            "samples_per_second_before": baseline.samples_per_second,
            "samples_per_second_after": best_batch.samples_per_second,
            "gpu_mem_peak_before": baseline.gpu_mem_peak_mb,
            "gpu_mem_peak_after": best_batch.gpu_mem_peak_mb,
            "accuracy_delta": "N/A",
            "adopt_yes_no": "yes" if adopt_batch else "no",
        })

    # Best precision
    if precision_results:
        fp32 = [r for r in precision_results if r.precision_mode == "fp32"]
        non_fp32 = [r for r in precision_results if r.precision_mode != "fp32" and r.nan_count == 0]
        if non_fp32:
            best_prec = max(non_fp32, key=lambda r: r.tokens_per_second)
            adopt_prec = best_prec.tokens_per_second > base_tok * 1.05 and best_prec.nan_count == 0
            rows.append({
                "optimization": f"precision_{best_prec.precision_mode}",
                "step_time_ms_before": base_step,
                "step_time_ms_after": best_prec.step_time_ms,
                "tokens_per_second_before": base_tok,
                "tokens_per_second_after": best_prec.tokens_per_second,
                "samples_per_second_before": baseline.samples_per_second,
                "samples_per_second_after": best_prec.samples_per_second,
                "gpu_mem_peak_before": baseline.gpu_mem_peak_mb,
                "gpu_mem_peak_after": best_prec.gpu_mem_peak_mb,
                "accuracy_delta": f"{best_prec.loss_delta:+.4f}",
                "adopt_yes_no": "yes" if adopt_prec else "no",
            })

    # Cached mask
    if overhead_results:
        orig = [r for r in overhead_results if r.variant == "original"]
        cached = [r for r in overhead_results if r.variant == "cached_mask"]
        if orig and cached:
            adopt_mask = cached[0].tokens_per_second > orig[0].tokens_per_second * 1.02
            rows.append({
                "optimization": "cached_vectorized_mask",
                "step_time_ms_before": orig[0].step_time_ms,
                "step_time_ms_after": cached[0].step_time_ms,
                "tokens_per_second_before": orig[0].tokens_per_second,
                "tokens_per_second_after": cached[0].tokens_per_second,
                "samples_per_second_before": "N/A",
                "samples_per_second_after": "N/A",
                "gpu_mem_peak_before": baseline.gpu_mem_peak_mb,
                "gpu_mem_peak_after": "N/A",
                "accuracy_delta": "0.0",
                "adopt_yes_no": "yes" if adopt_mask else "no",
            })

    # Retrieval batching
    if retrieval_results:
        per_chunk = [r for r in retrieval_results if r.variant == "per_chunk"]
        per_2chunk = [r for r in retrieval_results if r.variant == "per_2chunk"]
        if per_chunk and per_2chunk:
            adopt_ret = per_2chunk[0].tokens_per_second > per_chunk[0].tokens_per_second * 1.05
            rows.append({
                "optimization": "retrieval_2x_chunk",
                "step_time_ms_before": per_chunk[0].streaming_step_time_ms,
                "step_time_ms_after": per_2chunk[0].streaming_step_time_ms,
                "tokens_per_second_before": per_chunk[0].tokens_per_second,
                "tokens_per_second_after": per_2chunk[0].tokens_per_second,
                "samples_per_second_before": "N/A",
                "samples_per_second_after": "N/A",
                "gpu_mem_peak_before": "N/A",
                "gpu_mem_peak_after": "N/A",
                "accuracy_delta": "N/A",
                "adopt_yes_no": "yes" if adopt_ret else "no",
            })

    # Compile results
    if compile_results:
        comp_baseline = [r for r in compile_results if r.variant == "baseline"]
        for cr in compile_results:
            if cr.variant == "baseline":
                continue
            if comp_baseline:
                adopt_comp = cr.tokens_per_second > comp_baseline[0].tokens_per_second * 1.05
                rows.append({
                    "optimization": cr.variant,
                    "step_time_ms_before": comp_baseline[0].step_time_ms,
                    "step_time_ms_after": cr.step_time_ms,
                    "tokens_per_second_before": comp_baseline[0].tokens_per_second,
                    "tokens_per_second_after": cr.tokens_per_second,
                    "samples_per_second_before": "N/A",
                    "samples_per_second_after": "N/A",
                    "gpu_mem_peak_before": comp_baseline[0].gpu_mem_peak_mb,
                    "gpu_mem_peak_after": cr.gpu_mem_peak_mb,
                    "accuracy_delta": "0.0",
                    "adopt_yes_no": "yes" if adopt_comp else "no",
                })

    # Write CSV
    csv_path = out_dir / "final_optimization_report.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # --- Markdown report ---
    md = [
        "# Tri-Memory Performance Optimization Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}",
        f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'}",
        "",
        "## Phase 1: Baseline",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| step_time_ms | {baseline.step_time_ms:.2f} |",
        f"| forward_time_ms | {baseline.forward_time_ms:.2f} |",
        f"| backward_time_ms | {baseline.backward_time_ms:.2f} |",
        f"| optimizer_time_ms | {baseline.optimizer_time_ms:.2f} |",
        f"| window_mask_time_ms | {baseline.window_mask_time_ms:.3f} |",
        f"| trn_scan_time_ms | {baseline.trn_scan_time_ms:.3f} |",
        f"| tokens_per_second | {baseline.tokens_per_second:.0f} |",
        f"| samples_per_second | {baseline.samples_per_second:.1f} |",
        f"| gpu_mem_peak_mb | {baseline.gpu_mem_peak_mb:.1f} |",
        "",
        "### Streaming Eval Breakdown",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| streaming_step_time_ms | {baseline.streaming_step_time_ms:.2f} |",
        f"| streaming_eviction_time_ms | {baseline.streaming_eviction_time_ms:.3f} |",
        f"| streaming_retrieval_time_ms | {baseline.streaming_retrieval_time_ms:.3f} |",
        f"| streaming_trn_state_time_ms | {baseline.streaming_trn_state_time_ms:.3f} |",
        f"| streaming_tokens_per_second | {baseline.streaming_tokens_per_second:.0f} |",
        "",
    ]

    # Phase 2
    if batch_results:
        md.extend(["## Phase 2: Batch Size Sweep", ""])
        md.append("| batch_size | step_time_ms | tokens/s | samples/s | gpu_mem_mb | oom |")
        md.append("|-----------|-------------|---------|----------|-----------|-----|")
        for r in batch_results:
            md.append(f"| {r.batch_size} | {r.step_time_ms:.1f} | {r.tokens_per_second:.0f} | {r.samples_per_second:.1f} | {r.gpu_mem_peak_mb:.0f} | {r.oom} |")
        md.append("")

    # Phase 3
    if precision_results:
        md.extend(["## Phase 3: Mixed Precision", ""])
        md.append("| mode | step_time_ms | tokens/s | gpu_mem_mb | nan | inf | loss_delta |")
        md.append("|------|-------------|---------|-----------|-----|-----|------------|")
        for r in precision_results:
            md.append(f"| {r.precision_mode} | {r.step_time_ms:.1f} | {r.tokens_per_second:.0f} | {r.gpu_mem_peak_mb:.0f} | {r.nan_count} | {r.inf_count} | {r.loss_delta:+.4f} |")
        md.append("")

    # Phase 4
    if overhead_results:
        md.extend(["## Phase 4: Python Overhead", ""])
        md.append("| variant | step_time_ms | tokens/s |")
        md.append("|---------|-------------|---------|")
        for r in overhead_results:
            md.append(f"| {r.variant} | {r.step_time_ms:.2f} | {r.tokens_per_second:.0f} |")
        md.append("")

    # Phase 5
    if retrieval_results:
        md.extend(["## Phase 5: Retrieval Batching", ""])
        md.append("| variant | ret_calls | step_time_ms | tokens/s |")
        md.append("|---------|----------|-------------|---------|")
        for r in retrieval_results:
            md.append(f"| {r.variant} | {r.retrieval_calls} | {r.streaming_step_time_ms:.2f} | {r.tokens_per_second:.0f} |")
        md.append("")

    # Phase 6
    if compile_results:
        md.extend(["## Phase 6: Compile / Kernel", ""])
        md.append("| variant | step_time_ms | tokens/s | gpu_mem_mb | startup_s |")
        md.append("|---------|-------------|---------|-----------|-----------|")
        for r in compile_results:
            md.append(f"| {r.variant} | {r.step_time_ms:.1f} | {r.tokens_per_second:.0f} | {r.gpu_mem_peak_mb:.0f} | {r.startup_overhead_s:.1f} |")
        md.append("")

    # Final comparison
    if rows:
        md.extend(["## Final Optimization Comparison", ""])
        md.append("| optimization | step_before | step_after | tok/s_before | tok/s_after | mem_before | mem_after | accuracy_delta | adopt |")
        md.append("|-------------|------------|-----------|-------------|------------|-----------|----------|---------------|-------|")
        for r in rows:
            md.append(
                f"| {r['optimization']} "
                f"| {r['step_time_ms_before']} "
                f"| {r['step_time_ms_after']} "
                f"| {r['tokens_per_second_before']} "
                f"| {r['tokens_per_second_after']} "
                f"| {r['gpu_mem_peak_before']} "
                f"| {r['gpu_mem_peak_after']} "
                f"| {r['accuracy_delta']} "
                f"| {r['adopt_yes_no']} |"
            )
        md.append("")

    # Adopted optimizations
    adopted = [r for r in rows if r["adopt_yes_no"] == "yes"]
    not_adopted = [r for r in rows if r["adopt_yes_no"] == "no"]

    md.extend(["## Adopted Optimizations", ""])
    if adopted:
        for r in adopted:
            before = r["tokens_per_second_before"]
            after = r["tokens_per_second_after"]
            if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before > 0:
                gain = (after - before) / before * 100
                md.append(f"- **{r['optimization']}**: {gain:+.1f}% throughput")
            else:
                md.append(f"- **{r['optimization']}**")
    else:
        md.append("- None (baseline already optimal for this model scale)")

    md.extend(["", "## Not Adopted", ""])
    if not_adopted:
        for r in not_adopted:
            md.append(f"- {r['optimization']}: insufficient gain or instability")
    else:
        md.append("- None")

    md.extend([
        "",
        "## Key Insight",
        "",
        "High GPU utilization is not the target.",
        "Practical throughput is the target.",
        "Optimization must preserve benchmark fairness.",
        "Toy models may never reach 90% GPU util if kernels are too small.",
    ])

    with open(out_dir / "final_optimization_report.md", "w") as f:
        f.write("\n".join(md))

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_dir = out_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Throughput comparison
        if rows:
            fig, ax = plt.subplots(figsize=(12, 6))
            names = [r["optimization"] for r in rows]
            before = [float(r["tokens_per_second_before"]) if isinstance(r["tokens_per_second_before"], (int, float)) else 0 for r in rows]
            after = [float(r["tokens_per_second_after"]) if isinstance(r["tokens_per_second_after"], (int, float)) else 0 for r in rows]
            x = np.arange(len(names))
            w = 0.35
            ax.bar(x - w/2, before, w, label="Before", color="#C44E52", alpha=0.8)
            ax.bar(x + w/2, after, w, label="After", color="#55A868", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.set_ylabel("Tokens/second")
            ax.set_title("Throughput Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(plot_dir / "throughput_comparison.png", dpi=150)
            plt.close(fig)

        # Step time breakdown
        fig, ax = plt.subplots(figsize=(10, 6))
        components = ["forward", "backward", "optimizer"]
        vals = [baseline.forward_time_ms, baseline.backward_time_ms, baseline.optimizer_time_ms]
        other = baseline.step_time_ms - sum(vals)
        components.append("other")
        vals.append(max(0, other))
        colors = ["#4C72B0", "#C44E52", "#55A868", "#999999"]
        ax.bar(components, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.1, f"{v:.1f}ms", ha="center", fontsize=10)
        ax.set_ylabel("Time (ms)")
        ax.set_title("Training Step Time Breakdown")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(plot_dir / "step_time_breakdown.png", dpi=150)
        plt.close(fig)

        # GPU memory comparison (batch sweep)
        if batch_results:
            valid_b = [r for r in batch_results if not r.oom]
            if valid_b:
                fig, ax = plt.subplots(figsize=(10, 6))
                bs_vals = [r.batch_size for r in valid_b]
                mem_vals = [r.gpu_mem_peak_mb for r in valid_b]
                tok_vals = [r.tokens_per_second for r in valid_b]
                ax.bar(range(len(bs_vals)), mem_vals, color="#4C72B0", alpha=0.7)
                ax.set_xticks(range(len(bs_vals)))
                ax.set_xticklabels([str(b) for b in bs_vals])
                ax.set_xlabel("Batch Size")
                ax.set_ylabel("GPU Memory Peak (MB)")
                ax2 = ax.twinx()
                ax2.plot(range(len(bs_vals)), tok_vals, "o-", color="#C44E52")
                ax2.set_ylabel("Tokens/second", color="#C44E52")
                ax.set_title("GPU Memory vs Throughput by Batch Size")
                fig.tight_layout()
                fig.savefig(plot_dir / "gpu_memory_comparison.png", dpi=150)
                plt.close(fig)

        # Optimization gains
        if rows:
            adopted_rows = [r for r in rows if r["adopt_yes_no"] == "yes"]
            if adopted_rows:
                fig, ax = plt.subplots(figsize=(10, 6))
                names = [r["optimization"] for r in adopted_rows]
                gains = []
                for r in adopted_rows:
                    b = r["tokens_per_second_before"]
                    a = r["tokens_per_second_after"]
                    if isinstance(b, (int, float)) and isinstance(a, (int, float)) and b > 0:
                        gains.append((a - b) / b * 100)
                    else:
                        gains.append(0)
                colors_g = ["#55A868" if g > 0 else "#C44E52" for g in gains]
                ax.barh(names, gains, color=colors_g)
                for i, g in enumerate(gains):
                    ax.text(g + 0.5, i, f"{g:+.1f}%", va="center")
                ax.set_xlabel("Throughput Gain (%)")
                ax.set_title("Optimization Gains")
                ax.grid(True, alpha=0.3, axis="x")
                fig.tight_layout()
                fig.savefig(plot_dir / "optimization_gains.png", dpi=150)
                plt.close(fig)

        print(f"  Plots saved to {plot_dir}")

    except ImportError:
        print("  matplotlib not available, skipping plots")

    print(f"  Report: {out_dir / 'final_optimization_report.md'}")
    print(f"  CSV: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tri-Memory Performance Profiler")
    parser.add_argument("--phase", default="all",
                        help="Phase(s) to run: 1,2,3,4,5,6 or 'all'")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100,
                        help="Training steps per measurement")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Default batch size for phases 1,3,4,6")
    parser.add_argument("--output-dir", default="artifacts/perf/")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "all":
        phases = {1, 2, 3, 4, 5, 6}
    else:
        phases = {int(p.strip()) for p in args.phase.split(",")}

    print("=" * 60)
    print("  Tri-Memory Performance Profiler")
    print(f"  Phases: {sorted(phases)}")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Steps: {args.steps}")
    print(f"  Default batch: {args.batch_size}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    baseline = BaselineProfile()
    batch_results: list[BatchSweepResult] = []
    precision_results: list[PrecisionResult] = []
    overhead_results: list[PythonOverheadResult] = []
    retrieval_results: list[RetrievalBatchResult] = []
    compile_results: list[CompileResult] = []

    if 1 in phases:
        baseline = profile_baseline(device, steps=args.steps, batch_size=args.batch_size)
        with open(out_dir / "baseline_profile.json", "w") as f:
            json.dump(asdict(baseline), f, indent=2)

    if 2 in phases:
        batch_results = batch_sweep(device, steps=min(args.steps, 50))

    if 3 in phases:
        bs = args.batch_size
        if batch_results:
            valid = [r for r in batch_results if not r.oom]
            if valid:
                best = max(valid, key=lambda r: r.tokens_per_second)
                bs = best.batch_size
        precision_results = precision_compare(device, batch_size=bs, steps=min(args.steps, 50))

    if 4 in phases:
        overhead_results = python_overhead_check(
            device, steps=min(args.steps, 50), batch_size=args.batch_size,
        )

    if 5 in phases:
        retrieval_results = retrieval_batching_check(device)

    if 6 in phases:
        bs = args.batch_size
        if batch_results:
            valid = [r for r in batch_results if not r.oom]
            if valid:
                best = max(valid, key=lambda r: r.tokens_per_second)
                bs = best.batch_size
        compile_results = compile_check(device, batch_size=bs, steps=min(args.steps, 50))

    generate_final_report(
        baseline, batch_results, precision_results,
        overhead_results, retrieval_results, compile_results,
        out_dir,
    )

    print("\n" + "=" * 60)
    print("  Profiling complete.")
    print(f"  Output: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
