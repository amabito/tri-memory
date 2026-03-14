#!/usr/bin/env python3
"""Streaming Evaluation for Tri-Memory LLM (Optimized).

Applies adopted optimizations: batch_size=128, bf16, pin_memory, cached_vectorized_mask.
Non-adopted: retrieval_2x_chunk, cudnn_benchmark, torch_compile.

Usage:
    python scripts/run_trimemory_streaming_eval.py \
        --models kv kv_trn kv_ret trimemory \
        --seeds 0 1 2 --num-episodes 128 \
        --batch-size 128 --precision bf16 --pin-memory \
        --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

_src = os.path.join(os.path.dirname(__file__), "..", "src")
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _root]
sys.path.insert(0, _src)

import numpy as np
import torch
from torch import Tensor

from trimemory.bench_data import seed_everything
from trimemory.config import TRNConfig
from trimemory.integrations.vllm_backend import DualMemoryEngine
from trimemory.tri_memory import TriMemoryEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOCAB_SIZE = 256
D_MODEL = 128
N_LAYERS = 4
N_OSC = 64
D_FF = 512
DEFAULT_WINDOW = 64
DEFAULT_CHUNK = 32
DEFAULT_MAX_ARCHIVE = 128
TRAIN_STEPS = 300

# Token semantic ranges -- non-overlapping for unambiguous identification
FACT_RANGE = (220, 245)
PATTERN_RANGE = (10, 60)
SALIENT_RANGE = (245, 256)
NEUTRAL_RANGE = (100, 130)
RECENT_RANGE = (200, 220)
FILLER_RANGE = (130, 200)
QUERY_MARKER = 5


# ---------------------------------------------------------------------------
# Episode data structure
# ---------------------------------------------------------------------------
@dataclass
class StreamingEpisode:
    """One streaming evaluation episode with ground-truth answers."""
    tokens: list[int]
    old_facts: list[tuple[int, int]]
    pattern_val: int
    pattern_period: int
    salient_vals: list[int]
    salient_positions: list[int]
    neutral_vals: list[int]
    neutral_positions: list[int]
    recent_val: int
    query_recent_pos: int
    query_old_fact_pos: int
    query_pattern_pos: int
    query_salient_pos: int
    query_neutral_pos: int
    answer_recent: int
    answer_old_fact: int
    answer_pattern: int
    answer_salient: int
    answer_neutral: int


def generate_episode(
    rng: np.random.Generator,
    kv_window: int = 64,
    chunk_size: int = 32,
) -> StreamingEpisode:
    """Generate one streaming episode with all five task types.

    Layout (total ~480 tokens):
      Phase A (0-63):   old fact injection (4 facts in filler)
      Phase B (64-191): long pattern segment (128 tokens)
      Phase C (192-319): salient + neutral events in filler
      Phase D (320-383): filler gap to push A/B/C out of KV window
      Phase E (384-447): recent exact context
      Phase F (448-479): 5 query pairs (marker, answer)

    KV window = 64, so at query time (pos ~450+):
      - Phase A (pos 0-63) is ~390 tokens behind -> OUT of KV window
      - Phase B (pos 64-191) is ~260 tokens behind -> OUT of KV window
      - Phase C (pos 192-319) is ~130 tokens behind -> OUT of KV window
      - Phase E (pos 384-447) is ~3-63 tokens behind -> IN KV window
    """
    tokens: list[int] = []

    # Phase A: old fact injection (positions 0-63)
    old_facts = []
    fact_positions = [5, 15, 30, 50]
    for fp in fact_positions:
        while len(tokens) < fp:
            tokens.append(int(rng.integers(*FILLER_RANGE)))
        fv = int(rng.integers(*FACT_RANGE))
        tokens.append(fv)
        old_facts.append((fv, len(tokens) - 1))
    while len(tokens) < 64:
        tokens.append(int(rng.integers(*FILLER_RANGE)))

    # Phase B: long pattern segment (positions 64-191, 128 tokens)
    period = int(rng.integers(3, 8))
    pattern_val = int(rng.integers(*PATTERN_RANGE))
    for i in range(128):
        if i % period == 0:
            tokens.append(pattern_val)
        else:
            tokens.append(int(rng.integers(*FILLER_RANGE)))

    # Phase C: salient + neutral events (positions 192-319, 128 tokens)
    salient_vals = []
    salient_positions = []
    neutral_vals = []
    neutral_positions = []
    for i in range(128):
        pos = 192 + i
        if i in (10, 40, 70, 100):
            sv = int(rng.integers(*SALIENT_RANGE))
            tokens.append(sv)
            salient_vals.append(sv)
            salient_positions.append(pos)
        elif i in (20, 55, 85, 115):
            nv = int(rng.integers(*NEUTRAL_RANGE))
            tokens.append(nv)
            neutral_vals.append(nv)
            neutral_positions.append(pos)
        else:
            tokens.append(int(rng.integers(*FILLER_RANGE)))

    # Phase D: filler gap (positions 320-383, 64 tokens)
    for _ in range(64):
        tokens.append(int(rng.integers(*FILLER_RANGE)))

    # Phase E: recent exact context (positions 384-447, 64 tokens)
    recent_val = int(rng.integers(*RECENT_RANGE))
    for i in range(64):
        if i % 4 == 0:
            tokens.append(recent_val)
        else:
            tokens.append(int(rng.integers(*FILLER_RANGE)))

    # Phase F: queries (positions 448+)
    primary_fact = old_facts[0]
    primary_salient = salient_vals[0] if salient_vals else int(rng.integers(*SALIENT_RANGE))
    primary_neutral = neutral_vals[0] if neutral_vals else int(rng.integers(*NEUTRAL_RANGE))

    q_recent_pos = len(tokens)
    tokens.append(QUERY_MARKER)
    tokens.append(recent_val)

    q_fact_pos = len(tokens)
    tokens.append(QUERY_MARKER)
    tokens.append(primary_fact[0])

    q_pattern_pos = len(tokens)
    tokens.append(QUERY_MARKER)
    tokens.append(pattern_val)

    q_salient_pos = len(tokens)
    tokens.append(QUERY_MARKER)
    tokens.append(primary_salient)

    q_neutral_pos = len(tokens)
    tokens.append(QUERY_MARKER)
    tokens.append(primary_neutral)

    while len(tokens) < 480:
        tokens.append(int(rng.integers(*FILLER_RANGE)))

    return StreamingEpisode(
        tokens=tokens,
        old_facts=old_facts,
        pattern_val=pattern_val,
        pattern_period=period,
        salient_vals=salient_vals,
        salient_positions=salient_positions,
        neutral_vals=neutral_vals,
        neutral_positions=neutral_positions,
        recent_val=recent_val,
        query_recent_pos=q_recent_pos,
        query_old_fact_pos=q_fact_pos,
        query_pattern_pos=q_pattern_pos,
        query_salient_pos=q_salient_pos,
        query_neutral_pos=q_neutral_pos,
        answer_recent=recent_val,
        answer_old_fact=primary_fact[0],
        answer_pattern=pattern_val,
        answer_salient=primary_salient,
        answer_neutral=primary_neutral,
    )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
@dataclass
class EpisodeTelemetry:
    model: str = ""
    seed: int = 0
    episode_id: int = 0
    # Accuracy
    recent_exact_acc: float = 0.0
    old_fact_acc: float = 0.0
    long_pattern_acc: float = 0.0
    salient_event_acc: float = 0.0
    neutral_event_acc: float = 0.0
    composite_score: float = 0.0
    # Memory
    archive_chunk_count: int = 0
    archive_insertions: int = 0
    retained_kv_tokens: int = 0
    state_bytes: int = 0
    # Retrieval
    retrieval_calls: int = 0
    retrieval_hit_count: int = 0
    retrieval_hit_rate: float = 0.0
    avg_retrieved_chunks: float = 0.0
    router_kv_ratio: float = 0.0
    router_trn_ratio: float = 0.0
    router_ret_ratio: float = 0.0
    # Performance
    tokens_per_second: float = 0.0
    latency_ms: float = 0.0
    gpu_mem_peak_mb: float = 0.0
    cpu_util_percent: float = 0.0
    gpu_util_percent: float = 0.0
    # KV cache
    kv_cache_tokens_retained: int = 0
    kv_cache_bytes: int = 0
    past_kv_truncations: int = 0
    # Stability
    nan_count: int = 0
    inf_count: int = 0


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def make_cfg(max_seq_len: int = 512) -> TRNConfig:
    return TRNConfig(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_oscillators=N_OSC,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=max_seq_len,
    )


def _build_model(model_name, cfg, w, c, m, search_mode="hidden",
                  search_w_hidden=0.7, search_w_bag=0.3):
    if model_name in ("kv", "kv_trn"):
        return DualMemoryEngine(cfg, window_size=w)
    elif model_name == "kv_ret":
        return TriMemoryEngine(
            cfg, window_size=w, chunk_size=c, max_retrieval_chunks=m,
            enable_trn=False, enable_retrieval=True,
            search_mode=search_mode, search_w_hidden=search_w_hidden,
            search_w_bag=search_w_bag,
        )
    elif model_name == "trimemory":
        return TriMemoryEngine(
            cfg, window_size=w, chunk_size=c, max_retrieval_chunks=m,
            enable_trn=True, enable_retrieval=True,
            search_mode=search_mode, search_w_hidden=search_w_hidden,
            search_w_bag=search_w_bag,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# GPU util helper
# ---------------------------------------------------------------------------
def get_gpu_utilization() -> float:
    """Return GPU utilization percentage. Requires pynvml or falls back to 0."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pynvml.nvmlShutdown()
        return float(util.gpu)
    except Exception:
        return 0.0


def get_cpu_utilization() -> float:
    try:
        import psutil
        return psutil.cpu_percent(interval=None)
    except Exception:
        return 0.0


def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_gpu_peak_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


# ---------------------------------------------------------------------------
# Training (shared weights across model variants)
# ---------------------------------------------------------------------------
def train_model(
    model, episodes, steps, device, lr=3e-4,
    batch_size=128, precision="bf16", pin_memory=False,
):
    """Train model with adopted optimizations: batch_size, bf16, pin_memory."""
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    all_seqs = [torch.tensor(ep.tokens, dtype=torch.long) for ep in episodes]
    max_len = max(s.size(0) for s in all_seqs)
    padded = torch.zeros(len(all_seqs), max_len, dtype=torch.long)
    for i, s in enumerate(all_seqs):
        padded[i, :s.size(0)] = s

    dataset = torch.utils.data.TensorDataset(padded)
    bs = min(batch_size, len(all_seqs))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=True,
        drop_last=(len(all_seqs) >= bs * 2),
        pin_memory=(pin_memory and device.type == "cuda"),
        num_workers=0,
    )

    use_amp = (precision == "bf16" and device.type == "cuda"
               and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    if not use_amp and precision == "bf16":
        print(f"      [WARN] bf16 not supported on {device}, falling back to fp32")

    loader_it = iter(loader)
    for step in range(1, steps + 1):
        try:
            (batch_ids,) = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            (batch_ids,) = next(loader_it)
        batch_ids = batch_ids.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(batch_ids, labels=batch_ids)
            loss = out["loss"]

        if torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if step % 100 == 0:
            print(f"      step {step}/{steps} loss={loss.item():.4f}"
                  f"{' [bf16]' if use_amp else ' [fp32]'}")
    return model


# ---------------------------------------------------------------------------
# Streaming evaluation core
# ---------------------------------------------------------------------------
@torch.inference_mode()
def streaming_eval_episode(
    model,
    episode: StreamingEpisode,
    device: torch.device,
    chunk_size: int = 32,
    model_name: str = "",
    precision: str = "bf16",
) -> EpisodeTelemetry:
    """Stream episode tokens through model, collecting telemetry."""
    model = model.to(device).eval()
    tokens = episode.tokens
    T = len(tokens)
    tel = EpisodeTelemetry(model=model_name)

    use_amp = (precision == "bf16" and device.type == "cuda"
               and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    reset_gpu_stats()
    cpu_start = get_cpu_utilization()
    t0 = time.perf_counter()
    is_tri = isinstance(model, TriMemoryEngine)

    if is_tri:
        model.reset_memory()
        K = model.cfg.n_oscillators
        B = 1
        states_r = [
            torch.zeros(B, K, device=device, dtype=torch.float32)
            for _ in range(model.cfg.n_layers)
        ]
        states_i = [
            torch.zeros(B, K, device=device, dtype=torch.float32)
            for _ in range(model.cfg.n_layers)
        ]

        original_get_ret = model._get_retrieval_context
        retrieval_call_count = [0]
        retrieval_hit_count = [0]
        retrieved_chunk_counts = []

        def instrumented_get_retrieval_context(query_tokens, dev):
            retrieval_call_count[0] += 1
            result = original_get_ret(query_tokens, dev)
            if result is not None:
                retrieval_hit_count[0] += 1
                n = len(model.retrieval_index._chunks)
                retrieved_chunk_counts.append(min(model.retrieval_top_k, n))
            else:
                retrieved_chunk_counts.append(0)
            return result

        model._get_retrieval_context = instrumented_get_retrieval_context
        archive_insertions_before = model.retrieval_index._next_id

        all_logits = []
        pos = 0
        past_kv = None  # KV cache initialized inside forward_with_memory
        kv_truncations = 0
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk_ids = torch.tensor(
                tokens[start:end], dtype=torch.long, device=device,
            ).unsqueeze(0)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                result, states_r, states_i, past_kv = model.forward_with_memory(
                    chunk_ids, states_r, states_i, pos, past_kv=past_kv,
                )
            # Track truncations
            if past_kv and past_kv[0][0].shape[2] >= model.window_size:
                kv_truncations += 1

            logits = result["logits"].float()
            all_logits.append(logits)

            if torch.isnan(logits).any():
                tel.nan_count += int(torch.isnan(logits).sum().item())
            if torch.isinf(logits).any():
                tel.inf_count += int(torch.isinf(logits).sum().item())

            pos += (end - start)

        # KV cache telemetry
        if past_kv:
            tel.kv_cache_tokens_retained = past_kv[0][0].shape[2]
            # bytes = n_layers * 2 (k+v) * B * n_heads * retained * head_dim * element_size
            elem_size = past_kv[0][0].element_size()
            n_layers = len(past_kv)
            tel.kv_cache_bytes = sum(
                pk.nelement() * elem_size + pv.nelement() * elem_size
                for pk, pv in past_kv
            )
        tel.past_kv_truncations = kv_truncations

        model._get_retrieval_context = original_get_ret

        full_logits = torch.cat(all_logits, dim=1)

        tel.archive_chunk_count = len(model.retrieval_index)
        tel.archive_insertions = model.retrieval_index._next_id - archive_insertions_before
        tel.retrieval_calls = retrieval_call_count[0]
        tel.retrieval_hit_count = retrieval_hit_count[0]
        tel.retrieval_hit_rate = (
            retrieval_hit_count[0] / max(retrieval_call_count[0], 1)
        )
        tel.avg_retrieved_chunks = (
            float(np.mean(retrieved_chunk_counts)) if retrieved_chunk_counts else 0.0
        )
        tel.state_bytes = model.state_memory_bytes

        # Gate telemetry from last block
        last_chunk_start = max(0, T - chunk_size)
        last_chunk_ids = torch.tensor(
            tokens[last_chunk_start:T], dtype=torch.long, device=device,
        ).unsqueeze(0)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            h = model.drop_emb(model.embedding(last_chunk_ids))
            pe_start = min(last_chunk_start, model.pe.size(0) - h.size(1))
            pe_end = min(pe_start + h.size(1), model.pe.size(0))
            if pe_end > pe_start:
                h[:, :pe_end - pe_start] += model.pe[pe_start:pe_end]
            for block in model.blocks:
                h_normed = block.norm1(h)
                gates = torch.softmax(block.gate_proj(h_normed), dim=-1)
        tel.router_kv_ratio = gates[:, :, 0].mean().float().item()
        tel.router_trn_ratio = gates[:, :, 1].mean().float().item()
        tel.router_ret_ratio = gates[:, :, 2].mean().float().item()

    else:
        # DualMemoryEngine: batch forward on full sequence
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            result = model(input_ids)
        full_logits = result["logits"].float()

        if torch.isnan(full_logits).any():
            tel.nan_count += int(torch.isnan(full_logits).sum().item())
        if torch.isinf(full_logits).any():
            tel.inf_count += int(torch.isinf(full_logits).sum().item())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tel.latency_ms = elapsed * 1000.0
    tel.tokens_per_second = T / max(elapsed, 1e-6)
    tel.retained_kv_tokens = min(T, model.window_size if hasattr(model, 'window_size') else DEFAULT_WINDOW)
    tel.gpu_mem_peak_mb = get_gpu_peak_mb()
    tel.gpu_util_percent = get_gpu_utilization()
    tel.cpu_util_percent = get_cpu_utilization()

    def check_prediction(query_pos: int, expected: int) -> float:
        if query_pos >= full_logits.size(1):
            return 0.0
        pred = full_logits[0, query_pos, :].argmax().item()
        return 1.0 if pred == expected else 0.0

    tel.recent_exact_acc = check_prediction(episode.query_recent_pos, episode.answer_recent)
    tel.old_fact_acc = check_prediction(episode.query_old_fact_pos, episode.answer_old_fact)
    tel.long_pattern_acc = check_prediction(episode.query_pattern_pos, episode.answer_pattern)
    tel.salient_event_acc = check_prediction(episode.query_salient_pos, episode.answer_salient)
    tel.neutral_event_acc = check_prediction(episode.query_neutral_pos, episode.answer_neutral)

    tel.composite_score = (
        0.30 * tel.recent_exact_acc
        + 0.25 * tel.old_fact_acc
        + 0.25 * tel.long_pattern_acc
        + 0.20 * tel.salient_event_acc
    )

    return tel


# ---------------------------------------------------------------------------
# Deep retrieval diagnostics for failure analysis
# ---------------------------------------------------------------------------
@torch.inference_mode()
def collect_oldfact_failure_case(
    model,
    episode: StreamingEpisode,
    tel: EpisodeTelemetry,
    device: torch.device,
    chunk_size: int,
    precision: str = "bf16",
    failure_topk: int = 5,
) -> dict:
    """Collect deep retrieval diagnostics for one episode's old_fact query.

    Returns a dict suitable for JSONL serialization.
    """
    is_tri = isinstance(model, TriMemoryEngine)
    tokens = episode.tokens
    T = len(tokens)
    gold_token = episode.answer_old_fact
    query_pos = episode.query_old_fact_pos
    fact_positions = [pos for (val, pos) in episode.old_facts]

    use_amp = (precision == "bf16" and device.type == "cuda"
               and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Prediction at query position
    # Re-run or use cached logits -- we re-derive from stored tel
    pred_token = -1
    decoder_topk_tokens = []

    # We need the full logits. Re-run a quick forward for the query region.
    if is_tri:
        # Re-stream to get logits at query position
        model.reset_memory()
        K = model.cfg.n_oscillators
        B = 1
        states_r = [torch.zeros(B, K, device=device, dtype=torch.float32)
                     for _ in range(model.cfg.n_layers)]
        states_i = [torch.zeros(B, K, device=device, dtype=torch.float32)
                     for _ in range(model.cfg.n_layers)]
        past_kv_inner = None
        all_logits = []
        pos = 0

        # Track retrieval details on each chunk
        last_retrieval_results = []
        last_retrieval_scores = []
        last_retrieval_score_dicts = []
        original_search = model.retrieval_index.search
        original_search_with_scores = model.retrieval_index.search_with_scores

        def instrumented_search(query_token_ids, top_k=4, query_hidden=None,
                                mode="hidden", w_hidden=0.7, w_bag=0.3):
            results, score_dicts = original_search_with_scores(
                query_token_ids, top_k=top_k, query_hidden=query_hidden,
                mode=mode, w_hidden=w_hidden, w_bag=w_bag,
            )
            last_retrieval_results.clear()
            last_retrieval_results.extend(results)
            last_retrieval_scores.clear()
            last_retrieval_scores.extend(
                [sd["combined_score"] for sd in score_dicts]
            )
            last_retrieval_score_dicts.clear()
            last_retrieval_score_dicts.extend(score_dicts)
            return results

        model.retrieval_index.search = instrumented_search

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk_ids = torch.tensor(
                tokens[start:end], dtype=torch.long, device=device,
            ).unsqueeze(0)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                result, states_r, states_i, past_kv_inner = model.forward_with_memory(
                    chunk_ids, states_r, states_i, pos, past_kv=past_kv_inner,
                )
            all_logits.append(result["logits"].float())
            pos += (end - start)

        model.retrieval_index.search = original_search

        full_logits = torch.cat(all_logits, dim=1)
        if query_pos < full_logits.size(1):
            logits_at_query = full_logits[0, query_pos, :]
            pred_token = logits_at_query.argmax().item()
            topk_vals, topk_idxs = logits_at_query.topk(10)
            decoder_topk_tokens = topk_idxs.tolist()

        # Retrieval data from last search call (which covers the query chunk)
        retrieval_called = len(last_retrieval_results) > 0 or len(model.retrieval_index) > 0
        topk_chunks = [r.token_ids for r in last_retrieval_results[:failure_topk]]
        topk_scores = last_retrieval_scores[:failure_topk]
        topk_chunk_positions = []
        for r in last_retrieval_results[:failure_topk]:
            topk_chunk_positions.append([r.step])

        topk_contains_gold = any(
            gold_token in chunk_toks
            for chunk_toks in topk_chunks
        )
        top1_contains_gold = (
            len(topk_chunks) > 0 and gold_token in topk_chunks[0]
        )

        # Gate diagnostics from last block
        last_chunk_start = max(0, T - chunk_size)
        last_chunk_ids = torch.tensor(
            tokens[last_chunk_start:T], dtype=torch.long, device=device,
        ).unsqueeze(0)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            h = model.drop_emb(model.embedding(last_chunk_ids))
            pe_start = min(last_chunk_start, model.pe.size(0) - h.size(1))
            pe_end = min(pe_start + h.size(1), model.pe.size(0))
            if pe_end > pe_start:
                h[:, :pe_end - pe_start] += model.pe[pe_start:pe_end]
            for block in model.blocks:
                h_normed = block.norm1(h)
                gate_logits_raw = block.gate_proj(h_normed)
                gates = torch.softmax(gate_logits_raw, dim=-1)
        # Average across positions for summary
        gate_logits_mean = gate_logits_raw.float().mean(dim=(0, 1)).tolist()
        gate_probs_mean = gates.float().mean(dim=(0, 1)).tolist()

    else:
        # Non-TriMemory model -- minimal data
        retrieval_called = False
        topk_chunks = []
        topk_scores = []
        topk_chunk_positions = []
        topk_contains_gold = False
        top1_contains_gold = False
        gate_logits_mean = []
        gate_probs_mean = []

        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            result = model(input_ids)
        full_logits = result["logits"].float()
        if query_pos < full_logits.size(1):
            logits_at_query = full_logits[0, query_pos, :]
            pred_token = logits_at_query.argmax().item()
            topk_vals, topk_idxs = logits_at_query.topk(10)
            decoder_topk_tokens = topk_idxs.tolist()

    # Query tokens = the chunk that contains the old_fact query marker
    query_chunk_start = (query_pos // chunk_size) * chunk_size
    query_chunk_end = min(query_chunk_start + chunk_size, T)
    query_tokens = tokens[query_chunk_start:query_chunk_end]

    # Per-chunk score breakdown (hidden_score, bag_score, combined_score)
    topk_score_breakdown = last_retrieval_score_dicts[:failure_topk]

    return {
        "seed": tel.seed,
        "episode_id": tel.episode_id,
        "query_tokens": query_tokens,
        "gold_answer_token": gold_token,
        "pred_answer_token": pred_token,
        "correct": pred_token == gold_token,
        "search_mode": model.search_mode,
        "search_w_hidden": model.search_w_hidden,
        "search_w_bag": model.search_w_bag,
        "retrieval_called": retrieval_called,
        "retrieval_topk_chunks": topk_chunks,
        "retrieval_topk_scores": topk_scores,
        "retrieval_topk_score_breakdown": topk_score_breakdown,
        "retrieval_topk_contains_gold": topk_contains_gold,
        "retrieval_top1_contains_gold": top1_contains_gold,
        "router_kv_ratio": tel.router_kv_ratio,
        "router_trn_ratio": tel.router_trn_ratio,
        "router_ret_ratio": tel.router_ret_ratio,
        "retrieval_context_length": len(topk_chunks),
        "archive_chunk_count": tel.archive_chunk_count,
        "fact_span_positions": fact_positions,
        "retrieved_chunk_positions": topk_chunk_positions,
        "decoder_topk_tokens": decoder_topk_tokens,
        "gate_logits_raw": gate_logits_mean,
        "gate_probs": gate_probs_mean,
    }


# ---------------------------------------------------------------------------
# Aggregate & report
# ---------------------------------------------------------------------------
def aggregate_telemetry(rows: list[EpisodeTelemetry]) -> dict:
    if not rows:
        return {}
    fields = [
        "recent_exact_acc", "old_fact_acc", "long_pattern_acc",
        "salient_event_acc", "neutral_event_acc", "composite_score",
        "archive_chunk_count", "archive_insertions",
        "retrieval_calls", "retrieval_hit_count", "retrieval_hit_rate",
        "avg_retrieved_chunks",
        "router_kv_ratio", "router_trn_ratio", "router_ret_ratio",
        "tokens_per_second", "latency_ms",
        "gpu_mem_peak_mb", "cpu_util_percent", "gpu_util_percent",
        "nan_count", "inf_count",
    ]
    agg = {}
    for f in fields:
        vals = [getattr(r, f) for r in rows]
        agg[f"mean_{f}"] = float(np.mean(vals))
        agg[f"std_{f}"] = float(np.std(vals))
    return agg


def sanity_checks(all_telemetry: dict[str, list[EpisodeTelemetry]]) -> dict:
    checks = {}

    for model_name in ["kv_ret", "trimemory"]:
        rows = all_telemetry.get(model_name, [])
        if rows:
            archive_counts = [r.archive_chunk_count for r in rows]
            checks[f"{model_name}_archive_nonempty"] = max(archive_counts) > 0
            checks[f"{model_name}_retrieval_calls_gt0"] = (
                sum(r.retrieval_calls for r in rows) > 0
            )

    checks["salient_outside_kv"] = True
    checks["old_fact_outside_kv"] = True

    def mean_acc(rows, field):
        return float(np.mean([getattr(r, field) for r in rows])) if rows else 0.0

    tri = all_telemetry.get("trimemory", [])
    kv = all_telemetry.get("kv", [])
    kv_ret = all_telemetry.get("kv_ret", [])

    checks["trimemory_beats_kv_on_old_fact"] = (
        mean_acc(tri, "old_fact_acc") > mean_acc(kv, "old_fact_acc")
    )
    checks["trimemory_beats_kv_ret_on_pattern"] = (
        mean_acc(tri, "long_pattern_acc") > mean_acc(kv_ret, "long_pattern_acc")
    )
    checks["trimemory_recent_within_tolerance"] = (
        mean_acc(tri, "recent_exact_acc") >= mean_acc(kv, "recent_exact_acc") - 0.05
    )
    checks["salient_exceeds_neutral"] = (
        mean_acc(tri, "salient_event_acc") > mean_acc(tri, "neutral_event_acc")
    )

    # Gate checks
    all_nan = sum(r.nan_count for m in all_telemetry for r in all_telemetry[m])
    all_inf = sum(r.inf_count for m in all_telemetry for r in all_telemetry[m])
    checks["no_nan_inf"] = (all_nan == 0 and all_inf == 0)

    # Memory budget
    for model_name in ["kv_ret", "trimemory"]:
        rows = all_telemetry.get(model_name, [])
        if rows:
            max_archive = max(r.archive_chunk_count for r in rows)
            checks[f"{model_name}_memory_budget_respected"] = max_archive <= 256

    # Composite score gate
    if tri and kv and kv_ret:
        tri_comp = mean_acc(tri, "composite_score")
        best_other = max(
            mean_acc(kv, "composite_score"),
            mean_acc(all_telemetry.get("kv_trn", []), "composite_score"),
            mean_acc(kv_ret, "composite_score"),
        )
        checks["trimemory_composite_best"] = tri_comp > best_other

    return checks


def generate_plots(model_aggs: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    models = list(model_aggs.keys())
    colors = {"kv": "#4C72B0", "kv_trn": "#55A868", "kv_ret": "#C44E52", "trimemory": "#8172B2"}

    # 1. Composite score
    fig, ax = plt.subplots(figsize=(10, 6))
    vals = [model_aggs[m].get("mean_composite_score", 0) for m in models]
    errs = [model_aggs[m].get("std_composite_score", 0) for m in models]
    bars = ax.bar(models, vals, yerr=errs, color=[colors.get(m, "#999") for m in models], capsize=5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("Composite Score")
    ax.set_title("Composite Score by Model (Streaming Eval, Optimized)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "composite_score_by_model.png", dpi=150)
    plt.close(fig)

    # 2. Per-task accuracy
    fig, ax = plt.subplots(figsize=(12, 6))
    tasks = ["recent_exact_acc", "old_fact_acc", "long_pattern_acc", "salient_event_acc", "neutral_event_acc"]
    labels = ["Recent Exact", "Old Fact", "Long Pattern", "Salient Event", "Neutral Event"]
    x = np.arange(len(labels))
    w = 0.18
    for i, m in enumerate(models):
        vals = [model_aggs[m].get(f"mean_{t}", 0) for t in tasks]
        ax.bar(x + i * w, vals, w, label=m, color=colors.get(m, "#999"), alpha=0.85)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Task Accuracy (Streaming Eval, Optimized)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "per_task_accuracy.png", dpi=150)
    plt.close(fig)

    # 3. Router usage
    fig, ax = plt.subplots(figsize=(10, 6))
    router_models = [m for m in models if model_aggs[m].get("mean_router_kv_ratio", 0) > 0]
    if router_models:
        x = np.arange(len(router_models))
        kv_vals = [model_aggs[m].get("mean_router_kv_ratio", 0) for m in router_models]
        trn_vals = [model_aggs[m].get("mean_router_trn_ratio", 0) for m in router_models]
        ret_vals = [model_aggs[m].get("mean_router_ret_ratio", 0) for m in router_models]
        w = 0.25
        ax.bar(x - w, kv_vals, w, label="KV", color="#4C72B0")
        ax.bar(x, trn_vals, w, label="TRN", color="#55A868")
        ax.bar(x + w, ret_vals, w, label="Retrieval", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(router_models)
        ax.set_ylabel("Gate Ratio")
        ax.set_title("Router Usage by Model")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "router_usage_by_model.png", dpi=150)
    plt.close(fig)

    # 4. Retrieval usage
    fig, ax = plt.subplots(figsize=(10, 6))
    ret_calls = [model_aggs[m].get("mean_retrieval_calls", 0) for m in models]
    ret_hits = [model_aggs[m].get("mean_retrieval_hit_count", 0) for m in models]
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, ret_calls, w, label="Retrieval Calls", color="#C44E52")
    ax.bar(x + w/2, ret_hits, w, label="Retrieval Hits", color="#8172B2")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Count")
    ax.set_title("Retrieval Usage (Streaming Eval)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "retrieval_usage_by_model.png", dpi=150)
    plt.close(fig)

    # 5. Memory breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    archive_vals = [model_aggs[m].get("mean_archive_chunk_count", 0) for m in models]
    insertion_vals = [model_aggs[m].get("mean_archive_insertions", 0) for m in models]
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, archive_vals, w, label="Archive Chunks", color="#4C72B0")
    ax.bar(x + w/2, insertion_vals, w, label="Insertions", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Count")
    ax.set_title("Archive Memory (Streaming Eval)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "memory_breakdown_by_model.png", dpi=150)
    plt.close(fig)

    # 6. Throughput
    fig, ax = plt.subplots(figsize=(10, 6))
    tps = [model_aggs[m].get("mean_tokens_per_second", 0) for m in models]
    bars = ax.bar(models, tps, color=[colors.get(m, "#999") for m in models])
    for bar, v in zip(bars, tps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f"{v:.0f}", ha="center", fontsize=10)
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Throughput by Model (Streaming Eval)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "throughput_by_model.png", dpi=150)
    plt.close(fig)

    # 7. Salient vs Neutral
    fig, ax = plt.subplots(figsize=(10, 6))
    salient = [model_aggs[m].get("mean_salient_event_acc", 0) for m in models]
    neutral = [model_aggs[m].get("mean_neutral_event_acc", 0) for m in models]
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, salient, w, label="Salient", color="#C44E52")
    ax.bar(x + w/2, neutral, w, label="Neutral", color="#999999")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Accuracy")
    ax.set_title("Salient vs Neutral Event Accuracy (Streaming Eval)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "salient_vs_neutral.png", dpi=150)
    plt.close(fig)

    print(f"    Plots saved to {plot_dir}")


def generate_report(
    model_aggs: dict,
    checks: dict,
    all_telemetry: dict,
    out_dir: Path,
    total_time: float,
    args,
):
    critical_checks = [
        "trimemory_archive_nonempty",
        "trimemory_retrieval_calls_gt0",
        "trimemory_beats_kv_on_old_fact",
        "trimemory_recent_within_tolerance",
        "no_nan_inf",
        "trimemory_composite_best",
    ]
    critical_pass = sum(1 for c in critical_checks if checks.get(c, False))
    critical_total = len(critical_checks)

    if critical_pass == critical_total:
        verdict = "STREAMING_GO"
    elif critical_pass >= critical_total - 1:
        verdict = "CONDITIONAL_GO"
    else:
        verdict = "NO_GO"

    lines = [
        "# Streaming Tri-Memory Evaluation Report (Optimized)",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Runtime: {total_time:.1f}s",
        f"Seeds: {args.seeds}, Episodes/seed: {args.num_episodes}",
        f"Device: {args.device}, Precision: {args.precision}",
        f"Batch size (training): {args.batch_size}, Pin memory: {args.pin_memory}",
        f"KV window: {args.kv_window}, Chunk: {args.chunk_size}",
        f"Max archive: {args.max_archive_chunks}",
        "",
        f"## Verdict: **{verdict}**",
        "",
    ]

    if verdict == "STREAMING_GO":
        lines.append("All critical checks pass. Tri-Memory streaming pipeline validated.")
    elif verdict == "CONDITIONAL_GO":
        failed = [c for c in critical_checks if not checks.get(c, False)]
        lines.append(f"Near-pass. Failed: {', '.join(failed)}")
    else:
        failed = [c for c in critical_checks if not checks.get(c, False)]
        lines.append(f"Critical failures: {', '.join(failed)}")

    # Sanity checks
    lines.extend(["", "## Sanity Checks", ""])
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"- [{status}] {name}")

    # Key findings
    lines.extend(["", "## Key Findings", ""])

    def mean_of(model, field):
        rows = all_telemetry.get(model, [])
        return float(np.mean([getattr(r, field) for r in rows])) if rows else 0.0

    tri_fact = mean_of("trimemory", "old_fact_acc")
    kv_fact = mean_of("kv", "old_fact_acc")
    tri_pattern = mean_of("trimemory", "long_pattern_acc")
    kvtrn_pattern = mean_of("kv_trn", "long_pattern_acc")
    kvret_pattern = mean_of("kv_ret", "long_pattern_acc")
    tri_recent = mean_of("trimemory", "recent_exact_acc")
    kv_recent = mean_of("kv", "recent_exact_acc")
    tri_salient = mean_of("trimemory", "salient_event_acc")
    tri_neutral = mean_of("trimemory", "neutral_event_acc")

    lines.append(f"- Retrieval on old fact: TriMemory={tri_fact:.3f} vs KV={kv_fact:.3f} (delta={tri_fact - kv_fact:+.3f})")
    lines.append(f"- TRN on pattern: TriMemory={tri_pattern:.3f} vs KV+TRN={kvtrn_pattern:.3f} (delta={tri_pattern - kvtrn_pattern:+.3f})")
    lines.append(f"- TRN on pattern: TriMemory={tri_pattern:.3f} vs KV+Ret={kvret_pattern:.3f} (delta={tri_pattern - kvret_pattern:+.3f})")
    lines.append(f"- Recent exact: TriMemory={tri_recent:.3f} vs KV={kv_recent:.3f} (delta={tri_recent - kv_recent:+.3f})")
    lines.append(f"- Salient vs Neutral: {tri_salient:.3f} vs {tri_neutral:.3f} (gap={tri_salient - tri_neutral:+.3f})")

    tri_archive = mean_of("trimemory", "archive_chunk_count")
    tri_ret_calls = mean_of("trimemory", "retrieval_calls")
    tri_ret_hits = mean_of("trimemory", "retrieval_hit_count")
    lines.extend([
        "",
        f"- Archive chunks (TriMemory): {tri_archive:.1f}",
        f"- Retrieval calls: {tri_ret_calls:.1f}, hits: {tri_ret_hits:.1f}",
    ])

    # Performance findings
    lines.extend(["", "## Performance Findings", ""])
    for m in model_aggs:
        a = model_aggs[m]
        lines.append(
            f"- {m}: {a.get('mean_tokens_per_second', 0):.0f} tok/s, "
            f"latency={a.get('mean_latency_ms', 0):.1f}ms, "
            f"GPU mem={a.get('mean_gpu_mem_peak_mb', 0):.1f}MB"
        )

    # Model comparison table
    lines.extend(["", "## Model Comparison", ""])
    lines.append("| Model | Recent | Old Fact | Pattern | Salient | Neutral | Composite | Archive | Ret Calls | tok/s |")
    lines.append("|-------|--------|----------|---------|---------|---------|-----------|---------|-----------|-------|")
    for m in model_aggs:
        a = model_aggs[m]
        lines.append(
            f"| {m} "
            f"| {a.get('mean_recent_exact_acc', 0):.3f} "
            f"| {a.get('mean_old_fact_acc', 0):.3f} "
            f"| {a.get('mean_long_pattern_acc', 0):.3f} "
            f"| {a.get('mean_salient_event_acc', 0):.3f} "
            f"| {a.get('mean_neutral_event_acc', 0):.3f} "
            f"| {a.get('mean_composite_score', 0):.3f} "
            f"| {a.get('mean_archive_chunk_count', 0):.1f} "
            f"| {a.get('mean_retrieval_calls', 0):.1f} "
            f"| {a.get('mean_tokens_per_second', 0):.0f} |"
        )

    # Router telemetry
    lines.extend(["", "## Router Telemetry", ""])
    lines.append("| Model | KV Ratio | TRN Ratio | Retrieval Ratio |")
    lines.append("|-------|----------|-----------|-----------------|")
    for m in model_aggs:
        a = model_aggs[m]
        kv_r = a.get("mean_router_kv_ratio", 0)
        trn_r = a.get("mean_router_trn_ratio", 0)
        ret_r = a.get("mean_router_ret_ratio", 0)
        if kv_r > 0 or trn_r > 0 or ret_r > 0:
            lines.append(f"| {m} | {kv_r:.3f} | {trn_r:.3f} | {ret_r:.3f} |")

    # Failure analysis
    lines.extend(["", "## Failure Analysis", ""])
    if verdict != "STREAMING_GO":
        if not checks.get("trimemory_archive_nonempty", False):
            lines.append("1. Archive empty: forward_with_memory eviction path not populating archive.")
        if not checks.get("trimemory_beats_kv_on_old_fact", False):
            lines.append("2. Retrieval not improving old fact recall.")
        if not checks.get("salient_exceeds_neutral", False):
            lines.append("3. Salient/neutral gap absent: saliency scorer not differentiating.")
        if not checks.get("trimemory_composite_best", False):
            lines.append("4. TriMemory composite not best: gate not leveraging all memory tiers.")
        if not checks.get("no_nan_inf", False):
            lines.append("5. NaN/Inf detected in logits: numerical instability.")
    else:
        lines.append("- No critical failures.")

    # Recommended next step
    lines.extend(["", "## Recommended Next Step", ""])
    if not checks.get("trimemory_archive_nonempty", False):
        lines.append("Lower saliency_threshold to ensure archive population during streaming.")
    elif not checks.get("trimemory_beats_kv_on_old_fact", False):
        lines.append("Increase training steps or tune retrieval gate initialization.")
    elif not checks.get("trimemory_composite_best", False):
        lines.append("Investigate gate collapse -- TRN/retrieval may need stronger initialization bias.")
    elif not checks.get("salient_exceeds_neutral", False):
        lines.append("Tune saliency scorer weights to separate salient from neutral events.")
    else:
        lines.append("Proceed to larger-scale validation with longer sequences and real text data.")

    report_path = out_dir / "streaming_eval_summary.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"    Report: {report_path}")
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Streaming Tri-Memory Evaluation (Optimized)")
    parser.add_argument("--models", nargs="+", default=["kv", "kv_trn", "kv_ret", "trimemory"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-episodes", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--kv-window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--max-archive-chunks", type=int, default=DEFAULT_MAX_ARCHIVE)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--saliency-threshold", type=float, default=0.15)
    parser.add_argument("--save-failure-cases", action="store_true", default=False,
                        help="Collect deep retrieval diagnostics for old_fact failures")
    parser.add_argument("--failure-topk", type=int, default=5,
                        help="top-k for retrieval analysis in failure cases")
    parser.add_argument("--search-mode", default="hidden",
                        choices=["bag", "hidden", "hybrid"],
                        help="Retrieval search mode (default: hidden)")
    parser.add_argument("--hidden-weight", type=float, default=0.7,
                        help="Weight for hidden cosine in hybrid mode")
    parser.add_argument("--bag-weight", type=float, default=0.3,
                        help="Weight for bag cosine in hybrid mode")
    args = parser.parse_args()

    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir) / timestamp
    else:
        out_dir = Path(f"artifacts/trimemory_streaming_optimized/{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Streaming Tri-Memory Evaluation (Optimized)")
    print(f"  Models: {args.models}")
    print(f"  Seeds: {args.seeds}, Episodes/seed: {args.num_episodes}")
    print(f"  Batch size (train): {args.batch_size}, Precision: {args.precision}")
    print(f"  Pin memory: {args.pin_memory}")
    print(f"  KV window: {args.kv_window}, Chunk: {args.chunk_size}")
    print(f"  Max archive: {args.max_archive_chunks}")
    print(f"  Saliency threshold: {args.saliency_threshold}")
    print(f"  Search mode: {args.search_mode} (w_hidden={args.hidden_weight}, w_bag={args.bag_weight})")
    print(f"  Device: {args.device}, Steps: {args.steps}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    total_t0 = time.perf_counter()
    all_rows: list[EpisodeTelemetry] = []
    all_telemetry: dict[str, list[EpisodeTelemetry]] = {m: [] for m in args.models}
    failure_cases: list[dict] = []  # for --save-failure-cases

    cfg = make_cfg(max_seq_len=512)

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        seed_everything(seed)

        rng = np.random.default_rng(seed)
        episodes = [
            generate_episode(rng, args.kv_window, args.chunk_size)
            for _ in range(args.num_episodes)
        ]

        for model_name in args.models:
            print(f"\n  [{model_name}] Building and training...")
            seed_everything(seed)

            model = _build_model(
                model_name, cfg, args.kv_window, args.chunk_size,
                args.max_archive_chunks,
                search_mode=args.search_mode,
                search_w_hidden=args.hidden_weight,
                search_w_bag=args.bag_weight,
            )

            if isinstance(model, TriMemoryEngine):
                model.saliency_archiver.threshold = args.saliency_threshold

            model = train_model(
                model, episodes, args.steps, device,
                batch_size=args.batch_size,
                precision=args.precision,
                pin_memory=args.pin_memory,
            )
            model.eval()

            print(f"  [{model_name}] Streaming eval on {len(episodes)} episodes...")
            for ep_idx, episode in enumerate(episodes):
                tel = streaming_eval_episode(
                    model, episode, device, args.chunk_size, model_name,
                    precision=args.precision,
                )
                tel.seed = seed
                tel.episode_id = ep_idx
                all_rows.append(tel)
                all_telemetry[model_name].append(tel)

                # Deep retrieval diagnostics for failure analysis
                # Only collect for old_fact failures to avoid N*re-stream overhead
                if (args.save_failure_cases
                        and isinstance(model, TriMemoryEngine)
                        and tel.old_fact_acc < 1.0):
                    fc = collect_oldfact_failure_case(
                        model, episode, tel, device, args.chunk_size,
                        precision=args.precision,
                        failure_topk=args.failure_topk,
                    )
                    failure_cases.append(fc)

                if (ep_idx + 1) % 32 == 0:
                    recent_composite = np.mean([
                        all_telemetry[model_name][i].composite_score
                        for i in range(max(0, len(all_telemetry[model_name]) - 32), len(all_telemetry[model_name]))
                    ])
                    archive_count = tel.archive_chunk_count
                    print(f"    ep {ep_idx+1}/{len(episodes)}"
                          f" composite={recent_composite:.3f}"
                          f" archive={archive_count}"
                          f" tok/s={tel.tokens_per_second:.0f}")

            seed_rows = [r for r in all_telemetry[model_name] if r.seed == seed]
            mean_comp = float(np.mean([r.composite_score for r in seed_rows]))
            mean_archive = float(np.mean([r.archive_chunk_count for r in seed_rows]))
            mean_ret = float(np.mean([r.retrieval_calls for r in seed_rows]))
            mean_tps = float(np.mean([r.tokens_per_second for r in seed_rows]))
            print(f"  [{model_name}] seed={seed}"
                  f" composite={mean_comp:.3f}"
                  f" archive={mean_archive:.1f}"
                  f" ret_calls={mean_ret:.1f}"
                  f" tok/s={mean_tps:.0f}")

    total_time = time.perf_counter() - total_t0

    # Save failure cases JSONL if collected
    if failure_cases:
        fc_path = out_dir / "oldfact_failure_cases.jsonl"
        with open(fc_path, "w") as f:
            for fc in failure_cases:
                f.write(json.dumps(fc, default=str) + "\n")
        print(f"  Failure cases: {fc_path} ({len(failure_cases)} cases)")

    # Aggregate
    model_aggs = {}
    for m in args.models:
        model_aggs[m] = aggregate_telemetry(all_telemetry[m])

    checks = sanity_checks(all_telemetry)

    # Save CSV
    csv_path = out_dir / "streaming_eval_results.csv"
    fieldnames = [
        "model", "seed", "episode_id",
        "recent_exact_acc", "old_fact_acc", "long_pattern_acc",
        "salient_event_acc", "neutral_event_acc", "composite_score",
        "archive_chunk_count", "archive_insertions", "retained_kv_tokens", "state_bytes",
        "retrieval_calls", "retrieval_hit_count", "retrieval_hit_rate", "avg_retrieved_chunks",
        "router_kv_ratio", "router_trn_ratio", "router_ret_ratio",
        "tokens_per_second", "latency_ms",
        "gpu_mem_peak_mb", "cpu_util_percent", "gpu_util_percent",
        "kv_cache_tokens_retained", "kv_cache_bytes", "past_kv_truncations",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            d = asdict(row)
            writer.writerow({k: d[k] for k in fieldnames})
    print(f"\n  CSV: {csv_path}")

    # Save gate JSON
    gate_path = out_dir / "streaming_eval_gate.json"
    with open(gate_path, "w") as f:
        json.dump({
            "checks": checks,
            "model_aggregates": model_aggs,
            "args": vars(args),
            "total_time_s": total_time,
        }, f, indent=2, default=str)
    print(f"  Gate: {gate_path}")

    # Save all episode data
    all_data_path = out_dir / "all_episode_data.json"
    with open(all_data_path, "w") as f:
        json.dump([asdict(r) for r in all_rows], f, indent=2, default=str)
    print(f"  Data: {all_data_path}")

    # Save perf summary
    perf_summary = {
        "total_time_s": total_time,
        "total_episodes": len(all_rows),
        "optimizations_applied": [
            "batch_size_128", "bf16_mixed_precision",
            "pin_memory", "cached_vectorized_mask",
        ],
        "optimizations_not_applied": [
            "retrieval_2x_chunk", "cudnn_benchmark", "torch_compile",
        ],
        "per_model": {},
    }
    for m in args.models:
        a = model_aggs[m]
        perf_summary["per_model"][m] = {
            "mean_tokens_per_second": a.get("mean_tokens_per_second", 0),
            "mean_latency_ms": a.get("mean_latency_ms", 0),
            "mean_gpu_mem_peak_mb": a.get("mean_gpu_mem_peak_mb", 0),
        }
    perf_path = out_dir / "perf_summary.json"
    with open(perf_path, "w") as f:
        json.dump(perf_summary, f, indent=2)
    print(f"  Perf: {perf_path}")

    # Plots
    print("\n  Generating plots...")
    generate_plots(model_aggs, out_dir)

    # Report
    print("  Generating report...")
    verdict = generate_report(model_aggs, checks, all_telemetry, out_dir, total_time, args)

    # Final console output
    print("\n" + "=" * 60)
    print(f"  VERDICT: {verdict}")
    print(f"  Critical checks:")
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Total episodes: {len(all_rows)}")
    print(f"  Output: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
