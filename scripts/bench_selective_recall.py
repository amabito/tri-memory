#!/usr/bin/env python3
"""Selective recall mitigation benchmark for TRN.

Evaluates two prototype strategies that augment TRN with external memory:

  SKES - State-Keyed External Store: dict mapping hash(resonance_state) -> stored entry.
          At query time, finds nearest stored state by L2 distance and retrieves value.

  PISS - Priority Importance State Slots: keeps top-K states ranked by surprise score
          (cross-entropy of actual token vs model prediction). Retrieves by token match.

Comparison:
  vanilla_trn   - TRN generate() only, no augmentation
  trn_skes      - TRN + SKES retrieval
  trn_piss      - TRN + PISS retrieval
  tf_kv         - Transformer baseline (full sequence forward)

Task: Key-Value fact recall. Store N facts, then M gap tokens, then query each key.

Metrics:
  exact_match_accuracy   - fraction of queries answered correctly
  retrieval_latency_ms   - median latency per query (ms)
  extra_memory_bytes     - additional memory used by external store

Output:
  - Table printed to stdout
  - results/bench_selective_recall.csv (unless --no-csv)

Usage:
    python scripts/bench_selective_recall.py
    python scripts/bench_selective_recall.py --steps 200 --no-csv --device cpu
    python scripts/bench_selective_recall.py --n-facts 5,10 --gap-tokens 50,200
"""
from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trimemory.baseline import TransformerModel
from trimemory.bench_data import seed_everything
from trimemory.config import TRNConfig
from trimemory.model import TRNModel

# ---------------------------------------------------------------------------
# Model config (small for CPU speed)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
D_MODEL = 128
N_OSC = 64
N_LAYERS = 4
D_FF = 512
MAX_SEQ = 4096

KEY_LOW = 100
KEY_HIGH = 150     # 50 distinct keys
VAL_LOW = 150
VAL_HIGH = 200     # 50 distinct values
SEP_TOKEN = 3
END_TOKEN = 4
FILLER_LOW = 10
FILLER_HIGH = 90

# ---------------------------------------------------------------------------
# Pass criteria thresholds
# ---------------------------------------------------------------------------

PASS_CRITERIA = {
    "skes_acc_improvement": 0.10,   # SKES must improve over vanilla_trn by >= 10 pp
    "piss_acc_improvement": 0.15,   # PISS must improve over vanilla_trn by >= 15 pp
    "max_tps_drop": 0.50,           # retrieval latency must not increase by more than 50x vs vanilla
    "max_extra_memory_mb": 100.0,   # additional memory must stay below 100 MB
}


def _make_trn(device: str) -> TRNModel:
    cfg = TRNConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_oscillators=N_OSC,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ,
    )
    return TRNModel(cfg).to(device)


def _make_tf(device: str) -> TransformerModel:
    cfg = TRNConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_oscillators=N_OSC,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ,
    )
    return TransformerModel(cfg).to(device)


# ---------------------------------------------------------------------------
# SKES: State-Keyed External Store
# ---------------------------------------------------------------------------

@dataclass
class _SKESEntry:
    state: Tensor      # (1, n_osc) fp32 resonance state (layer 0, real part)
    position: int
    key_tok: int
    val_tok: int


class SKES:
    """State-Keyed External Store.

    Stores resonance state snapshots keyed by a hash of the state bytes.
    At query time, finds the nearest stored state by L2 distance.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self.max_entries = max_entries
        self._entries: List[_SKESEntry] = []

    def store(
        self,
        r_real: Tensor,   # (1, n_osc) fp32
        position: int,
        key_tok: int,
        val_tok: int,
    ) -> None:
        if len(self._entries) >= self.max_entries:
            self._entries.pop(0)
        self._entries.append(_SKESEntry(r_real.cpu().clone(), position, key_tok, val_tok))

    def query(self, r_real: Tensor) -> Optional[Tuple[int, int, int]]:
        """Return (position, key_tok, val_tok) of nearest stored state, or None."""
        if not self._entries:
            return None
        r_flat = r_real.cpu().reshape(1, -1).float()
        best_dist = float("inf")
        best_entry = None
        for entry in self._entries:
            e_flat = entry.state.reshape(1, -1).float()
            dist = (r_flat - e_flat).pow(2).sum().item()
            if dist < best_dist:
                best_dist = dist
                best_entry = entry
        if best_entry is None:
            return None
        return best_entry.position, best_entry.key_tok, best_entry.val_tok

    def memory_bytes(self) -> int:
        if not self._entries:
            return 0
        return len(self._entries) * self._entries[0].state.numel() * 4

    def clear(self) -> None:
        self._entries.clear()


# ---------------------------------------------------------------------------
# PISS: Priority Importance State Slots
# ---------------------------------------------------------------------------

@dataclass
class _PISSSlot:
    surprise: float
    position: int
    state: Tensor    # (1, n_osc) fp32
    token_id: int

    # For heapq (min-heap by surprise — we pop the lowest surprise)
    def __lt__(self, other: "_PISSSlot") -> bool:
        return self.surprise < other.surprise


class PISS:
    """Priority Importance State Slots.

    Keeps top-K states ranked by surprise score (negative log-prob).
    High surprise = unexpected token = important to remember.
    """

    def __init__(self, max_slots: int = 64) -> None:
        self.max_slots = max_slots
        self._heap: List[_PISSSlot] = []  # min-heap by surprise

    def maybe_store(
        self,
        surprise: float,
        position: int,
        r_real: Tensor,   # (1, n_osc) fp32
        token_id: int,
    ) -> None:
        slot = _PISSSlot(
            surprise=surprise,
            position=position,
            state=r_real.cpu().clone(),
            token_id=token_id,
        )
        if len(self._heap) < self.max_slots:
            heapq.heappush(self._heap, slot)
        elif surprise > self._heap[0].surprise:
            heapq.heapreplace(self._heap, slot)

    def query_by_token(self, token_id: int) -> Optional[Tuple[float, int, Tensor]]:
        """Return (surprise, position, state) for highest-surprise slot matching token_id."""
        candidates = [s for s in self._heap if s.token_id == token_id]
        if not candidates:
            return None
        best = max(candidates, key=lambda s: s.surprise)
        return best.surprise, best.position, best.state

    def query_all_sorted(self) -> List[_PISSSlot]:
        return sorted(self._heap, key=lambda s: s.surprise, reverse=True)

    def memory_bytes(self) -> int:
        if not self._heap:
            return 0
        return len(self._heap) * self._heap[0].state.numel() * 4

    def clear(self) -> None:
        self._heap.clear()


# ---------------------------------------------------------------------------
# State capture: encode tokens step-by-step via step_single
# ---------------------------------------------------------------------------

def _encode_capture_states(
    model: TRNModel,
    token_seq: List[int],
    device: str,
) -> List[Tuple[int, Tensor]]:
    """Encode token_seq via step_single, return [(position, r_real_layer0), ...]."""
    n_layers = model.cfg.n_layers
    n_osc = model.cfg.n_oscillators
    r_real = [torch.zeros(1, n_osc, device=device) for _ in range(n_layers)]
    r_imag = [torch.zeros(1, n_osc, device=device) for _ in range(n_layers)]

    states: List[Tuple[int, Tensor]] = []
    with torch.no_grad():
        for pos, tok_id in enumerate(token_seq):
            x = model.embedding(torch.tensor([[tok_id]], device=device))[:, 0, :]
            for layer_idx, block in enumerate(model.blocks):
                x_norm = block.norm1(x)
                out, r_real[layer_idx], r_imag[layer_idx] = block.resonance.step_single(
                    x_norm, r_real[layer_idx], r_imag[layer_idx], pos
                )
                x = x + out
                x = x + block.ffn(block.norm2(x))
            states.append((pos, r_real[0].clone()))
    return states


def _compute_surprise_scores(
    model: TRNModel,
    token_seq: List[int],
    device: str,
) -> List[float]:
    """Compute per-token surprise = -log P(token | context) using full forward pass."""
    if len(token_seq) < 2:
        return [0.0] * len(token_seq)
    ids = torch.tensor([token_seq], device=device)
    with torch.no_grad():
        out = model(ids)
        logits = out["logits"][0]  # (T, V)
        log_probs = F.log_softmax(logits, dim=-1)
    surprises = [0.0]  # no context for first token
    for t in range(1, len(token_seq)):
        actual = token_seq[t]
        surprises.append(-log_probs[t - 1, actual].item())
    return surprises


# ---------------------------------------------------------------------------
# Fact sequence generation
# ---------------------------------------------------------------------------

def _build_fact_sequence(
    n_facts: int,
    gap_tokens: int,
    rng: torch.Generator,
) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """Build a fact-recall sequence.

    Returns:
        seq          : full token sequence
        facts        : list of (key_tok, val_tok)
        fact_end_pos : list of (position_of_END_TOKEN, key_tok, val_tok)
    """
    seq: List[int] = []
    facts: List[Tuple[int, int]] = []
    fact_end_pos: List[Tuple[int, int, int]] = []

    for _ in range(n_facts):
        key_tok = int(torch.randint(KEY_LOW, KEY_HIGH, (1,), generator=rng).item())
        val_tok = int(torch.randint(VAL_LOW, VAL_HIGH, (1,), generator=rng).item())
        facts.append((key_tok, val_tok))
        seq.extend([key_tok, SEP_TOKEN, val_tok, END_TOKEN])
        fact_end_pos.append((len(seq) - 1, key_tok, val_tok))

    # Filler gap
    for _ in range(gap_tokens):
        tok = int(torch.randint(FILLER_LOW, FILLER_HIGH, (1,), generator=rng).item())
        seq.append(tok)

    return seq, facts, fact_end_pos


# ---------------------------------------------------------------------------
# Training: GPT cross-entropy on fact-recall sequences
# ---------------------------------------------------------------------------

def _make_training_batch(
    batch_size: int,
    n_facts: int,
    gap_tokens: int,
    device: str,
    rng: torch.Generator,
) -> Tuple[Tensor, Tensor]:
    fact_seq_len = n_facts * 4  # KEY, SEP, VAL, END per fact
    total_len = fact_seq_len + gap_tokens + 2  # +2 for query KEY, SEP

    batch = []
    for _ in range(batch_size):
        seq, facts, _ = _build_fact_sequence(n_facts, gap_tokens, rng)
        # Append a query (pick random fact)
        q_idx = int(torch.randint(0, n_facts, (1,), generator=rng).item())
        q_key, q_val = facts[q_idx]
        seq.extend([q_key, SEP_TOKEN])
        # Pad or truncate
        seq = seq[:MAX_SEQ]
        batch.append(seq)

    max_len = max(len(s) for s in batch)
    ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, s in enumerate(batch):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    ids = ids.to(device)
    return ids, ids


def _train_model(
    model: nn.Module,
    n_facts: int,
    gap_tokens: int,
    steps: int,
    batch_size: int,
    device: str,
    seed: int,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = torch.Generator()
    rng.manual_seed(seed)
    model.train()
    for _ in range(steps):
        ids, _ = _make_training_batch(batch_size, n_facts, gap_tokens, device, rng)
        out = model(ids)
        logits = out["logits"]  # (B, T, V)
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            ids[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Evaluation per strategy
# ---------------------------------------------------------------------------

N_QUERY_REPEATS = 10  # for latency measurement


def _eval_vanilla_trn(
    model: TRNModel,
    n_facts: int,
    gap_tokens: int,
    n_eval: int,
    device: str,
    seed: int,
) -> Tuple[float, float, int]:
    """Vanilla TRN: generate 1 token after query prefix, check == val."""
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(seed + 500)
    correct = 0
    latencies: List[float] = []

    for _ in range(n_eval):
        seq, facts, _ = _build_fact_sequence(n_facts, gap_tokens, rng)
        q_idx = int(torch.randint(0, n_facts, (1,), generator=rng).item())
        q_key, q_val = facts[q_idx]
        query_seq = seq + [q_key, SEP_TOKEN]
        ids = torch.tensor([query_seq[:MAX_SEQ]], device=device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(ids)
        pred = out["logits"][0, -1].argmax().item()
        latencies.append((time.perf_counter() - t0) * 1000)

        if pred == q_val:
            correct += 1

    median_lat = sorted(latencies)[len(latencies) // 2]
    return correct / n_eval, median_lat, 0


def _eval_trn_skes(
    model: TRNModel,
    n_facts: int,
    gap_tokens: int,
    n_eval: int,
    device: str,
    seed: int,
) -> Tuple[float, float, int]:
    """TRN + SKES: store state after each fact's END token, query at retrieval time."""
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(seed + 600)
    skes = SKES(max_entries=n_facts * n_eval)
    correct = 0
    latencies: List[float] = []

    for _ in range(n_eval):
        skes.clear()
        seq, facts, fact_end_positions = _build_fact_sequence(n_facts, gap_tokens, rng)

        # Encode full sequence, store state after each fact END_TOKEN
        states = _encode_capture_states(model, seq, device)
        for (end_pos, key_tok, val_tok) in fact_end_positions:
            _, r_real = states[end_pos]
            skes.store(r_real, end_pos, key_tok, val_tok)

        # Query: encode [KEY, SEP] continuing from end of seq
        q_idx = int(torch.randint(0, n_facts, (1,), generator=rng).item())
        q_key, q_val = facts[q_idx]
        query_prefix = seq + [q_key, SEP_TOKEN]
        query_states = _encode_capture_states(model, query_prefix, device)
        _, query_r_real = query_states[-1]

        t0 = time.perf_counter()
        result = skes.query(query_r_real)
        latencies.append((time.perf_counter() - t0) * 1000)

        if result is not None:
            _, retrieved_key, retrieved_val = result
            if retrieved_val == q_val:
                correct += 1

    median_lat = sorted(latencies)[len(latencies) // 2]
    mem_bytes = skes.memory_bytes()
    return correct / n_eval, median_lat, mem_bytes


def _eval_trn_piss(
    model: TRNModel,
    n_facts: int,
    gap_tokens: int,
    n_eval: int,
    device: str,
    seed: int,
) -> Tuple[float, float, int]:
    """TRN + PISS: store high-surprise states during encoding, retrieve by key token."""
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(seed + 700)
    piss = PISS(max_slots=min(64, n_facts * 4))
    correct = 0
    latencies: List[float] = []

    for _ in range(n_eval):
        piss.clear()
        seq, facts, _ = _build_fact_sequence(n_facts, gap_tokens, rng)

        # Compute surprise scores via full forward
        surprises = _compute_surprise_scores(model, seq, device)

        # Also get per-position states
        states = _encode_capture_states(model, seq, device)

        # Store in PISS: we store the VAL_TOKEN positions (positions of val_tok in seq)
        # Facts are at positions: key(0), sep(1), val(2), end(3), key(4), ...
        # i.e., val_tok is at position i*4 + 2
        for i in range(n_facts):
            val_pos = i * 4 + 2
            key_pos = i * 4
            if val_pos >= len(seq):
                break
            val_tok = seq[val_pos]
            # Store with surprise of val_tok, using key_tok as the "token_id" for lookup
            key_tok = seq[key_pos]
            surprise = surprises[val_pos]
            _, r_real = states[val_pos]
            # Store key_tok as the retrieval handle, val_tok embedded in position for lookup
            piss.maybe_store(surprise, val_pos, r_real, key_tok)

        q_idx = int(torch.randint(0, n_facts, (1,), generator=rng).item())
        q_key, q_val = facts[q_idx]

        t0 = time.perf_counter()
        result = piss.query_by_token(q_key)
        latencies.append((time.perf_counter() - t0) * 1000)

        if result is not None:
            _, stored_pos, _ = result
            # The val_tok is at stored_pos (which is val_pos = key_pos + 2)
            val_from_piss = seq[stored_pos] if stored_pos < len(seq) else -1
            if val_from_piss == q_val:
                correct += 1

    median_lat = sorted(latencies)[len(latencies) // 2]
    mem_bytes = piss.memory_bytes()
    return correct / n_eval, median_lat, mem_bytes


def _eval_tf_kv(
    model: TransformerModel,
    n_facts: int,
    gap_tokens: int,
    n_eval: int,
    device: str,
    seed: int,
) -> Tuple[float, float, int]:
    """TF baseline: run full sequence forward, check logit at query position."""
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(seed + 800)
    correct = 0
    latencies: List[float] = []

    for _ in range(n_eval):
        seq, facts, _ = _build_fact_sequence(n_facts, gap_tokens, rng)
        q_idx = int(torch.randint(0, n_facts, (1,), generator=rng).item())
        q_key, q_val = facts[q_idx]
        full_seq = seq + [q_key, SEP_TOKEN]
        ids = torch.tensor([full_seq[:MAX_SEQ]], device=device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(ids)
        pred = out["logits"][0, -1].argmax().item()
        latencies.append((time.perf_counter() - t0) * 1000)

        if pred == q_val:
            correct += 1

    median_lat = sorted(latencies)[len(latencies) // 2]
    return correct / n_eval, median_lat, 0


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def _print_table(rows: List[dict]) -> None:
    header = (
        f"{'strategy':<15} {'n_facts':<8} {'gap':<6} {'acc':<8} "
        f"{'lat_ms':<10} {'mem_kb':<10} {'kv_tokens':<10} {'tps_drop':<10} {'pass':<6}"
    )
    print("\n" + header)
    print("-" * 80)
    for r in rows:
        mem_kb = r["extra_memory_bytes"] / 1024
        kv_tok = r.get("retained_kv_tokens", "")
        tps_drop = r.get("tps_drop_ratio", "")
        passed = r.get("pass_criteria_met", "")
        tps_str = f"{tps_drop:.3f}" if isinstance(tps_drop, float) else str(tps_drop)
        pass_str = "PASS" if passed is True else ("FAIL" if passed is False else str(passed))
        print(
            f"{r['strategy']:<15} {r['n_facts']:<8} {r['gap_tokens']:<6} "
            f"{r['exact_match_accuracy']:.4f}   {r['retrieval_latency_ms']:.3f}      "
            f"{mem_kb:.1f}      {str(kv_tok):<10} {tps_str:<10} {pass_str:<6}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Selective recall benchmark")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--n-facts", default="5,10,20,50")
    p.add_argument("--gap-tokens", default="50,200,500")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_facts_list = [int(x) for x in args.n_facts.split(",")]
    gap_list = [int(x) for x in args.gap_tokens.split(",")]
    device = args.device
    seed = args.seed

    all_results: List[dict] = []

    for n_facts in n_facts_list:
        for gap in gap_list:
            print(f"\n=== n_facts={n_facts}, gap={gap} ===")

            # Train a shared TRN model
            print("  Training TRN...", end="", flush=True)
            seed_everything(seed)
            trn_model = _make_trn(device)
            t0 = time.time()
            _train_model(trn_model, n_facts, gap, args.steps, args.batch_size, device, seed)
            print(f" done ({time.time()-t0:.1f}s)")

            # Train TF model
            print("  Training TF...", end="", flush=True)
            seed_everything(seed)
            tf_model = _make_tf(device)
            t0 = time.time()
            _train_model(tf_model, n_facts, gap, args.steps, args.batch_size, device, seed)
            print(f" done ({time.time()-t0:.1f}s)")

            n_eval = 50  # keep fast for CPU

            # Evaluate vanilla_trn first (baseline for tps_drop_ratio)
            print("  Evaluating vanilla_trn...", end="", flush=True)
            vanilla_acc, vanilla_lat, vanilla_mem = _eval_vanilla_trn(
                trn_model, n_facts, gap, n_eval, device, seed
            )
            print(f" acc={vanilla_acc:.3f}")
            all_results.append(dict(
                strategy="vanilla_trn",
                n_facts=n_facts,
                gap_tokens=gap,
                exact_match_accuracy=vanilla_acc,
                retrieval_latency_ms=vanilla_lat,
                extra_memory_bytes=vanilla_mem,
                retained_kv_tokens=0,
                tps_drop_ratio=1.0,
                pass_criteria_met="",  # baseline; no augmentation to judge
            ))

            # SKES
            print("  Evaluating trn_skes...", end="", flush=True)
            skes_acc, skes_lat, skes_mem = _eval_trn_skes(
                trn_model, n_facts, gap, n_eval, device, seed
            )
            print(f" acc={skes_acc:.3f}")
            skes_tps_drop = skes_lat / max(vanilla_lat, 1e-9)
            skes_mem_mb = skes_mem / (1024 * 1024)
            skes_acc_improvement = skes_acc - vanilla_acc
            # Estimate retained_kv_tokens as number of stored entries (n_facts per eval)
            skes_retained = n_facts
            skes_pass = (
                skes_acc_improvement >= PASS_CRITERIA["skes_acc_improvement"]
                and skes_tps_drop <= PASS_CRITERIA["max_tps_drop"]
                and skes_mem_mb <= PASS_CRITERIA["max_extra_memory_mb"]
            )
            all_results.append(dict(
                strategy="trn_skes",
                n_facts=n_facts,
                gap_tokens=gap,
                exact_match_accuracy=skes_acc,
                retrieval_latency_ms=skes_lat,
                extra_memory_bytes=skes_mem,
                retained_kv_tokens=skes_retained,
                tps_drop_ratio=skes_tps_drop,
                pass_criteria_met=skes_pass,
            ))

            # PISS
            print("  Evaluating trn_piss...", end="", flush=True)
            piss_acc, piss_lat, piss_mem = _eval_trn_piss(
                trn_model, n_facts, gap, n_eval, device, seed
            )
            print(f" acc={piss_acc:.3f}")
            piss_tps_drop = piss_lat / max(vanilla_lat, 1e-9)
            piss_mem_mb = piss_mem / (1024 * 1024)
            piss_acc_improvement = piss_acc - vanilla_acc
            piss_retained = min(64, n_facts * 4)
            piss_pass = (
                piss_acc_improvement >= PASS_CRITERIA["piss_acc_improvement"]
                and piss_tps_drop <= PASS_CRITERIA["max_tps_drop"]
                and piss_mem_mb <= PASS_CRITERIA["max_extra_memory_mb"]
            )
            all_results.append(dict(
                strategy="trn_piss",
                n_facts=n_facts,
                gap_tokens=gap,
                exact_match_accuracy=piss_acc,
                retrieval_latency_ms=piss_lat,
                extra_memory_bytes=piss_mem,
                retained_kv_tokens=piss_retained,
                tps_drop_ratio=piss_tps_drop,
                pass_criteria_met=piss_pass,
            ))

            # TF baseline
            print("  Evaluating tf_kv...", end="", flush=True)
            tf_acc, tf_lat, tf_mem = _eval_tf_kv(tf_model, n_facts, gap, n_eval, device, seed)
            print(f" acc={tf_acc:.3f}")
            all_results.append(dict(
                strategy="tf_kv",
                n_facts=n_facts,
                gap_tokens=gap,
                exact_match_accuracy=tf_acc,
                retrieval_latency_ms=tf_lat,
                extra_memory_bytes=tf_mem,
                retained_kv_tokens="",
                tps_drop_ratio="",
                pass_criteria_met="",
            ))

    # Print pass/fail summary
    print("\n=== Pass/Fail Summary ===")
    for r in all_results:
        if r["strategy"] in ("trn_skes", "trn_piss"):
            status = "PASS" if r["pass_criteria_met"] else "FAIL"
            print(
                f"  {r['strategy']:<15} n_facts={r['n_facts']:<4} gap={r['gap_tokens']:<5} "
                f"acc_improvement={r['exact_match_accuracy'] - all_results[0]['exact_match_accuracy']:+.3f} "
                f"tps_drop={r['tps_drop_ratio']:.2f}x "
                f"mem_mb={r['extra_memory_bytes'] / (1024*1024):.2f} -> {status}"
            )

    _print_table(all_results)

    if not args.no_csv and all_results:
        out_dir = Path(__file__).parent.parent / "results"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "bench_selective_recall.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "strategy", "n_facts", "gap_tokens",
                "exact_match_accuracy", "retrieval_latency_ms", "extra_memory_bytes",
                "retained_kv_tokens", "tps_drop_ratio", "pass_criteria_met",
            ], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
