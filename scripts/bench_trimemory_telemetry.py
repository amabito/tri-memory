#!/usr/bin/env python3
"""Agent Telemetry Benchmark: commercial use-case validation.

Simulates agent telemetry streams with:
  - cpu_pct, mem_pct (quantized to token range)
  - error_count spikes
  - tool boundaries
  - regime shifts
  - intermittent incidents (old exact facts)

Tasks:
  1. trend_direction: predict if next metric goes up/down/flat
  2. anomaly_tendency: detect anomaly regime
  3. old_incident_lookup: recall specific old incident value
  4. recent_state: predict current metric value

Comparisons: A/B/C/D (KV only / KV+TRN / KV+Ret / KV+TRN+Ret)

Output:
  artifacts/trimemory/{timestamp}/telemetry_results.json
  artifacts/trimemory/{timestamp}/telemetry_summary.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_src = os.path.join(os.path.dirname(__file__), "..", "src")
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _root]
sys.path.insert(0, _src)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from trn.bench_data import seed_everything
from trn.config import TRNConfig
from trn.integrations.vllm_backend import DualMemoryEngine
from trn.tri_memory import TriMemoryEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE = 64
CHUNK_SIZE = 32
D_MODEL = 128
N_LAYERS = 4
N_OSC = 64
D_FF = 512
VOCAB_SIZE = 256
SEQ_LEN = 320
TRAIN_STEPS = 300
BATCH_SIZE = 16

# Token ranges for telemetry encoding
METRIC_LOW = 10
METRIC_HIGH = 110     # 100 bins for metric values
INCIDENT_LOW = 200
INCIDENT_HIGH = 240
TOOL_BOUNDARY = 5
TREND_UP = 240
TREND_DOWN = 241
TREND_FLAT = 242
ANOMALY_YES = 243
ANOMALY_NO = 244
QUERY_TREND = 6
QUERY_ANOMALY = 7
QUERY_INCIDENT = 8
QUERY_RECENT = 9


# ---------------------------------------------------------------------------
# Telemetry Dataset
# ---------------------------------------------------------------------------

class TelemetryDataset(Dataset):
    """Synthetic agent telemetry stream.

    Generates sequences with:
      - Smooth metric values (sine + noise + drift)
      - Regime shifts (sudden level changes)
      - Old incidents (specific tokens placed early)
      - Queries for trend, anomaly, old incident, recent value
    """

    def __init__(
        self,
        n_samples: int = 2000,
        seq_len: int = SEQ_LEN,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_samples

    def _quantize_metric(self, value: float) -> int:
        """Map a float [0, 1] to token range [METRIC_LOW, METRIC_HIGH)."""
        v = max(0.0, min(1.0, value))
        return METRIC_LOW + int(v * (METRIC_HIGH - METRIC_LOW - 1))

    def __getitem__(self, idx: int) -> dict:
        T = self.seq_len
        seq = np.zeros(T, dtype=np.int64)

        # Generate metric stream
        t = np.arange(T - 10, dtype=np.float64)  # leave room for queries
        n_metric = len(t)

        # Base signal: sine + linear drift + noise
        freq = self.rng.uniform(0.01, 0.05)
        drift = self.rng.uniform(-0.001, 0.001)
        noise_std = self.rng.uniform(0.02, 0.08)
        base = 0.5 + 0.3 * np.sin(2 * np.pi * freq * t) + drift * t
        base += self.rng.normal(0, noise_std, n_metric)
        base = np.clip(base, 0, 1)

        # Regime shift at random position
        shift_pos = self.rng.integers(n_metric // 3, 2 * n_metric // 3)
        shift_amount = self.rng.uniform(-0.2, 0.2)
        base[shift_pos:] += shift_amount
        base = np.clip(base, 0, 1)

        # Place incident (old exact fact) early in sequence
        incident_pos = self.rng.integers(5, 20)
        incident_val = int(self.rng.integers(INCIDENT_LOW, INCIDENT_HIGH))
        seq[incident_pos] = incident_val

        # Tool boundary
        tool_pos = self.rng.integers(40, 80)
        seq[tool_pos] = TOOL_BOUNDARY

        # Fill metric values
        for i in range(n_metric):
            if seq[i] == 0:  # don't overwrite special tokens
                seq[i] = self._quantize_metric(base[i])

        # Determine ground truth
        # Trend: compare last 5 metrics vs previous 5
        recent_mean = base[n_metric - 5:n_metric].mean()
        prev_mean = base[n_metric - 10:n_metric - 5].mean()
        diff = recent_mean - prev_mean
        if diff > 0.05:
            trend_answer = TREND_UP
        elif diff < -0.05:
            trend_answer = TREND_DOWN
        else:
            trend_answer = TREND_FLAT

        # Anomaly: is current regime shifted?
        is_anomaly = abs(shift_amount) > 0.1 and shift_pos < n_metric - 20
        anomaly_answer = ANOMALY_YES if is_anomaly else ANOMALY_NO

        # Recent value: the last quantized metric
        recent_answer = self._quantize_metric(base[n_metric - 1])

        # Place queries at end
        q_start = T - 10
        seq[q_start] = QUERY_TREND
        seq[q_start + 1] = trend_answer
        seq[q_start + 2] = QUERY_ANOMALY
        seq[q_start + 3] = anomaly_answer
        seq[q_start + 4] = QUERY_INCIDENT
        seq[q_start + 5] = incident_val
        seq[q_start + 6] = QUERY_RECENT
        seq[q_start + 7] = recent_answer
        # Padding
        seq[q_start + 8] = METRIC_LOW
        seq[q_start + 9] = METRIC_LOW

        return {
            "input_ids": torch.from_numpy(seq),
            "trend_answer": trend_answer,
            "anomaly_answer": anomaly_answer,
            "incident_val": incident_val,
            "recent_answer": recent_answer,
        }


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_model(model, dataset, steps, device, lr=3e-4):
    model = model.to(device).train()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader_it = iter(loader)
    records = []
    for step in range(1, steps + 1):
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            batch = next(loader_it)
        ids = batch["input_ids"].to(device)
        out = model(ids, labels=ids)
        loss = out["loss"]
        if torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if step % 50 == 0:
            records.append({"step": step, "loss": loss.item()})
    return records


def evaluate_model(model, dataset, device, n_eval=200):
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=min(32, n_eval), shuffle=False)

    trend_ok = anomaly_ok = incident_ok = recent_ok = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if total >= n_eval:
                break
            ids = batch["input_ids"].to(device)
            B = ids.size(0)
            logits = model(ids)["logits"]
            q_start = dataset.seq_len - 10

            trend_preds = logits[:, q_start, :].argmax(-1).cpu()
            trend_ok += (trend_preds == batch["trend_answer"]).sum().item()

            anomaly_preds = logits[:, q_start + 2, :].argmax(-1).cpu()
            anomaly_ok += (anomaly_preds == batch["anomaly_answer"]).sum().item()

            incident_preds = logits[:, q_start + 4, :].argmax(-1).cpu()
            incident_ok += (incident_preds == batch["incident_val"]).sum().item()

            recent_preds = logits[:, q_start + 6, :].argmax(-1).cpu()
            recent_ok += (recent_preds == batch["recent_answer"]).sum().item()

            total += B

    trend_acc = trend_ok / max(total, 1)
    anomaly_acc = anomaly_ok / max(total, 1)
    incident_acc = incident_ok / max(total, 1)
    recent_acc = recent_ok / max(total, 1)

    accs = [trend_acc, anomaly_acc, incident_acc, recent_acc]
    composite = len(accs) / sum(1 / max(a, 1e-6) for a in accs) if any(a > 0 for a in accs) else 0.0

    return {
        "trend_direction_acc": trend_acc,
        "anomaly_detect_acc": anomaly_acc,
        "old_incident_lookup_acc": incident_acc,
        "recent_state_acc": recent_acc,
        "composite_score": composite,
        "n_eval": total,
    }


def make_cfg():
    return TRNConfig(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_oscillators=N_OSC,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=SEQ_LEN + 16,
    )


def build_models(cfg):
    return {
        "A_kv_only": DualMemoryEngine(cfg, window_size=WINDOW_SIZE),
        "B_kv_trn": DualMemoryEngine(cfg, window_size=WINDOW_SIZE),
        "C_kv_ret": TriMemoryEngine(cfg, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE),
        "D_kv_trn_ret": TriMemoryEngine(cfg, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE),
    }


# ---------------------------------------------------------------------------
# Analytical scaling estimate
# ---------------------------------------------------------------------------

def analytical_scaling(cfg, n_agents_list):
    """Estimate memory per agent for different architectures."""
    n_heads = max(1, cfg.d_model // 64)
    head_dim = cfg.d_model // n_heads
    trn_state = cfg.n_layers * cfg.n_oscillators * 2 * 4  # fp32
    kv_per_token = cfg.n_layers * n_heads * head_dim * 2 * 2  # fp16 k+v

    rows = []
    for n_agents in n_agents_list:
        # Full KV (no window)
        full_kv_mb = n_agents * kv_per_token * 10000 / (1024 * 1024)

        # KV window only
        kv_window_mb = n_agents * kv_per_token * WINDOW_SIZE / (1024 * 1024)

        # KV+TRN
        kv_trn_mb = n_agents * (kv_per_token * WINDOW_SIZE + trn_state) / (1024 * 1024)

        # TriMemory (KV+TRN+Ret)
        ret_per_agent = 256 * (cfg.d_model * 4 + cfg.vocab_size * 4 + 32 * 4)  # max chunks
        tri_mb = n_agents * (kv_per_token * WINDOW_SIZE + trn_state + ret_per_agent) / (1024 * 1024)

        rows.append({
            "n_agents": n_agents,
            "full_kv_mb": full_kv_mb,
            "kv_window_mb": kv_window_mb,
            "kv_trn_mb": kv_trn_mb,
            "tri_memory_mb": tri_mb,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    seed_everything(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"artifacts/trimemory/{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = make_cfg()
    dataset = TelemetryDataset(n_samples=2000, seq_len=SEQ_LEN, seed=args.seed)

    print(f"[Telemetry Benchmark] device={args.device} steps={args.steps}")
    results = {}
    models = build_models(cfg)

    for name, model in models.items():
        print(f"  Training {name}...")
        seed_everything(args.seed)
        t0 = time.perf_counter()
        loss_curve = train_model(model, dataset, args.steps, device)
        train_time = time.perf_counter() - t0
        eval_result = evaluate_model(model, dataset, device)
        results[name] = {**eval_result, "train_time_s": train_time, "loss_curve": loss_curve}
        print(f"    trend={eval_result['trend_direction_acc']:.3f}"
              f" anomaly={eval_result['anomaly_detect_acc']:.3f}"
              f" incident={eval_result['old_incident_lookup_acc']:.3f}"
              f" recent={eval_result['recent_state_acc']:.3f}"
              f" composite={eval_result['composite_score']:.3f}")

    # Analytical scaling
    scaling = analytical_scaling(cfg, [1, 10, 100, 1000, 10000])

    all_results = {"benchmark": results, "scaling": scaling}
    with open(out_dir / "telemetry_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    md_lines = [
        "# Agent Telemetry Benchmark Results",
        "",
        f"Timestamp: {timestamp}",
        "",
        "## 4-Way Comparison",
        "",
        "| Model | Trend | Anomaly | Incident | Recent | Composite |",
        "|-------|-------|---------|----------|--------|-----------|",
    ]
    for name, r in results.items():
        md_lines.append(
            f"| {name} | {r['trend_direction_acc']:.3f} | {r['anomaly_detect_acc']:.3f}"
            f" | {r['old_incident_lookup_acc']:.3f} | {r['recent_state_acc']:.3f}"
            f" | {r['composite_score']:.3f} |"
        )
    md_lines.extend([
        "",
        "## Agent Scaling (Analytical)",
        "",
        "| Agents | Full KV (MB) | KV Window (MB) | KV+TRN (MB) | Tri-Memory (MB) |",
        "|--------|-------------|----------------|-------------|-----------------|",
    ])
    for s in scaling:
        md_lines.append(
            f"| {s['n_agents']:>6} | {s['full_kv_mb']:.1f} | {s['kv_window_mb']:.1f}"
            f" | {s['kv_trn_mb']:.1f} | {s['tri_memory_mb']:.1f} |"
        )

    with open(out_dir / "telemetry_summary.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n[DONE] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
