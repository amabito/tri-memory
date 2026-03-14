#!/usr/bin/env python3
"""TRN Deep Diagnosis: branch-level probe at gate/residual/FFN junction.

Runs D-only (trimemory) for 500 steps on two contrasting seeds,
logging per-step:
  - gate logits (mean, std per branch)
  - gate softmax ratios
  - gate logits grad norm (total + per-branch)
  - branch output norms (attn, trn, ret, mixed)
  - mixed grad norm (backward signal into branches)

Purpose: determine which of H1/H2/H3 explains TRN seed-sensitivity.

Usage:
    python scripts/run_trn_deep_diagnosis.py --seeds 0 3 --steps 500 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_src = os.path.join(os.path.dirname(__file__), "..", "src")
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _root]
sys.path.insert(0, _src)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trimemory.config import TRNConfig
from trimemory.tri_memory import TriMemoryEngine

# Import shared constants and dataset from v5 reeval
from run_trimemory_v5_trn_reeval import (
    RetrievalOnlyDataset,
    WINDOW_SIZE, CHUNK_SIZE, D_MODEL, N_LAYERS, N_OSC, D_FF,
    VOCAB_SIZE, SEQ_LEN, BATCH_SIZE, LR,
    weighted_cross_entropy_with_labels,
    compute_retrieval_aux_loss,
    W_RET_AUX, QUERY_MARKER_POS, MARKER_TEMPERATURE,
    H6_OLD_FACT_SPAN_LEN, QUERY_REGION_SIZE,
)
from run_trimemory_v3_eval import seed_everything


def make_cfg() -> TRNConfig:
    return TRNConfig(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_oscillators=N_OSC,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=SEQ_LEN + 16,
    )


def build_d_model(cfg: TRNConfig) -> TriMemoryEngine:
    """Build trimemory (D) config: TRN + Retrieval both enabled."""
    return TriMemoryEngine(
        cfg, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE,
        enable_trn=True, enable_retrieval=True,
    )


def enable_diagnosis(model: TriMemoryEngine) -> None:
    """Enable diagnosis probes on all TriMemoryBlocks."""
    for block in model.blocks:
        block._diag_enabled = True


def collect_diagnosis(model: TriMemoryEngine) -> dict:
    """Collect diagnosis data from all blocks, average across layers."""
    all_diag = []
    for block in model.blocks:
        d = getattr(block, "_diag", None)
        if d:
            all_diag.append(d)

    if not all_diag:
        return {}

    result = {}
    # Average scalar/tensor values across layers
    for key in all_diag[0]:
        vals = [d[key] for d in all_diag if key in d]
        if not vals:
            continue
        if isinstance(vals[0], torch.Tensor):
            stacked = torch.stack(vals)
            result[key] = stacked.mean(dim=0).tolist()
            if stacked.dim() > 1:
                result[f"{key}_std_across_layers"] = stacked.std(dim=0).tolist()
        else:
            result[key] = sum(vals) / len(vals)
    return result


def run_diagnosis(
    seed: int,
    steps: int,
    device: torch.device,
    batch_size: int | None = None,
) -> list[dict]:
    """Run D-only training with diagnosis probes, return per-step records."""
    bs = batch_size or BATCH_SIZE
    seed_everything(seed)
    cfg = make_cfg()
    model = build_d_model(cfg)
    model = model.to(device).train()
    enable_diagnosis(model)

    dataset = RetrievalOnlyDataset(n_samples=2000, seq_len=SEQ_LEN, seed=seed + 1000)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    records = []
    loader_it = iter(loader)

    for step in range(1, steps + 1):
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            batch = next(loader_it)

        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        wts = batch["loss_weights"].to(device)

        logits = model(
            ids,
            retrieval_query_mode="mean",
            retrieval_query_pos=None,
            retrieval_temperature=5.0,
            retrieval_decoder_mode="pooled",
        )["logits"]

        main_loss = weighted_cross_entropy_with_labels(logits, ids, lbl, wts)
        aux_loss = compute_retrieval_aux_loss(model)
        loss = main_loss + W_RET_AUX * aux_loss

        if not torch.isfinite(loss):
            records.append({"step": step, "loss": float("nan"), "stable": False})
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Collect diagnosis AFTER backward (grad hooks have fired)
        diag = collect_diagnosis(model)

        # Also collect TRN-specific param grad norms
        trn_grad_norms = {}
        for name, p in model.named_parameters():
            if p.grad is not None and "trn" in name:
                trn_grad_norms[f"grad_{name}"] = p.grad.detach().norm().item()

        # Gate proj grad norm
        for i, block in enumerate(model.blocks):
            if block.gate_proj.weight.grad is not None:
                trn_grad_norms[f"grad_block{i}_gate_proj_w"] = (
                    block.gate_proj.weight.grad.detach().norm().item()
                )
            if block.gate_proj.bias.grad is not None:
                trn_grad_norms[f"grad_block{i}_gate_proj_b"] = (
                    block.gate_proj.bias.grad.detach().norm().item()
                )

        optimizer.step()

        # Collect gate telemetry
        gate_tel = model.collect_gate_telemetry()

        record = {
            "step": step,
            "loss": loss.item(),
            "main_loss": main_loss.item(),
            "stable": True,
            "gate_kv": gate_tel["router_kv_ratio"],
            "gate_trn": gate_tel["router_trn_ratio"],
            "gate_ret": gate_tel["router_ret_ratio"],
            **diag,
            **trn_grad_norms,
        }
        records.append(record)

        if step % 50 == 0 or step <= 5:
            gl_mean = diag.get("gate_logits_mean", [0, 0, 0])
            gl_grad = diag.get("gate_logits_grad_norm", 0)
            trn_norm = diag.get("trn_out_norm", 0)
            attn_norm = diag.get("attn_out_norm", 0)
            mixed_grad = diag.get("mixed_grad_norm", 0)
            gl_per = diag.get("gate_logits_grad_per_branch", [0, 0, 0])

            # Format for readability
            if isinstance(gl_mean, list):
                gl_str = f"[{gl_mean[0]:+.3f},{gl_mean[1]:+.3f},{gl_mean[2]:+.3f}]"
            else:
                gl_str = str(gl_mean)
            if isinstance(gl_per, list):
                gp_str = f"[{gl_per[0]:.4f},{gl_per[1]:.4f},{gl_per[2]:.4f}]"
            else:
                gp_str = str(gl_per)

            trn_attn_ratio = trn_norm / max(attn_norm, 1e-8) if isinstance(trn_norm, (int, float)) else 0
            print(
                f"  step {step:>4} loss={loss.item():.4f} "
                f"gl={gl_str} "
                f"g_grad={gl_grad:.4f} gp={gp_str} "
                f"trn/attn={trn_attn_ratio:.3f} "
                f"mixed_grad={mixed_grad:.4f}",
                flush=True,
            )

    return records


def enable_nan_trace(model: TriMemoryEngine) -> None:
    """Enable NaN tracing on all TriMemoryBlocks and TRN internals."""
    for block in model.blocks:
        block._nan_trace_enabled = True
        block._nan_trace = None
        block._nan_trace_scalars = None
        # Enable rho debug in resonance layer
        block.trn._debug_trace = True

    # Hook norm_out to trace backward grad through final RMSNorm
    model._engine_grad_snapshot = {}

    def _hook_norm_out_input(module, input, output):
        """Forward hook on norm_out: register backward hooks on input/output."""
        inp = input[0]
        if inp.requires_grad:
            def _norm_in_grad(grad, _ref=model):
                g = grad.detach()
                is_fin = bool(torch.isfinite(g).all())
                _ref._engine_grad_snapshot["norm_out_input"] = {
                    "norm": g.norm().item() if is_fin else float("inf"),
                    "abs_max": g.abs().max().item() if is_fin else float("inf"),
                    "finite": is_fin,
                }
            inp.register_hook(_norm_in_grad)
        if output.requires_grad:
            def _norm_out_grad(grad, _ref=model):
                g = grad.detach()
                is_fin = bool(torch.isfinite(g).all())
                _ref._engine_grad_snapshot["norm_out_output"] = {
                    "norm": g.norm().item() if is_fin else float("inf"),
                    "abs_max": g.abs().max().item() if is_fin else float("inf"),
                    "finite": is_fin,
                }
            output.register_hook(_norm_out_grad)

    model.norm_out.register_forward_hook(_hook_norm_out_input)

    # Hook lm_head output (logits)
    def _hook_lm_head(module, input, output):
        if input[0].requires_grad:
            def _lm_input_grad(grad, _ref=model):
                g = grad.detach()
                is_fin = bool(torch.isfinite(g).all())
                _ref._engine_grad_snapshot["lm_head_input"] = {
                    "norm": g.norm().item() if is_fin else float("inf"),
                    "abs_max": g.abs().max().item() if is_fin else float("inf"),
                    "finite": is_fin,
                }
            input[0].register_hook(_lm_input_grad)
        if output.requires_grad:
            def _lm_output_grad(grad, _ref=model):
                g = grad.detach()
                is_fin = bool(torch.isfinite(g).all())
                _ref._engine_grad_snapshot["lm_head_output"] = {
                    "norm": g.norm().item() if is_fin else float("inf"),
                    "abs_max": g.abs().max().item() if is_fin else float("inf"),
                    "finite": is_fin,
                }
            output.register_hook(_lm_output_grad)

    model.lm_head.register_forward_hook(_hook_lm_head)


def collect_nan_trace(model: TriMemoryEngine) -> dict | None:
    """Return first NaN trace found across layers, or None."""
    for i, block in enumerate(model.blocks):
        trace = getattr(block, "_nan_trace", None)
        if trace is not None:
            trace["layer"] = i
            return trace
    return None


def collect_nan_scalars(model: TriMemoryEngine) -> list[dict]:
    """Collect per-layer scalar snapshots (last healthy step)."""
    result = []
    for i, block in enumerate(model.blocks):
        s = getattr(block, "_nan_trace_scalars", None)
        if s is not None:
            s["layer"] = i
            result.append(s)
    return result


def clear_nan_trace(model: TriMemoryEngine) -> None:
    """Reset NaN trace state for next step."""
    for block in model.blocks:
        block._nan_trace = None


def collect_grad_snapshots(model: TriMemoryEngine) -> dict:
    """Collect per-layer grad snapshots from backward hooks, average across layers."""
    all_snaps = []
    for block in model.blocks:
        snap = getattr(block, "_grad_snapshot", None)
        if snap:
            all_snaps.append(snap)
    if not all_snaps:
        return {}
    # Average across layers for each tensor
    result = {}
    all_keys = set()
    for s in all_snaps:
        all_keys.update(s.keys())
    for key in all_keys:
        entries = [s[key] for s in all_snaps if key in s]
        if not entries:
            continue
        if "norm" in entries[0]:
            result[f"grad_{key}_norm"] = sum(e["norm"] for e in entries) / len(entries)
            result[f"grad_{key}_abs_max"] = max(e["abs_max"] for e in entries)
            result[f"grad_{key}_finite"] = all(e["finite"] for e in entries)
        elif "val" in entries[0]:
            result[f"grad_{key}_val"] = sum(e["val"] for e in entries) / len(entries)
            result[f"grad_{key}_finite"] = all(e["finite"] for e in entries)
    return result


def collect_per_layer_grad_snapshots(model: TriMemoryEngine) -> list[dict]:
    """Collect per-layer grad snapshots (no averaging)."""
    result = []
    for i, block in enumerate(model.blocks):
        snap = getattr(block, "_grad_snapshot", None)
        if snap:
            entry = {"layer": i}
            for key, val in snap.items():
                for vk, vv in val.items():
                    entry[f"{key}_{vk}"] = vv
            result.append(entry)
    return result


def collect_param_grad_norms(model: TriMemoryEngine) -> dict:
    """Collect grad norms for key named parameters."""
    result = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        is_fin = bool(torch.isfinite(g).all())
        # Only record params of interest (TRN, gate, norm, res_scale)
        short = name.replace("blocks.", "b").replace(".weight", ".w").replace(".bias", ".b")
        if any(k in name for k in ("trn", "gate_proj", "trn_out_norm", "res_scale",
                                     "trn_branch_log_scale", "norm1", "norm2")):
            result[short] = {
                "norm": g.norm().item() if is_fin else float("inf"),
                "abs_max": g.abs().max().item() if is_fin else float("inf"),
                "finite": is_fin,
            }
    return result


GRAD_HISTORY_WINDOW = 60  # keep last 60 steps


def run_nan_trace(
    seed: int,
    steps: int,
    device: torch.device,
    batch_size: int | None = None,
    use_anomaly_detection: bool = False,
) -> dict:
    """Run D-only training with NaN tracing + per-step grad history. Stop at first NaN."""
    from collections import deque

    bs = batch_size or BATCH_SIZE
    seed_everything(seed)
    cfg = make_cfg()
    model = build_d_model(cfg)
    model = model.to(device).train()
    enable_diagnosis(model)
    enable_nan_trace(model)

    dataset = RetrievalOnlyDataset(n_samples=2000, seq_len=SEQ_LEN, seed=seed + 1000)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    records = []
    loader_it = iter(loader)
    nan_trace_result = None
    last_healthy_scalars = None
    last_healthy_diag = None
    anomaly_msg = None

    # Rolling grad history: deque of (step, grad_snapshot_dict)
    grad_history = deque(maxlen=GRAD_HISTORY_WINDOW)

    for step in range(1, steps + 1):
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            batch = next(loader_it)

        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        wts = batch["loss_weights"].to(device)

        # Clear per-step NaN trace
        clear_nan_trace(model)

        logits = model(
            ids,
            retrieval_query_mode="mean",
            retrieval_query_pos=None,
            retrieval_temperature=5.0,
            retrieval_decoder_mode="pooled",
        )["logits"]

        # Check forward NaN
        fwd_trace = collect_nan_trace(model)
        if fwd_trace is not None:
            nan_trace_result = {
                "step": step,
                "phase": "forward",
                "trace": fwd_trace,
                "last_healthy_scalars": last_healthy_scalars,
                "last_healthy_diag": last_healthy_diag,
                "grad_history": list(grad_history),
            }
            print(f"  [NaN FOUND] step={step} forward: {fwd_trace}", flush=True)
            break

        main_loss = weighted_cross_entropy_with_labels(logits, ids, lbl, wts)
        aux_loss = compute_retrieval_aux_loss(model)
        loss = main_loss + W_RET_AUX * aux_loss

        if not torch.isfinite(loss):
            nan_trace_result = {
                "step": step,
                "phase": "loss",
                "loss_val": loss.item() if torch.is_tensor(loss) else loss,
                "main_loss": main_loss.item(),
                "aux_loss": aux_loss.item(),
                "logits_finite": bool(torch.isfinite(logits).all()),
                "last_healthy_scalars": last_healthy_scalars,
                "last_healthy_diag": last_healthy_diag,
                "grad_history": list(grad_history),
            }
            print(f"  [NaN FOUND] step={step} loss: main={main_loss.item():.6f} aux={aux_loss.item():.6f}", flush=True)
            break

        optimizer.zero_grad()

        if use_anomaly_detection:
            try:
                with torch.autograd.detect_anomaly():
                    loss.backward()
            except RuntimeError as e:
                anomaly_msg = str(e)[:500]
                nan_trace_result = {
                    "step": step,
                    "phase": "backward_anomaly",
                    "anomaly_msg": anomaly_msg,
                    "loss": loss.item(),
                    "last_healthy_scalars": last_healthy_scalars,
                    "last_healthy_diag": last_healthy_diag,
                    "grad_history": list(grad_history),
                }
                print(f"  [ANOMALY] step={step}: {anomaly_msg[:200]}", flush=True)
                break
        else:
            loss.backward()

        # Collect tensor grad snapshots from hooks (BEFORE checking NaN)
        tensor_grads = collect_grad_snapshots(model)
        # Merge engine-level grad snapshots (norm_out, lm_head)
        engine_snaps = getattr(model, "_engine_grad_snapshot", {})
        for ek, ev in engine_snaps.items():
            if "norm" in ev:
                tensor_grads[f"grad_{ek}_norm"] = ev["norm"]
                tensor_grads[f"grad_{ek}_abs_max"] = ev["abs_max"]
                tensor_grads[f"grad_{ek}_finite"] = ev["finite"]
        model._engine_grad_snapshot = {}  # reset for next step
        param_grads = collect_param_grad_norms(model)

        # Check backward NaN trace
        bwd_trace = collect_nan_trace(model)
        if bwd_trace is not None:
            # Collect per-layer detail at NaN step
            per_layer = collect_per_layer_grad_snapshots(model)
            nan_trace_result = {
                "step": step,
                "phase": "backward",
                "trace": bwd_trace,
                "loss": loss.item(),
                "tensor_grads_at_nan": tensor_grads,
                "param_grads_at_nan": param_grads,
                "per_layer_grads_at_nan": per_layer,
                "engine_grads_at_nan": engine_snaps,
                "last_healthy_scalars": last_healthy_scalars,
                "last_healthy_diag": last_healthy_diag,
                "grad_history": list(grad_history),
            }
            print(f"  [NaN FOUND] step={step} backward: {bwd_trace}", flush=True)
            break

        # Check param grads for NaN
        grad_nan_params = []
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grad_nan_params.append(name)
        if grad_nan_params:
            nan_trace_result = {
                "step": step,
                "phase": "backward_param_grad",
                "nan_params": grad_nan_params[:10],
                "loss": loss.item(),
                "tensor_grads_at_nan": tensor_grads,
                "param_grads_at_nan": param_grads,
                "last_healthy_scalars": last_healthy_scalars,
                "last_healthy_diag": last_healthy_diag,
                "grad_history": list(grad_history),
            }
            print(f"  [NaN FOUND] step={step} param grads: {grad_nan_params[:5]}", flush=True)
            break

        # Record grad history entry (BEFORE clip)
        grad_entry = {
            "step": step,
            "loss": loss.item(),
            **tensor_grads,
        }
        # Add key param grads (flatten for JSON)
        for pname, pdata in param_grads.items():
            grad_entry[f"p_{pname}_norm"] = pdata["norm"]
            grad_entry[f"p_{pname}_abs_max"] = pdata["abs_max"]
        grad_history.append(grad_entry)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Collect diagnostics
        diag = collect_diagnosis(model)
        scalars = collect_nan_scalars(model)
        gate_tel = model.collect_gate_telemetry()

        record = {
            "step": step,
            "loss": loss.item(),
            "main_loss": main_loss.item(),
            "gate_kv": gate_tel["router_kv_ratio"],
            "gate_trn": gate_tel["router_trn_ratio"],
            "gate_ret": gate_tel["router_ret_ratio"],
            **diag,
        }
        if scalars:
            record["nan_scalars_layer0"] = scalars[0] if scalars else {}
        records.append(record)

        last_healthy_scalars = scalars
        last_healthy_diag = {k: v for k, v in diag.items()
                            if isinstance(v, (int, float))}

        if step % 50 == 0 or step <= 5:
            trn_norm = diag.get("trn_out_norm", 0)
            attn_norm = diag.get("attn_out_norm", 0)
            rho_max = diag.get("rho_abs_max", 0)
            scale = diag.get("trn_branch_scale", 0)
            res_s = diag.get("res_scale_val", 0)
            mixed_n = diag.get("mixed_norm", 0)
            if isinstance(mixed_n, torch.Tensor):
                mixed_n = mixed_n.item()
            blk_n = diag.get("block_output_norm", 0)
            # Grad summary
            g_mixed = tensor_grads.get("grad_mixed_norm", 0)
            g_gate = tensor_grads.get("grad_gate_logits_norm", 0)
            g_trn = tensor_grads.get("grad_trn_out_norm", 0)
            g_attn = tensor_grads.get("grad_attn_out_norm", 0)
            g_scale_val = tensor_grads.get("grad_trn_branch_log_scale_val", 0)
            print(
                f"  step {step:>4} loss={loss.item():.4f} "
                f"trn={trn_norm:.3f} attn={attn_norm:.3f} "
                f"rho_max={rho_max:.3f} scale={scale:.4f} res_s={res_s:.4f} "
                f"mixed={mixed_n:.3f} blk={blk_n:.3f} "
                f"g_trn={gate_tel['router_trn_ratio']:.3f} | "
                f"grad: mixed={g_mixed:.4f} gate={g_gate:.4f} "
                f"trn_out={g_trn:.4f} attn_out={g_attn:.4f} "
                f"scale={g_scale_val:.6f}",
                flush=True,
            )

    return {
        "seed": seed,
        "nan_found": nan_trace_result is not None,
        "nan_trace": nan_trace_result,
        "records": records,
        "total_steps_completed": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 3])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--nan-trace", action="store_true",
                        help="Enable NaN tracing mode (stop at first NaN)")
    parser.add_argument("--anomaly-detection", action="store_true",
                        help="Use torch.autograd.detect_anomaly (slow)")
    args = parser.parse_args()

    device = torch.device(args.device)
    bs = args.batch_size
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"artifacts/trn_deep_diagnosis/{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # NaN trace mode
    if args.nan_trace:
        for seed in args.seeds:
            print(f"\n{'='*70}")
            print(f"NaN TRACE: Seed {seed} -- {args.steps} steps -- bs={bs or BATCH_SIZE}")
            if args.anomaly_detection:
                print("  [anomaly detection ON]")
            print(f"{'='*70}")
            t0 = time.time()
            result = run_nan_trace(
                seed, args.steps, device, batch_size=bs,
                use_anomaly_detection=args.anomaly_detection,
            )
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")

            trace_path = out_dir / f"seed{seed}_nan_trace.json"
            # Convert non-serializable values
            def _serialize(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                if isinstance(obj, float) and (obj != obj):  # NaN
                    return "NaN"
                return obj
            with open(trace_path, "w") as f:
                json.dump(result, f, indent=2, default=_serialize)
            print(f"  Saved: {trace_path}")

            if result["nan_found"]:
                nt = result["nan_trace"]
                print(f"\n  === NaN TRACE RESULT ===")
                print(f"  Step: {nt['step']}")
                print(f"  Phase: {nt['phase']}")
                if "trace" in nt:
                    print(f"  Source: {nt['trace'].get('source', '?')}")
                    print(f"  Layer: {nt['trace'].get('layer', '?')}")
                    for k, v in nt["trace"].items():
                        if k not in ("source", "layer"):
                            print(f"    {k}: {v}")
                if "nan_params" in nt:
                    print(f"  NaN params: {nt['nan_params']}")
                if "anomaly_msg" in nt:
                    print(f"  Anomaly: {nt['anomaly_msg'][:300]}")

                # Tensor grads at NaN step
                if nt.get("tensor_grads_at_nan"):
                    print(f"\n  Tensor grad snapshots at NaN step:")
                    for k, v in sorted(nt["tensor_grads_at_nan"].items()):
                        print(f"    {k}: {v}")

                # Param grads at NaN step
                if nt.get("param_grads_at_nan"):
                    print(f"\n  Key param grads at NaN step:")
                    for k, v in sorted(nt["param_grads_at_nan"].items()):
                        print(f"    {k}: norm={v['norm']:.6f} abs_max={v['abs_max']:.6f} finite={v['finite']}")

                # Engine-level grads (norm_out, lm_head)
                if nt.get("engine_grads_at_nan"):
                    print(f"\n  Engine-level grad snapshots at NaN step:")
                    for k, v in sorted(nt["engine_grads_at_nan"].items()):
                        if isinstance(v, dict):
                            print(f"    {k}: norm={v.get('norm', '?')} abs_max={v.get('abs_max', '?')} finite={v.get('finite', '?')}")

                # Per-layer grad detail at NaN
                if nt.get("per_layer_grads_at_nan"):
                    print(f"\n  Per-layer grad detail at NaN step:")
                    for layer_data in nt["per_layer_grads_at_nan"]:
                        layer_i = layer_data.get("layer", "?")
                        items = {k: v for k, v in layer_data.items() if k != "layer"}
                        finite_flags = {k: v for k, v in items.items() if "finite" in k}
                        norms = {k: v for k, v in items.items() if "norm" in k}
                        non_finite = [k for k, v in finite_flags.items() if not v]
                        if non_finite:
                            print(f"    Layer {layer_i}: NON-FINITE in {non_finite}")
                            for k, v in sorted(norms.items()):
                                print(f"      {k}: {v}")

                # Last healthy scalars
                if nt.get("last_healthy_diag"):
                    print(f"\n  Last healthy step diagnostics:")
                    for k, v in nt["last_healthy_diag"].items():
                        if isinstance(v, float):
                            print(f"    {k}: {v:.6f}")

                # Grad history analysis: find first step with grad > 10x median
                gh = nt.get("grad_history", [])
                if gh:
                    nan_step = nt["step"]
                    print(f"\n  === GRAD HISTORY (last {len(gh)} steps before NaN @ step {nan_step}) ===")
                    # Extract key grad time series
                    grad_keys = [k for k in gh[0].keys() if k.startswith("grad_") and k.endswith("_norm")]
                    for gk in sorted(grad_keys):
                        vals = [e.get(gk, 0) for e in gh]
                        finite_vals = [v for v in vals if v != float("inf") and v == v]
                        if not finite_vals:
                            print(f"    {gk}: all non-finite")
                            continue
                        median_v = sorted(finite_vals)[len(finite_vals) // 2]
                        max_v = max(finite_vals)
                        max_step = gh[vals.index(max_v)]["step"] if max_v in vals else "?"
                        # Find first step where grad > 10x median
                        anomaly_step = None
                        if median_v > 0:
                            for e in gh:
                                v = e.get(gk, 0)
                                if v > 10 * median_v:
                                    anomaly_step = e["step"]
                                    break
                        print(
                            f"    {gk}: median={median_v:.6f} max={max_v:.6f} "
                            f"(max@step={max_step}) "
                            f"first_10x_anomaly={'step ' + str(anomaly_step) if anomaly_step else 'none'}"
                        )

                    # Print last 10 steps detail
                    print(f"\n  Last 10 steps grad detail:")
                    for e in gh[-10:]:
                        parts = [f"step={e['step']:>4} loss={e.get('loss', 0):.4f}"]
                        for gk in sorted(grad_keys):
                            v = e.get(gk, 0)
                            parts.append(f"{gk.replace('grad_', '').replace('_norm', '')}={v:.4f}")
                        # Also show trn_branch_log_scale grad
                        sv = e.get("grad_trn_branch_log_scale_val", None)
                        if sv is not None:
                            parts.append(f"scale_grad={sv:.6f}")
                        print(f"    {' '.join(parts)}")
            else:
                print(f"  No NaN found in {args.steps} steps")

            torch.cuda.empty_cache()
        return

    all_results = {}
    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"Seed {seed} -- D (trimemory) -- {args.steps} steps -- bs={bs or BATCH_SIZE}")
        print(f"{'='*70}")
        t0 = time.time()
        records = run_diagnosis(seed, args.steps, device, batch_size=bs)
        elapsed = time.time() - t0
        all_results[str(seed)] = records
        print(f"  Done in {elapsed:.1f}s, {len(records)} records")

        # Save per-seed
        seed_path = out_dir / f"seed{seed}_diagnosis.json"
        with open(seed_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  Saved: {seed_path}")

        # Free GPU memory
        torch.cuda.empty_cache()

    # Save combined
    combined_path = out_dir / "all_diagnosis.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] All results: {combined_path}")

    # Quick summary: compare key metrics at step 50, 200, 500
    print(f"\n{'='*70}")
    print("SUMMARY: seed comparison at key steps")
    print(f"{'='*70}")
    for checkpoint in [10, 50, 200, args.steps]:
        print(f"\n--- Step {checkpoint} ---")
        for seed_str, records in all_results.items():
            matching = [r for r in records if r["step"] == checkpoint]
            if not matching:
                continue
            r = matching[0]
            gl_grad = r.get("gate_logits_grad_norm", 0)
            trn_norm = r.get("trn_out_norm", 0)
            attn_norm = r.get("attn_out_norm", 0)
            mixed_grad = r.get("mixed_grad_norm", 0)
            gl_per = r.get("gate_logits_grad_per_branch", [0, 0, 0])
            ratio = trn_norm / max(attn_norm, 1e-8) if isinstance(trn_norm, (int, float)) else 0
            gp_str = ", ".join(f"{v:.4f}" for v in gl_per) if isinstance(gl_per, list) else str(gl_per)
            print(
                f"  seed={seed_str:>2} loss={r['loss']:.4f} "
                f"g_kv={r.get('gate_kv', 0):.3f} g_trn={r.get('gate_trn', 0):.3f} g_ret={r.get('gate_ret', 0):.3f} "
                f"gl_grad={gl_grad:.4f} gl_per=[{gp_str}] "
                f"trn/attn={ratio:.3f} mixed_grad={mixed_grad:.4f}"
            )


if __name__ == "__main__":
    main()
