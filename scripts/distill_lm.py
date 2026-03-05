"""Knowledge distillation: pretrained HF Transformer teacher -> TRN student.

Usage (run from scripts/ dir):
    cd scripts

    # Quick smoke test (< 5 min, CPU)
    python distill_lm.py --quick --device cpu

    # Full 100k-step run (GPU)
    python distill_lm.py --student-size 100m --teacher gpt2 --steps 100000 --device cuda

    # CE-only baseline (no distillation)
    python distill_lm.py --kl-weight 0.0 --ce-weight 1.0 --steps 2000 --device cpu

    # Cache teacher logits to disk
    python distill_lm.py --cache-teacher-logits --steps 5000 --device cpu

Output CSV: scripts/results/distill_{student}_{teacher}_curves.csv
Checkpoints: checkpoints/distill_{student}_{teacher}/step_NNNNNN.pt
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trn.config import TRNConfig
from trn.model import TRNModel
from trn.checkpoint import save_checkpoint
from trn.scheduler import CosineWithWarmup

CACHE_DIR = ROOT / "scripts" / "data"
RESULTS_DIR = ROOT / "scripts" / "results"


# ---------------------------------------------------------------------------
# Student presets
# ---------------------------------------------------------------------------

STUDENT_PRESETS: dict[str, callable] = {
    "small": lambda vs: TRNConfig(
        vocab_size=vs, d_model=128, n_oscillators=64,
        n_layers=4, d_ff=512, max_seq_len=1024,
    ),
    "100m": lambda vs: TRNConfig.trn_100m(),
    "400m": lambda vs: TRNConfig.trn_400m(),
    "1b": lambda vs: TRNConfig.trn_1b(),
}


# ---------------------------------------------------------------------------
# Teacher loading
# ---------------------------------------------------------------------------

def load_teacher(name_or_path: str, device: str) -> tuple[nn.Module, object]:
    """Load a HuggingFace causal LM as frozen teacher.

    Returns (model, tokenizer). Model is on device, eval mode, no grad.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[teacher] Loading {name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForCausalLM.from_pretrained(name_or_path)
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    vocab_size = model.config.vocab_size
    print(f"[teacher] {name_or_path}: {n_params:,} params, vocab={vocab_size}")
    return model, tokenizer


def build_student(preset: str, vocab_size: int, device: str) -> TRNModel:
    """Build a TRNModel student from preset name."""
    if preset in ("100m", "400m", "1b"):
        cfg = STUDENT_PRESETS[preset](vocab_size)
        cfg.vocab_size = vocab_size
    else:
        cfg = STUDENT_PRESETS[preset](vocab_size)

    model = TRNModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[student] {preset}: {n_params:,} params, d_model={cfg.d_model}, "
          f"n_layers={cfg.n_layers}, n_osc={cfg.n_oscillators}")
    return model


# ---------------------------------------------------------------------------
# Data: tokenize with teacher's tokenizer, cache as .npy
# ---------------------------------------------------------------------------

def _load_text() -> tuple[str, str]:
    """Load train/val text. Tries HuggingFace WikiText-2, then NLTK Gutenberg, then synthetic."""
    try:
        from datasets import load_dataset
        ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        ds_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        train_text = "\n".join(t for t in ds_train["text"] if t.strip())
        val_text = "\n".join(t for t in ds_val["text"] if t.strip())
        print(f"[data] WikiText-2: train={len(train_text):,} chars, val={len(val_text):,} chars")
        return train_text, val_text
    except Exception as e:
        print(f"[data] WikiText-2 unavailable: {e}")

    try:
        import nltk
        nltk.download("gutenberg", quiet=True)
        from nltk.corpus import gutenberg
        text = gutenberg.raw()
        split = int(len(text) * 0.9)
        train_text, val_text = text[:split], text[split:]
        print(f"[data] NLTK Gutenberg: train={len(train_text):,}, val={len(val_text):,} chars")
        return train_text, val_text
    except Exception as e:
        print(f"[data] NLTK unavailable: {e}")

    print("[data] Using synthetic fallback")
    sample = (
        "the quick brown fox jumps over the lazy dog. "
        "a language model learns to predict the next token in a sequence. "
        "knowledge distillation transfers learned representations from teacher to student. "
    ) * 10000
    split = int(len(sample) * 0.9)
    return sample[:split], sample[split:]


def tokenize_dataset(
    tokenizer, teacher_name: str, force_rebuild: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize text with teacher tokenizer and cache as .npy."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = teacher_name.replace("/", "_")
    train_path = CACHE_DIR / f"distill_{safe_name}_train.npy"
    val_path = CACHE_DIR / f"distill_{safe_name}_val.npy"

    if not force_rebuild and train_path.exists() and val_path.exists():
        train_ids = np.load(train_path)
        val_ids = np.load(val_path)
        print(f"[data] Cache hit: train={len(train_ids):,}, val={len(val_ids):,} tokens")
        return train_ids, val_ids

    print("[data] Tokenizing with teacher tokenizer ...")
    train_text, val_text = _load_text()

    train_ids = np.array(tokenizer.encode(train_text), dtype=np.int32)
    val_ids = np.array(tokenizer.encode(val_text), dtype=np.int32)

    np.save(train_path, train_ids)
    np.save(val_path, val_ids)
    print(f"[data] Cached: train={len(train_ids):,}, val={len(val_ids):,} tokens")
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Teacher logit caching (optional, for small datasets)
# ---------------------------------------------------------------------------

def cache_teacher_logits(
    teacher: nn.Module,
    train_ids: np.ndarray,
    seq_len: int,
    batch_size: int,
    device: str,
    cache_path: Path,
) -> np.ndarray:
    """Run teacher over all training data and save logits to disk as mmap .npy."""
    if cache_path.exists():
        print(f"[cache] Loading teacher logits from {cache_path}")
        return np.load(cache_path, mmap_mode="r")

    print("[cache] Computing teacher logits (this may take a while) ...")
    n_tokens = len(train_ids)
    n_chunks = n_tokens // seq_len
    vocab_size = teacher.config.vocab_size

    # Pre-allocate on disk
    logits_shape = (n_chunks, seq_len, vocab_size)
    fp = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float16, shape=logits_shape,
    )

    teacher.eval()
    with torch.no_grad():
        for i in range(0, n_chunks, batch_size):
            end = min(i + batch_size, n_chunks)
            batch_ids = []
            for j in range(i, end):
                chunk = train_ids[j * seq_len: (j + 1) * seq_len]
                batch_ids.append(chunk)
            input_tensor = torch.tensor(np.stack(batch_ids), dtype=torch.long, device=device)
            out = teacher(input_tensor)
            logits = out.logits.cpu().to(torch.float16).numpy()
            fp[i:end] = logits

            if (i // batch_size) % 50 == 0:
                print(f"  [cache] {end}/{n_chunks} chunks")

    print(f"[cache] Saved teacher logits: {cache_path} ({fp.nbytes / 1e9:.1f} GB)")
    return np.load(cache_path, mmap_mode="r")


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------

def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    kl_weight: float,
    ce_weight: float,
) -> dict[str, torch.Tensor]:
    """Compute distillation loss = kl_weight * KL + ce_weight * CE.

    KL divergence uses temperature scaling with T^2 correction.
    CE uses hard labels (next-token prediction).

    All inputs use causal shift: logits[:, :-1] predicts labels[:, 1:].
    """
    # Causal shift
    s_logits = student_logits[:, :-1].contiguous()
    t_logits = teacher_logits[:, :-1].contiguous()
    targets = labels[:, 1:].contiguous()

    result = {}
    total = torch.tensor(0.0, device=student_logits.device)

    # KL divergence with temperature
    # Flatten to 2D (B*S, V) so batchmean divides by B*S (per-token),
    # matching CE's per-token normalization.
    if kl_weight > 0.0:
        flat_s = s_logits.view(-1, s_logits.size(-1))
        flat_t = t_logits.view(-1, t_logits.size(-1))
        s_log_probs = F.log_softmax(flat_s / temperature, dim=-1)
        t_probs = F.softmax(flat_t / temperature, dim=-1)
        kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)
        result["kl_loss"] = kl
        total = total + kl_weight * kl

    # Cross-entropy with hard labels
    if ce_weight > 0.0:
        ce = F.cross_entropy(
            s_logits.view(-1, s_logits.size(-1)),
            targets.view(-1),
        )
        result["ce_loss"] = ce
        total = total + ce_weight * ce

    result["loss"] = total
    return result


# ---------------------------------------------------------------------------
# Batch sampling
# ---------------------------------------------------------------------------

def sample_batch(
    token_ids: np.ndarray,
    seq_len: int,
    batch_size: int,
    rng: np.random.Generator,
    device: str,
) -> torch.Tensor:
    """Sample a batch of random contiguous sequences."""
    max_start = max(1, len(token_ids) - seq_len - 1)
    starts = rng.integers(0, max_start, size=batch_size)
    seqs = [token_ids[s: s + seq_len] for s in starts]
    return torch.tensor(np.stack(seqs), dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_val(
    student: TRNModel,
    val_ids: np.ndarray,
    seq_len: int,
    batch_size: int,
    device: str,
    n_batches: int = 50,
) -> tuple[float, float]:
    """Evaluate student on val set. Returns (val_ppl, val_loss)."""
    student.eval()
    rng = np.random.default_rng(0)
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            batch = sample_batch(val_ids, seq_len, batch_size, rng, device)
            out = student(batch, labels=batch)
            total_loss += out["loss"].item()
    student.train()
    avg_loss = total_loss / n_batches
    return math.exp(avg_loss), avg_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_grad_norm(model: nn.Module) -> float:
    """Compute total gradient L2 norm (unclipped)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.float().norm().item() ** 2
    return total ** 0.5


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(args: argparse.Namespace) -> float:
    """Main distillation training loop. Returns final val_ppl."""
    seed_everything(args.seed)

    # Load teacher
    teacher, tokenizer = load_teacher(args.teacher, args.device)
    vocab_size = teacher.config.vocab_size

    # Tokenize data
    train_ids, val_ids = tokenize_dataset(tokenizer, args.teacher)

    # Build student
    student = build_student(args.student_size, vocab_size, args.device)

    # Optional: cache teacher logits
    cached_logits = None
    if args.cache_teacher_logits:
        safe_name = args.teacher.replace("/", "_")
        cache_path = CACHE_DIR / f"distill_{safe_name}_teacher_logits.npy"
        cached_logits = cache_teacher_logits(
            teacher, train_ids, args.seq_len, args.batch_size,
            args.device, cache_path,
        )

    # Optimizer
    param_groups = student.configure_optimizer_param_groups(weight_decay=0.1)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    scheduler = CosineWithWarmup(
        optimizer,
        warmup_steps=args.warmup,
        max_steps=args.steps,
        lr=args.lr,
        min_lr=args.lr * 0.1,
    )

    use_amp = args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)

    # CSV setup
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_teacher = args.teacher.replace("/", "_")
    csv_path = RESULTS_DIR / f"distill_{args.student_size}_{safe_teacher}_curves.csv"
    ckpt_dir = ROOT / "checkpoints" / f"distill_{args.student_size}_{safe_teacher}"

    fieldnames = [
        "step", "train_loss", "kl_loss", "ce_loss",
        "val_loss", "val_ppl", "tps", "peak_mb", "grad_norm", "lr",
    ]

    rng = np.random.default_rng(args.seed)
    log_every = args.eval_every
    save_every = args.save_every

    print(f"\n{'='*60}")
    print(f"  Distillation: {args.teacher} -> TRN/{args.student_size}")
    print(f"  Steps: {args.steps}, LR: {args.lr}, T: {args.temperature}")
    print(f"  KL weight: {args.kl_weight}, CE weight: {args.ce_weight}")
    print(f"  Batch: {args.batch_size}, Seq len: {args.seq_len}")
    print(f"  Device: {args.device}, Seed: {args.seed}")
    print(f"{'='*60}\n")

    final_val_ppl = float("nan")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        loss_acc = 0.0
        kl_acc = 0.0
        ce_acc = 0.0
        tokens_acc = 0
        t0 = time.perf_counter()
        grad_norms: list[float] = []

        student.train()

        for step in range(1, args.steps + 1):
            current_lr = scheduler.step(step)
            batch = sample_batch(train_ids, args.seq_len, args.batch_size, rng, args.device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    # Student forward
                    s_out = student(batch)
                    s_logits = s_out["logits"]

                    # Teacher forward (or cached)
                    with torch.no_grad():
                        t_out = teacher(batch)
                        t_logits = t_out.logits

                    losses = distill_loss(
                        s_logits, t_logits, batch,
                        args.temperature, args.kl_weight, args.ce_weight,
                    )
                    loss = losses["loss"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = compute_grad_norm(student)
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Student forward
                s_out = student(batch)
                s_logits = s_out["logits"]

                # Teacher forward
                with torch.no_grad():
                    t_out = teacher(batch)
                    t_logits = t_out.logits

                losses = distill_loss(
                    s_logits, t_logits, batch,
                    args.temperature, args.kl_weight, args.ce_weight,
                )
                loss = losses["loss"]

                loss.backward()
                gn = compute_grad_norm(student)
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

            loss_acc += loss.item()
            kl_acc += losses.get("kl_loss", torch.tensor(0.0)).item()
            ce_acc += losses.get("ce_loss", torch.tensor(0.0)).item()
            tokens_acc += args.batch_size * args.seq_len
            grad_norms.append(gn)

            if step % log_every == 0 or step == args.steps:
                t1 = time.perf_counter()
                elapsed = t1 - t0
                tps = tokens_acc / elapsed if elapsed > 0 else 0.0
                peak_mb = (
                    torch.cuda.max_memory_allocated(args.device) / 1e6
                    if args.device == "cuda" else 0.0
                )

                avg_loss = loss_acc / log_every
                avg_kl = kl_acc / log_every
                avg_ce = ce_acc / log_every

                val_ppl, val_loss = eval_val(
                    student, val_ids, args.seq_len, args.batch_size, args.device,
                )
                final_val_ppl = val_ppl

                gn_median = sorted(grad_norms[-log_every:])[len(grad_norms[-log_every:]) // 2]

                print(
                    f"  step={step:6d}/{args.steps}"
                    f"  loss={avg_loss:.4f}  kl={avg_kl:.4f}  ce={avg_ce:.4f}"
                    f"  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}"
                    f"  tps={tps:.0f}  gn={gn_median:.1f}  lr={current_lr:.2e}"
                )

                writer.writerow({
                    "step": step,
                    "train_loss": f"{avg_loss:.6f}",
                    "kl_loss": f"{avg_kl:.6f}",
                    "ce_loss": f"{avg_ce:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_ppl": f"{val_ppl:.4f}",
                    "tps": f"{tps:.1f}",
                    "peak_mb": f"{peak_mb:.1f}",
                    "grad_norm": f"{gn_median:.2f}",
                    "lr": f"{current_lr:.2e}",
                })
                f.flush()

                loss_acc = 0.0
                kl_acc = 0.0
                ce_acc = 0.0
                tokens_acc = 0
                t0 = t1

            if save_every > 0 and step % save_every == 0:
                save_checkpoint(
                    student, optimizer, step=step, loss=loss.item(),
                    checkpoint_dir=ckpt_dir, tag=f"step_{step:06d}",
                )

    # Final summary
    val_ppl, val_loss = eval_val(
        student, val_ids, args.seq_len, args.batch_size, args.device,
    )
    print(f"\n{'='*60}")
    print(f"  Distillation complete")
    print(f"  Teacher: {args.teacher}")
    print(f"  Student: TRN/{args.student_size}")
    print(f"  Final val_loss: {val_loss:.4f}")
    print(f"  Final val_ppl:  {val_ppl:.2f}")
    print(f"  CSV: {csv_path}")
    print(f"{'='*60}")

    return val_ppl


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Knowledge distillation: HF teacher -> TRN student",
    )
    parser.add_argument("--student-size", choices=list(STUDENT_PRESETS.keys()),
                        default="small", help="Student model preset (default: small)")
    parser.add_argument("--teacher", type=str, default="gpt2",
                        help="HuggingFace teacher model (default: gpt2)")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Training steps (default: 10000)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length (default: 256)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate (default: 3e-4)")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="Warmup steps (default: 1000)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda (default: cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--save-every", type=int, default=10000,
                        help="Checkpoint interval (default: 10000, 0=disabled)")
    parser.add_argument("--eval-every", type=int, default=500,
                        help="Eval/log interval (default: 500)")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Distillation temperature (default: 2.0)")
    parser.add_argument("--kl-weight", type=float, default=1.0,
                        help="KL loss weight (default: 1.0)")
    parser.add_argument("--ce-weight", type=float, default=0.1,
                        help="CE loss weight (default: 0.1)")
    parser.add_argument("--cache-teacher-logits", action="store_true",
                        help="Cache teacher logits to disk (saves repeated forward passes)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1000 steps, small student, eval every 200")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        args.steps = 1000
        args.student_size = "small"
        args.eval_every = 200
        args.save_every = 0
        args.warmup = 100
        print("[quick] Quick mode: 1000 steps, small student")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU")
        args.device = "cpu"

    train(args)


if __name__ == "__main__":
    main()
