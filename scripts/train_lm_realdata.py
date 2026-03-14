"""Train TRN vs Transformer on WikiText-2 (character-level or BPE, real text).

Usage (run from scripts/ dir to avoid profile.py conflict at project root):
    cd scripts

    # Char-level smoke test
    python train_lm_realdata.py --model trn --size small --steps 500 --batch-size 8 --seq-len 256
    python train_lm_realdata.py --model tf  --size small --steps 500 --batch-size 8 --seq-len 256

    # BPE tokenizer
    python train_lm_realdata.py --model trn --size small --steps 2000 --tokenizer bpe

    # Random-target sanity check
    python train_lm_realdata.py --model trn --size small --steps 500 --random-targets

    # Debug sample logging
    python train_lm_realdata.py --model trn --size small --steps 500 --debug-samples 3

    # GPU full run
    python train_lm_realdata.py --model trn --size large --steps 10000 --device cuda

Output CSV: scripts/results/train_lm_realdata_{model}_{size}[_{suffix}]_curves.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add src/ to path so we can import trimemory without install
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trimemory.config import TRNConfig
from trimemory.model import TRNModel
from trimemory.baseline import TransformerModel
from trimemory.tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# Model size configs
# ---------------------------------------------------------------------------

SIZE_CONFIGS: dict[str, dict] = {
    "small":  dict(d_model=128, n_layers=4,  n_oscillators=64,  d_ff=512),
    "medium": dict(d_model=256, n_layers=6,  n_oscillators=128, d_ff=1024),
    "large":  dict(d_model=768, n_layers=12, n_oscillators=256, d_ff=3072),
}


# ---------------------------------------------------------------------------
# Data loading & caching
# ---------------------------------------------------------------------------

CACHE_DIR = ROOT / "scripts" / "data"
TOK_PATH  = CACHE_DIR / "wikitext2_char_tokenizer.json"
TRAIN_NPY = CACHE_DIR / "wikitext2_train.npy"
VAL_NPY   = CACHE_DIR / "wikitext2_val.npy"

# BPE cache paths
BPE_TOK_PATH  = CACHE_DIR / "wikitext2_bpe_tokenizer.json"
BPE_TRAIN_NPY = CACHE_DIR / "wikitext2_bpe_train.npy"
BPE_VAL_NPY   = CACHE_DIR / "wikitext2_bpe_val.npy"


def _load_wikitext2_text() -> tuple[str, str]:
    """Return (train_text, val_text). Tries HuggingFace, NLTK, then synthetic."""
    try:
        from datasets import load_dataset
        ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        ds_val   = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        train_text = "\n".join(t for t in ds_train["text"] if t.strip())
        val_text   = "\n".join(t for t in ds_val["text"]   if t.strip())
        print(f"[data] HuggingFace WikiText-2: train={len(train_text):,} chars, val={len(val_text):,} chars")
        return train_text, val_text
    except Exception as hf_err:
        print(f"[data] HuggingFace unavailable: {hf_err}")

    try:
        import nltk
        nltk.download("gutenberg", quiet=True)
        from nltk.corpus import gutenberg
        text  = gutenberg.raw()
        split = int(len(text) * 0.9)
        train_text, val_text = text[:split], text[split:]
        print(f"[data] NLTK Gutenberg: train={len(train_text):,} val={len(val_text):,} chars")
        return train_text, val_text
    except Exception as nltk_err:
        print(f"[data] NLTK unavailable: {nltk_err}")

    # Synthetic fallback: repeating English-like sentences
    print("[data] Using synthetic text fallback")
    sample = (
        "the quick brown fox jumps over the lazy dog. "
        "a language model learns to predict the next character given the preceding context. "
        "temporal resonance networks use oscillatory dynamics instead of self-attention. "
        "characters flow through layers of resonance blocks updating hidden state recurrently. "
    ) * 6000
    split = int(len(sample) * 0.9)
    return sample[:split], sample[split:]


def load_or_build_cache() -> tuple[np.ndarray, np.ndarray, CharTokenizer]:
    """Load tokenized WikiText-2 from .npy cache, or rebuild and cache it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_NPY.exists() and VAL_NPY.exists() and TOK_PATH.exists():
        print("[data] Loading from cache ...")
        tok       = CharTokenizer.load(TOK_PATH)
        train_ids = np.load(TRAIN_NPY)
        val_ids   = np.load(VAL_NPY)
        print(f"[data] Cache hit: train={len(train_ids):,} tokens, val={len(val_ids):,} tokens, vocab={tok.vocab_size}")
        return train_ids, val_ids, tok

    print("[data] Building cache ...")
    train_text, val_text = _load_wikitext2_text()

    tok = CharTokenizer()
    tok.fit(train_text + val_text)

    train_ids = np.array(tok.encode(train_text), dtype=np.int32)
    val_ids   = np.array(tok.encode(val_text),   dtype=np.int32)

    tok.save(TOK_PATH)
    np.save(TRAIN_NPY, train_ids)
    np.save(VAL_NPY,   val_ids)

    print(f"[data] Cached: train={len(train_ids):,} tokens, val={len(val_ids):,} tokens, vocab={tok.vocab_size}")
    return train_ids, val_ids, tok


# ---------------------------------------------------------------------------
# BPE tokenizer support
# ---------------------------------------------------------------------------

class BPETokenizerWrapper:
    """Thin wrapper around HuggingFace tokenizers for BPE."""

    def __init__(self, vocab_size: int = 8192) -> None:
        self._vocab_size = vocab_size
        self._tokenizer = None

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def train_from_text(self, text: str) -> None:
        """Train a BPE tokenizer from raw text."""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        trainer = trainers.BpeTrainer(
            vocab_size=self._vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            min_frequency=2,
        )

        # Train from iterator (split into lines for the trainer)
        lines = text.split("\n")
        tokenizer.train_from_iterator(lines, trainer=trainer)
        self._tokenizer = tokenizer
        self._vocab_size = tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        assert self._tokenizer is not None
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        assert self._tokenizer is not None
        return self._tokenizer.decode(ids)

    def save(self, path: Path) -> None:
        assert self._tokenizer is not None
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "BPETokenizerWrapper":
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(path))
        wrapper = cls(vocab_size=tokenizer.get_vocab_size())
        wrapper._tokenizer = tokenizer
        return wrapper


def load_or_build_bpe_cache(
    bpe_vocab_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, BPETokenizerWrapper]:
    """Load BPE-tokenized WikiText-2 from cache, or build it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if BPE_TRAIN_NPY.exists() and BPE_VAL_NPY.exists() and BPE_TOK_PATH.exists():
        print("[data/bpe] Loading from cache ...")
        tok = BPETokenizerWrapper.load(BPE_TOK_PATH)
        train_ids = np.load(BPE_TRAIN_NPY)
        val_ids   = np.load(BPE_VAL_NPY)
        print(f"[data/bpe] Cache hit: train={len(train_ids):,} tokens, val={len(val_ids):,} tokens, vocab={tok.vocab_size}")
        return train_ids, val_ids, tok

    print("[data/bpe] Building BPE tokenizer and cache ...")
    train_text, val_text = _load_wikitext2_text()

    tok = BPETokenizerWrapper(vocab_size=bpe_vocab_size)
    tok.train_from_text(train_text + val_text)

    train_ids = np.array(tok.encode(train_text), dtype=np.int32)
    val_ids   = np.array(tok.encode(val_text),   dtype=np.int32)

    tok.save(BPE_TOK_PATH)
    np.save(BPE_TRAIN_NPY, train_ids)
    np.save(BPE_VAL_NPY,   val_ids)

    print(f"[data/bpe] Cached: train={len(train_ids):,} tokens, val={len(val_ids):,} tokens, vocab={tok.vocab_size}")
    return train_ids, val_ids, tok


# ---------------------------------------------------------------------------
# Batch sampling
# ---------------------------------------------------------------------------

def sample_batch(
    token_ids:  np.ndarray,
    seq_len:    int,
    batch_size: int,
    rng:        np.random.Generator,
    device:     str,
) -> torch.Tensor:
    """Sample a batch of random contiguous sequences."""
    max_start = max(1, len(token_ids) - seq_len - 1)
    starts    = rng.integers(0, max_start, size=batch_size)
    seqs      = [token_ids[s : s + seq_len] for s in starts]
    return torch.tensor(np.stack(seqs), dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(model_type: str, size: str, vocab_size: int, seq_len: int) -> nn.Module:
    sc  = SIZE_CONFIGS[size]
    cfg = TRNConfig(
        vocab_size=vocab_size,
        d_model=sc["d_model"],
        n_oscillators=sc["n_oscillators"],
        n_layers=sc["n_layers"],
        d_ff=sc["d_ff"],
        max_seq_len=seq_len,
        dropout=0.0,
        use_parallel_scan=True,
        tie_weights=True,
        # P0 stabilization defaults (applied to both TRN and TF; TF ignores them)
        log_phase=True,
        amplitude_max=3.0,
        state_norm=True,
        res_scale_init=0.05,
        gate_bias_init=0.65,
        phase_mode="log",
    )
    if model_type == "trn":
        return TRNModel(cfg)
    return TransformerModel(cfg)


# ---------------------------------------------------------------------------
# Debug sample logging
# ---------------------------------------------------------------------------

def log_debug_samples(
    model: nn.Module,
    batch: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    n_samples: int,
    tok,
    model_type: str,
) -> None:
    """Print decoded input/target/prediction samples for visual inspection."""
    model.eval()
    with torch.no_grad():
        out = model(batch, labels=labels)
        logits = out["logits"]  # (B, seq_len, vocab)

    model.train()

    # model.forward uses causal shift: logits[:, :-1] predicts labels[:, 1:]
    # So logits[:, t] predicts the token at position t+1 in labels
    pred_ids = logits[:, :-1].argmax(dim=-1)  # (B, seq_len-1)

    B = min(n_samples, batch.size(0))
    print(f"\n    --- Debug Samples (step={step}) ---")
    for b in range(B):
        inp_ids  = batch[b].tolist()
        tgt_ids  = labels[b].tolist()
        pred_top = pred_ids[b].tolist()

        # Show last 20 positions for brevity
        show_len = 20
        inp_tail  = inp_ids[-show_len:]
        # Targets after causal shift: labels[1:] are what logits[:-1] predicts
        tgt_tail  = tgt_ids[-(show_len - 1):]
        pred_tail = pred_top[-(show_len - 1):]

        # Decode if possible
        if hasattr(tok, 'decode'):
            try:
                inp_str  = tok.decode(inp_tail)
                tgt_str  = tok.decode(tgt_tail)
                pred_str = tok.decode(pred_tail)
            except Exception:
                inp_str  = str(inp_tail)
                tgt_str  = str(tgt_tail)
                pred_str = str(pred_tail)
        else:
            inp_str  = str(inp_tail)
            tgt_str  = str(tgt_tail)
            pred_str = str(pred_tail)

        print(f"    [sample {b}]")
        print(f"      input (last {show_len}): {inp_str.encode('ascii', 'replace').decode()}")
        print(f"      target (shifted):        {tgt_str.encode('ascii', 'replace').decode()}")
        print(f"      pred top-1:              {pred_str.encode('ascii', 'replace').decode()}")

        # Verify target is next-token (not identity)
        # input[1:] should equal target[:-1] if labels=input_ids
        if inp_ids[1:show_len] == tgt_ids[:show_len - 1]:
            print(f"      [OK] target = next-token shift of input")
        else:
            print(f"      [WARN] target does NOT match next-token shift")

    print(f"    --- End Debug Samples ---\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def eval_val_perplexity(
    model:          nn.Module,
    val_ids:        np.ndarray,
    seq_len:        int,
    batch_size:     int,
    device:         str,
    n_eval_batches: int = 50,
    random_targets: bool = False,
    vocab_size:     int = 0,
) -> tuple[float, float]:
    """Estimate validation perplexity. Returns (val_ppl, val_loss)."""
    model.eval()
    rng        = np.random.default_rng(0)
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(n_eval_batches):
            batch = sample_batch(val_ids, seq_len, batch_size, rng, device)
            if random_targets:
                labels = batch.clone()
                # Shuffle targets: flatten, permute, reshape
                flat = labels.view(-1)
                perm = flat[torch.randperm(flat.numel())]
                labels = perm.view_as(labels)
            else:
                labels = batch
            out        = model(batch, labels=labels)
            total_loss += out["loss"].item()
    model.train()
    avg_loss = total_loss / n_eval_batches
    return math.exp(avg_loss), avg_loss


def train(
    model_type:     str,
    size:           str,
    train_ids:      np.ndarray,
    val_ids:        np.ndarray,
    vocab_size:     int,
    n_steps:        int,
    batch_size:     int,
    seq_len:        int,
    device:         str,
    output_csv:     Path,
    log_every:      int = 200,
    random_targets: bool = False,
    debug_samples:  int = 0,
    tok=None,
) -> float:
    """Train model and write learning curves to CSV. Returns final val_perplexity."""
    model    = build_model(model_type, size, vocab_size, seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{model_type.upper()}/{size}] params={n_params:,}  vocab={vocab_size}  seq_len={seq_len}")

    if random_targets:
        # Shuffled targets preserve the marginal (unigram) distribution.
        # Models should converge to the unigram entropy H(p), NOT log(vocab).
        # Compute unigram entropy from training data.
        from collections import Counter
        counts = Counter(train_ids.tolist())
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        unigram_entropy = -sum(p * math.log(p) for p in probs)
        expected_loss = unigram_entropy
        print(
            f"[random-targets] ON -- expected converged loss = H_unigram = {expected_loss:.4f}"
            f"  (log({vocab_size}) = {math.log(vocab_size):.4f})"
        )

    param_groups = model.configure_optimizer_param_groups(weight_decay=0.1)
    optimizer    = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95))

    warmup_steps = 200

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = (device == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    fieldnames = [
        "step", "train_loss", "val_loss", "val_perplexity", "tokens_per_sec", "peak_mb",
        "grad_norm_unclipped", "grad_norm_clipped",
    ]

    def _log_alpha_stats(model: nn.Module, step: int) -> None:
        """Log mean/p25/p75 of sigmoid(gate_bias) per layer."""
        for name, param in model.named_parameters():
            if "proj.bias" in name:
                K = param.shape[0] // 4
                gate_bias = param.data[3 * K:]
                alpha_vals = torch.sigmoid(gate_bias)
                mean = alpha_vals.mean().item()
                p25 = alpha_vals.quantile(0.25).item()
                p75 = alpha_vals.quantile(0.75).item()
                layer = name.split(".")[1] if "blocks" in name else "?"
                print(f"    [alpha] layer={layer} mean={mean:.3f} p25={p25:.3f} p75={p75:.3f}")

    def _compute_grad_norm(model: nn.Module) -> float:
        """Compute total grad norm (unclipped) before clip_grad_norm_."""
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.data.float().norm().item() ** 2
        return total ** 0.5

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        loss_acc      = 0.0
        tokens_acc    = 0
        t0            = time.perf_counter()
        final_val_ppl = float("nan")
        grad_norms_unclipped: list[float] = []

        for step in range(1, n_steps + 1):
            batch = sample_batch(train_ids, seq_len, batch_size, rng, device)

            if random_targets:
                labels = batch.clone()
                flat = labels.view(-1)
                perm = flat[torch.randperm(flat.numel(), device=device)]
                labels = perm.view_as(labels)
            else:
                labels = batch

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    out  = model(batch, labels=labels)
                    loss = out["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn_unclipped = _compute_grad_norm(model)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out  = model(batch, labels=labels)
                loss = out["loss"]
                loss.backward()
                gn_unclipped = _compute_grad_norm(model)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            loss_acc   += loss.item()
            tokens_acc += batch_size * seq_len
            grad_norms_unclipped.append(gn_unclipped)

            if step % log_every == 0 or step == n_steps:
                t1      = time.perf_counter()
                elapsed = t1 - t0
                tps     = tokens_acc / elapsed if elapsed > 0 else 0.0

                peak_mb = (
                    torch.cuda.max_memory_allocated(device) / 1e6
                    if device == "cuda" else 0.0
                )

                avg_loss      = loss_acc / log_every
                val_ppl, val_loss = eval_val_perplexity(
                    model, val_ids, seq_len, batch_size, device,
                    random_targets=random_targets, vocab_size=vocab_size,
                )
                final_val_ppl = val_ppl

                # Grad norm stats for the interval
                recent_gn = grad_norms_unclipped[-log_every:]
                gn_median = sorted(recent_gn)[len(recent_gn) // 2]
                gn_max = max(recent_gn)

                rt_tag = " [RANDOM-TARGETS]" if random_targets else ""
                print(
                    f"  [{model_type.upper()}/{size}] step={step:5d}/{n_steps}"
                    f"  train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}"
                    f"  tps={tps:.0f}  peak_mb={peak_mb:.1f}"
                    f"  gn_median={gn_median:.1f}  gn_max={gn_max:.1f}"
                    f"{rt_tag}"
                )

                if random_targets:
                    diff = abs(val_loss - expected_loss)
                    status = "OK" if diff < 0.1 else "WARN"
                    print(
                        f"    [{status}] val_loss={val_loss:.4f} vs H_unigram={expected_loss:.4f}"
                        f"  diff={diff:.4f}"
                    )

                # Alpha diagnostics every log interval for TRN
                if model_type == "trn" and step <= 1000:
                    _log_alpha_stats(model, step)

                # Debug sample logging
                if debug_samples > 0 and tok is not None:
                    dbg_batch = sample_batch(val_ids, seq_len, min(debug_samples, batch_size), rng, device)
                    log_debug_samples(
                        model, dbg_batch, dbg_batch, step, debug_samples, tok, model_type,
                    )

                writer.writerow({
                    "step":              step,
                    "train_loss":        f"{avg_loss:.6f}",
                    "val_loss":          f"{val_loss:.6f}",
                    "val_perplexity":    f"{val_ppl:.4f}",
                    "tokens_per_sec":    f"{tps:.1f}",
                    "peak_mb":           f"{peak_mb:.1f}",
                    "grad_norm_unclipped": f"{gn_median:.2f}",
                    "grad_norm_clipped":   f"{1.0:.2f}",
                })

                loss_acc   = 0.0
                tokens_acc = 0
                t0 = t1

    print(f"  -> Saved: {output_csv}")
    return final_val_ppl


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TRN/Transformer LM on WikiText-2")
    parser.add_argument("--model",      choices=["trn", "tf"], default="trn",
                        help="Model type: trn or tf (default: trn)")
    parser.add_argument("--size",       choices=["small", "medium", "large"], default="small",
                        help="Model size preset (default: small)")
    parser.add_argument("--steps",      type=int,   default=2000,
                        help="Gradient steps (default: 2000)")
    parser.add_argument("--batch-size", type=int,   default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--seq-len",    type=int,   default=256,
                        help="Sequence length (default: 256)")
    parser.add_argument("--device",     type=str,   default="cpu",
                        help="Device: cpu or cuda (default: cpu)")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--log-every",  type=int,   default=200,
                        help="Log interval in steps (default: 200)")
    parser.add_argument("--tokenizer",  choices=["char", "bpe"], default="char",
                        help="Tokenizer type: char or bpe (default: char)")
    parser.add_argument("--bpe-vocab-size", type=int, default=8192,
                        help="BPE vocabulary size (default: 8192)")
    parser.add_argument("--random-targets", action="store_true",
                        help="Shuffle targets randomly (sanity check)")
    parser.add_argument("--debug-samples", type=int, default=0,
                        help="Print N decoded input/target/prediction samples each eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU")
        device = "cpu"

    tok_label = args.tokenizer
    rt_label  = "_randtargets" if args.random_targets else ""

    print(f"[config] model={args.model}  size={args.size}  steps={args.steps}"
          f"  batch={args.batch_size}  seq_len={args.seq_len}  device={device}"
          f"  tokenizer={tok_label}  random_targets={args.random_targets}"
          f"  debug_samples={args.debug_samples}")

    # Load data based on tokenizer choice
    if args.tokenizer == "bpe":
        train_ids, val_ids, tok = load_or_build_bpe_cache(
            bpe_vocab_size=args.bpe_vocab_size,
        )
        vocab_size = tok.vocab_size
    else:
        train_ids, val_ids, tok = load_or_build_cache()
        vocab_size = tok.vocab_size

    suffix = f"{tok_label}{rt_label}"
    output_csv = (
        ROOT / "scripts" / "results"
        / f"train_lm_realdata_{args.model}_{args.size}_{suffix}_curves.csv"
    )

    print(f"\n=== Training {args.model.upper()} / {args.size} ({args.steps} steps, {tok_label}) ===")
    if args.random_targets:
        expected_loss = math.log(vocab_size)
        print(f"=== RANDOM-TARGETS MODE: expected loss = log({vocab_size}) = {expected_loss:.4f} ===")

    val_ppl = train(
        model_type=args.model,
        size=args.size,
        train_ids=train_ids,
        val_ids=val_ids,
        vocab_size=vocab_size,
        n_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
        output_csv=output_csv,
        log_every=args.log_every,
        random_targets=args.random_targets,
        debug_samples=args.debug_samples,
        tok=tok,
    )

    val_loss = math.log(val_ppl)
    print(f"\n{'='*60}")
    print(f"  Model:          {args.model.upper()} / {args.size}")
    print(f"  Tokenizer:      {tok_label} (vocab={vocab_size})")
    print(f"  Val Loss:       {val_loss:.4f}")
    print(f"  Val Perplexity: {val_ppl:.2f}")
    if args.random_targets:
        # Compute unigram entropy for the correct baseline
        from collections import Counter
        counts = Counter(train_ids.tolist())
        total_count = sum(counts.values())
        probs = [c / total_count for c in counts.values()]
        unigram_entropy = -sum(p * math.log(p) for p in probs)
        diff = abs(val_loss - unigram_entropy)
        verdict = "PASS" if diff < 0.1 else ("MARGINAL" if diff < 0.3 else "FAIL")
        print(f"  Expected Loss:  {unigram_entropy:.4f}  (H_unigram, log({vocab_size})={math.log(vocab_size):.4f})")
        print(f"  Difference:     {diff:.4f}")
        print(f"  Verdict:        {verdict}")
    print(f"  CSV:            {output_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
