# Knowledge Distillation

Transfer learned representations from a pretrained HuggingFace Transformer teacher to a TRN student model.

## Requirements

```bash
pip install -e ".[data]"
pip install transformers
```

## Quick Start

```bash
cd scripts

# Quick smoke test (< 5 min, CPU)
python distill_lm.py --quick --device cpu

# Full run (GPU, 100k steps)
python distill_lm.py --student-size 100m --teacher gpt2 --steps 100000 --device cuda
```

## Commands

### Distillation Training

```bash
python scripts/distill_lm.py \
  --student-size small|100m|400m|1b \
  --teacher gpt2 \
  --steps 100000 \
  --batch-size 8 \
  --seq-len 256 \
  --lr 3e-4 \
  --warmup 1000 \
  --device cuda \
  --seed 42 \
  --temperature 2.0 \
  --kl-weight 1.0 \
  --ce-weight 0.1
```

Output: `scripts/results/distill_{student}_{teacher}_curves.csv`

### CE-Only Baseline

```bash
python scripts/distill_lm.py --kl-weight 0.0 --ce-weight 1.0 --steps 2000
```

### Teacher Logit Caching

For small datasets, cache teacher logits to avoid repeated forward passes:

```bash
python scripts/distill_lm.py --cache-teacher-logits --steps 5000
```

## Verification Suite

Run all verification scripts from `scripts/`:

```bash
cd scripts

# Overfit microset (PASS: loss < 0.5)
python verify_overfit_microset.py --steps 2000 --device cpu

# Random-target control (PASS: val_loss within 0.5 nats of H(p))
python verify_random_targets.py --steps 500 --device cpu

# Teacher ablation (PASS: distill ppl < CE-only ppl by 10%+)
python verify_teacher_ablation.py --steps 2000 --device cpu

# Determinism test
cd .. && pytest tests/test_distill_determinism.py -v
```

## Stage-3 Gate Checklist

| Check | Criterion | Script |
|-------|-----------|--------|
| Random-target control | val_loss within 0.5 nats of H(p) | verify_random_targets.py |
| Teacher ablation | distill val_ppl < CE-only val_ppl by 10%+ | verify_teacher_ablation.py |
| Overfit microset | train_loss < 0.5 on 2k tokens | verify_overfit_microset.py |
| Determinism | identical losses at same seed | test_distill_determinism.py |
| Grad stability | median grad norm < 1e4, no NaN/Inf | distill_lm.py logs |
| Val ppl trend | decreasing over training | distill_lm.py CSV |

## Architecture

```
Teacher (frozen HF causal LM)
    |
    |-- logits --> KL(student_soft || teacher_soft, T) * T^2
    |
Student (TRNModel)
    |-- logits --> CE(student, hard_labels)
    |
    loss = kl_weight * KL + ce_weight * CE
```

## Student Presets

| Preset | d_model | n_layers | n_osc | d_ff | ~params |
|--------|---------|----------|-------|------|---------|
| small  | 128     | 4        | 64    | 512  | ~0.95M  |
| 100m   | 512     | 8        | 256   | 2048 | ~100M   |
| 400m   | 1024    | 16       | 512   | 4096 | ~400M   |
| 1b     | 2048    | 24       | 512   | 8192 | ~1B     |

## CSV Columns

`step, train_loss, kl_loss, ce_loss, val_loss, val_ppl, tps, peak_mb, grad_norm, lr`
