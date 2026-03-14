# Release Checklist

Steps to regenerate all artifacts and verify the Go/No-Go gate before a release.

---

## Prerequisites

```bash
pip install -e .
python -c "import trn; print('OK')"
python -c "import torch; print(torch.__version__)"
```

Required Python packages: `torch >= 2.1`, `numpy`, `tqdm`.

---

## 1. Generate Transformer Reference Data

```bash
python scripts/gen_copy_tf_reference.py --device cpu --seed 42
```

Output: `results/go_nogo_copy_tf.csv`

This must run before Go/No-Go evaluation. If the file exists and has not changed,
this step can be skipped.

---

## 2. Run Benchmarks

### 2a. Agent History (CPU)

```bash
python scripts/bench_agent_history.py --checkpoints 1000,5000,10000 --device cpu
```

Output: `results/bench_agent_history.csv`

### 2b. Long Context Scaling (CPU)

```bash
python scripts/bench_kv_vs_trn.py --context-lens 512,1024,2048,4096 --device cpu
```

Output: `results/bench_kv_vs_trn.csv`

### 2c. Needle-in-Haystack / PPD / Goal Tracking (CPU)

```bash
python scripts/bench_needle_haystack.py --device cpu
python scripts/bench_needle_haystack.py --device cpu --backend dual_w64
```

Output: `results/bench_needle_haystack.csv`

### 2d. GPU Benchmark (requires CUDA)

```bash
python scripts/bench_phase7_gpu.py \
    --models trn_100m,tf_kv \
    --context-lens 512,1024,2048,4096,8192 \
    --device cuda
```

Output: `artifacts/phase7/{timestamp}/` (results.json, summary.md, env.json, nvidia_smi.txt)

### 2e. DualMemoryEngine Benchmark (requires CUDA)

```bash
python scripts/bench_vllm_trn.py \
    --models dual_100m_w64,dual_100m_w256,tf_kv,trn_100m \
    --context-lens 1024,4096,16384 \
    --device cuda
```

Output: `artifacts/phase7/{timestamp}/dual/`

---

## 3. Run Go/No-Go Evaluation

### 3a. TRN Standalone

```bash
python scripts/eval_go_no_go.py --device cpu
```

Output:
- `results/eval_go_no_go.csv` -- all criteria with values
- `results/gate_result.json` -- verdict + input file SHA-256 hashes
- `results/gate_result.md` -- human-readable verdict

Expected verdict: **GO** (T1: 6/6 PASS)

### 3b. DualMemoryEngine

```bash
python scripts/eval_go_no_go.py --device cpu --backend dual
```

Output:
- `results/eval_go_no_go_dual.csv`
- `results/gate_result_dual.json`
- `results/gate_result_dual.md`

Expected verdict: **CONDITIONAL_GO** (T1: 7/7 PASS, T2: 3 FAIL = known limitations)

T2 failures are expected and documented:
- `dual_nih_long_range`: NiH recall = 0.0 (structural limitation)
- `gt_window_recovery`: GT accuracy = ~0.25 at distance > W (chance level)
- `gt_reversal_recovery`: GT reversal = ~0.26 at distance > W (chance level)

---

## 4. Run Tests

```bash
pytest tests/ -v --timeout=300
```

Expected: 277/277 passing.

---

## 5. Verify Artifacts

### File Existence Check

```bash
ls results/gate_result.json
ls results/gate_result.md
ls results/gate_result_dual.json
ls results/gate_result_dual.md
ls results/eval_go_no_go.csv
ls results/bench_agent_history.csv
ls results/bench_needle_haystack.csv
```

### Verdict Check

```bash
python -c "import json; d=json.load(open('results/gate_result.json')); print(d['verdict'])"
# Expected: GO

python -c "import json; d=json.load(open('results/gate_result_dual.json')); print(d['verdict'])"
# Expected: CONDITIONAL_GO
```

### Hash Reproducibility

The `gate_result.json` files contain SHA-256 hashes of all input CSV files.
Regenerating benchmarks will produce new hashes. Verify that the hashes in
`gate_result.json` match the current files:

```bash
python -c "
import hashlib, json
gate = json.load(open('results/gate_result.json'))
for fname, expected_hash in gate['file_hashes'].items():
    try:
        actual = hashlib.sha256(open(f'results/{fname}', 'rb').read()).hexdigest()
        status = 'OK' if actual == expected_hash else 'MISMATCH'
    except FileNotFoundError:
        status = 'MISSING'
    print(f'{fname}: {status}')
"
```

---

## 6. Documentation Verification

Verify that all claims in documentation match measured results:

| Document | Key Claims to Verify |
|----------|---------------------|
| `README.md` | Throughput table matches `bench_agent_history.csv` |
| `README.md` | State sizes match `TRNConfig` formula |
| `README.md` | Go/No-Go verdicts match `gate_result*.md` |
| `BENCHMARK.md` | Measurement conditions match script defaults |
| `docs/PUBLIC_CLAIMS.md` | All evidence file references are valid |
| `docs/TRN_LIMITATIONS.md` | NiH, GT, PPD numbers match `bench_needle_haystack.csv` |

### Prohibited Words Check

```bash
grep -rn "revolutionary\|game-changer\|breakthrough\|infinite context\|replaces Transformer\|replaces attention\|production-ready\|zero memory" \
    README.md BENCHMARK.md docs/PUBLIC_CLAIMS.md docs/TRN_LIMITATIONS.md
# Expected: 0 matches outside of "Prohibited" sections in PUBLIC_CLAIMS.md
```

---

## 7. Claim Verification

Before publishing any document that references TRN performance:

- [ ] Check `docs/PUBLIC_CLAIMS.md` -- is the claim in the "Permitted" table?
- [ ] Premises stated -- every "Nx" ratio includes model config, T, and head config
- [ ] Selective copy limitation mentioned (8.8% accuracy)
- [ ] NiH limitation mentioned (0.0 recall)
- [ ] "Not a Transformer replacement" stated in README and any public summary
- [ ] DualMemoryEngine limitations documented (content recall fails beyond W)

---

## 8. Comparison Fairness

- [ ] TRN and TF use identical config (d_model, n_layers, d_ff, vocab_size)
- [ ] Same optimizer, LR schedule, batch size, seed
- [ ] Same dataset split and evaluation protocol
- [ ] TF uses KV cache for decode (not full-recompute) in throughput comparisons

---

## 9. Pre-Commit Checklist

- [ ] `pytest tests/ -v` -- 277/277 passing
- [ ] `python scripts/eval_go_no_go.py --device cpu` -- GO
- [ ] `python scripts/eval_go_no_go.py --device cpu --backend dual` -- CONDITIONAL_GO
- [ ] `gate_result.json` hashes match current CSVs
- [ ] No `NaN` or `Inf` in any results CSV
- [ ] All evidence files referenced in `PUBLIC_CLAIMS.md` exist
- [ ] README.md throughput numbers match `bench_agent_history.csv`
- [ ] README.md state sizes match analytical formula
- [ ] No prohibited words in any document (see `PUBLIC_CLAIMS.md`)
- [ ] Git hash in `env.json` matches current HEAD (GPU benchmarks only)

---

## 10. File Checklist

Before `git tag` and release:

| File | Exists | Up to date |
|------|--------|------------|
| `README.md` | [ ] | [ ] |
| `BENCHMARK.md` | [ ] | [ ] |
| `docs/PUBLIC_CLAIMS.md` | [ ] | [ ] |
| `docs/TRN_LIMITATIONS.md` | [ ] | [ ] |
| `docs/RELEASE_CHECKLIST.md` | [ ] | [ ] |
| `results/gate_result.json` | [ ] | [ ] |
| `results/gate_result.md` | [ ] | [ ] |
| `results/gate_result_dual.json` | [ ] | [ ] |
| `results/gate_result_dual.md` | [ ] | [ ] |
| `results/eval_go_no_go.csv` | [ ] | [ ] |
| `results/go_nogo_copy_tf.csv` | [ ] | [ ] |
| `results/bench_needle_haystack.csv` | [ ] | [ ] |
| `artifacts/phase7/*/env.json` | [ ] | [ ] |
| `artifacts/phase7/*/results.json` | [ ] | [ ] |
| `LICENSE` | [ ] | [ ] |

---

## 11. Pre-Push

```bash
# Lint
ruff check src/ scripts/ --select E,W,F,I

# Tests
pytest tests/ -x -q

# No secrets
grep -rn "api_key\|password\|secret" src/ scripts/ --include="*.py" | grep -v "test_\|#\|TODO"

# Git status clean
git status
git diff --stat
```

---

## Reproducing from Scratch

Full reproduction sequence (CPU-only, no GPU required):

```bash
git clone <repo-url> && cd trn
pip install -e .

# 1. Generate TF reference
python scripts/gen_copy_tf_reference.py --device cpu --seed 42

# 2. Run agent history benchmark
python scripts/bench_agent_history.py --device cpu

# 3. Run information retention
python scripts/bench_needle_haystack.py --device cpu

# 4. Run Go/No-Go
python scripts/eval_go_no_go.py --device cpu

# 5. Verify
cat results/gate_result.md
pytest tests/ -v
```

Total time: approximately 30--60 minutes on a modern CPU.
