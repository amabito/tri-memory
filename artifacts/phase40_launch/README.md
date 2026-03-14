# TriMemory + VERONICA

Documents change. Policies get superseded. Models answer anyway.
TriMemory resolves what is current. VERONICA decides whether the system may act on it.

Together: **governed knowledge execution** for stateful knowledge work.

---

## What it does

Most retrieval pipelines hand raw documents to a model and hope. That breaks when documents conflict, when authority matters, or when the model should refuse to answer.

TriMemory is a knowledge-state compiler. It resolves currentness, authority, and conflict before inference. VERONICA is a runtime containment layer. It blocks unsafe action after inference.

On IT security policy QA (PolicyBench, N=10), a 3B model with both layers matches a 32B reasoning baseline on current-value accuracy, at 26x lower cost and 8.5x lower latency -- with zero unsafe overclaims versus 40% for the baseline.

---

## Architecture

```
Documents (approved, draft, superseded, amended)
    |
    v
TriMemory Pipeline
  MetadataParser -> MemoryMediator -> Memory Compiler v6
  -> Schema Adaptation -> Compiled Knowledge State
    |
    +---> LLM (any model, any size)
    |
    +---> VERONICA Governance
            |
            v
          ALLOW | ABSTAIN | ESCALATE | BLOCK_ACTION
```

Schema adaptation: ~96 lines per domain.

---

## Why not just use a bigger model?

A larger model answers more fluently from stale context. It does not know which document is current policy versus a superseded draft. It cannot distinguish an authoritative source from an informal one. It has no runtime mechanism to block unsafe claims.

TriMemory handles knowledge-state resolution before inference. VERONICA handles containment after inference. Adding parameters does not address either problem.

On PolicyBench, the 32B reasoning baseline (deepseek-r1:32b) produces unsafe overclaims at a 0.400 rate. The 3B + TriMemory + VERONICA configuration produces 0.000. Authority accuracy: 1.000 versus 0.500. On this benchmark, those gaps correlate with the presence or absence of authority resolution and containment -- not with model size alone.

The 32B baseline has known JSON parse instability in reasoning mode, which may contribute to its safety scores. Results are N=10 on a single domain.

---

## Benchmark Snapshot

PolicyBench -- IT security policy QA -- N=10

| Configuration | CurrVal | Authority | Unsafe Overclaim | Cost/query | Latency | Safe outcomes/$ |
|---|---|---|---|---|---|---|
| 3B + TriMemory + VERONICA | 0.420 | 1.000 | 0.000 | $0.00006 | 2.8s | 16,294 |
| 7B + TriMemory + VERONICA | 0.380 | [1] | 0.000 | $0.00011 | 3.2s | ~8,600 |
| 32B reasoning baseline | 0.400 | 0.500 | 0.400 | $0.0016 | 23.4s | 312 |

[1] 7B Authority not measured in this benchmark run.

N=10, single domain (IT security policy). 32B baseline is deepseek-r1:32b, a reasoning model with documented JSON parse instability. Hard-case subset: 32B = 0.500, 3B = 0.429. Schema adaptation required per domain (~96 lines).

---

## What this is / what this is not

| This is | This is not |
|---|---|
| A knowledge-state compiler that resolves currentness, authority, and conflict | A general-purpose RAG framework |
| A runtime containment layer that blocks unsafe claims | A model fine-tuning approach |
| Designed for stateful knowledge work: compliance, policy QA, procedure following | A solution for open-domain QA or creative tasks |
| A lightweight augmentation path for small models on specific verticals | A drop-in replacement for reasoning models on all tasks |
| A benchmark (PolicyBench) for authority-aware, containment-checked policy QA | A multi-domain evaluation suite |
| A specific result on N=10 IT security policy QA | Evidence that 3B > 32B in general |

---

## Demo

PolicyBench runs a policy document set through TriMemory's knowledge-state compiler. The compiler identifies the authoritative, current version of each policy, resolves conflicts between drafts, and builds a resolved context for the inference model.

A 3B model answers each question against that context. VERONICA inspects the output for unsafe overclaims and blocks responses that assert beyond what the resolved policy permits. The benchmark scores each answer on CurrVal (does the answer reflect current policy), Authority (does it cite the correct authoritative source), and Unsafe Overclaim (did the system make a claim that policy does not support).

---

## Installation

```bash
git clone https://github.com/[repo]/trimemory-veronica.git
cd trimemory-veronica
pip install -e .
```

Requires Python 3.10+ and an Ollama instance for local model inference. Package publication to PyPI is pending.

---

## Quick Start

Usage examples and API documentation are in progress. See the PolicyBench benchmark in the repository for a working end-to-end example.

---

## Limitations

- PolicyBench is N=10, IT security policy QA. Results may not generalize to other domains or larger sample sizes.
- Hard cases favor the 32B baseline (0.500 vs 0.429 on the hard-case subset).
- Schema adaptation is required per domain (~96 lines). Porting to a new domain requires authoring equivalent schema definitions.
- VERONICA containment rules were calibrated against this benchmark. Other domains require separate calibration.
- The 32B baseline (deepseek-r1:32b) has JSON parse instability in reasoning mode. Some baseline scores are derived from partial parses, which may understate baseline performance.
- Cost and latency figures assume specific hardware and API pricing as of benchmark date.

---

## Repository Structure

See the repository for current directory layout. Documentation is co-located with source code.

---

## Citation

If you use TriMemory, VERONICA, or PolicyBench in your work, please cite:

Citation format will be provided with the preprint. In the meantime, please reference the repository URL.

---

## License

MIT
