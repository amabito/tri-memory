# Governed Knowledge Execution: TriMemory + VERONICA, Open Source

A 3B local model -- paired with a knowledge-state compiler and a runtime containment layer -- scored higher on currency and authority than a 32B reasoning baseline on PolicyBench (N=10, IT security policy QA), at 26x lower cost per query. The code and benchmarks are available for inspection.

---

## What this is

Most agent memory work focuses on retrieval. TriMemory takes a different angle: it treats what an agent currently knows as a compiled artifact, resolved and audited before any output is produced. VERONICA adds a runtime containment layer that enforces policy constraints on what the agent may claim or act on at execution time.

Together they form **governed knowledge execution** -- a pairing of knowledge-state management with behavioral containment.

The target use case is **stateful knowledge work**: IT policy QA, compliance reasoning, enterprise knowledge bases -- domains where an agent's knowledge state degrades over time and overclaiming has real cost.

---

## Three points worth examining

**1. On structured policy tasks, smaller models with compiled knowledge state can match larger models without it.**

On PolicyBench (N=10, IT security policy QA), a 3B model with TriMemory + VERONICA scored 0.420 on Currency Validity vs 0.400 for the 32B reasoning baseline (deepseek-r1:32b). Authority score was 1.000 vs 0.500. On hard cases, the 32B baseline scored 0.500 and the 3B configuration scored 0.429 -- the gap narrows but does not close. The claim is not that smaller is universally better; it is that compiled knowledge state changes the comparison on specific tasks.

**2. Unsafe overclaiming dropped to zero with containment.**

The 32B baseline produced unsafe overclaims on 40% of queries. The 3B + TriMemory + VERONICA configuration produced zero. The 7B variant also produced zero. VERONICA's containment layer is the operative mechanism -- not model scale.

**3. The economics shift when you include safety outcomes.**

At $0.00006 per query vs $0.0016 for the 32B baseline, the 3B configuration delivers approximately 52x more safe outcomes per dollar. A 7B variant sits at 14.1x cost reduction with CurrVal 0.380 and zero overclaim -- a different tradeoff point depending on accuracy requirements.

---

## Benchmark summary

| Configuration | CurrVal | Authority | Unsafe Overclaim | Cost/query | Safe outcomes/$ |
|---|---|---|---|---|---|
| 3B + TM + V | 0.420 | 1.000 | 0.000 | $0.00006 | 16,294 |
| 7B + TM + V | 0.380 | -- | 0.000 | ~$0.00011 | ~8,600 |
| 32B baseline | 0.400 | 0.500 | 0.400 | $0.0016 | 312 |

Baseline: deepseek-r1:32b (known JSON parse issues observed during evaluation). N=10 PolicyBench, IT security policy QA. Schema: ~96 lines per domain. 7B Authority not measured in this evaluation run.

---

## Limitations

N=10 is a small sample on a single domain. The 32B baseline had JSON parse failures that may have depressed its structured output scores -- this is documented in the benchmark logs rather than corrected away. TriMemory requires schema authoring per domain, which is manual work. VERONICA containment policies require the same. Hard cases favor the 32B baseline (0.500 vs 0.429).

These results do not cover open-domain QA, creative tasks, or long-horizon planning. They are scoped to structured, policy-governed knowledge retrieval.

---

## Links

- Repository: `[REPO_URL]`
- Benchmark results and methodology: `[BENCHMARK_URL]`
- Design rationale (manifesto): `[MANIFESTO_URL]`

---

## Running it on your domain

The architecture is **MIT open source** and the evaluation pipeline is included in the repository. If you have a knowledge domain with a defined policy schema, you can run the benchmark against your own baseline in an afternoon. The schema format is documented.

This system requires upfront schema work that a raw language model does not. That cost is part of the tradeoff. If your use case is stateful knowledge work where overclaiming has consequences, the tradeoff may be worth examining. The benchmark and code are available -- run it and see what the numbers look like on your data.
