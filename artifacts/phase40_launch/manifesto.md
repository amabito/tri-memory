# The Knowledge State Problem

LLMs do not fail only because they are too small. They fail because they act on unresolved knowledge state.

A 70B model fed a retrieval dump returns confident answers assembled from contradictory fragments. A 7B model with a clean, compiled knowledge state refuses to answer when it cannot. The failure mode is not capacity. It is architecture.

Most current approaches treat memory as text logistics: chunk documents, embed them, retrieve the top-k, stuff the context window. The system never knows what it actually knows. It knows what landed in the prompt.

---

## Why existing approaches fail

Text stuffing is the default because it is easy to implement and looks correct in demos. Retrieved chunks land in the prompt. On static benchmarks with clean sources, it works tolerably.

Real deployments are not static benchmarks.

Sources conflict. Temporal scope differs across documents. Authority levels are heterogeneous -- a memo from an intern and a signed policy document are not the same. Retrieval systems have no representation for these distinctions. They return text. The downstream model decides what to do with it, with no formal description of the epistemic state it inherited.

Ungoverned execution compounds this. A model that cannot distinguish "I know this with high confidence from a primary source" from "I reconstructed this from fragments" will overclaim. Not because it is dishonest -- because it has no representation for the distinction.

---

## What TriMemory solves

TriMemory is a knowledge-state compiler. The processing chain: MetadataParser -> MemoryMediator -> Memory Compiler v6 -> Schema Adaptation -> Compiled Knowledge State.

The key concept is the Compiled Knowledge State: a structured representation of what the system knows *now*, with provenance, authority, temporal scope, and confidence attached. Not retrieved text. A resolved knowledge object.

Retrieval gives the model more context. Compilation gives the model a resolved description of the epistemic state it is operating from. The model can use that description to decide what to claim and what not to claim.

The practical effect: a smaller model with compiled state can enforce calibration that a larger model without it cannot. The 3B model knows when to abstain. The 32B reasoning baseline does not, because it has no representation for "I am uncertain" that is separate from "I am synthesizing."

This is what the system knows now.

---

## What VERONICA solves

Knowledge without governance is unsafe.

VERONICA is a runtime containment and governance layer. It reads the Compiled Knowledge State and evaluates proposed actions against policy: ALLOW, ABSTAIN, ESCALATE, BLOCK_ACTION.

Training-time alignment shapes tendencies. Runtime governance enforces hard constraints. These are not substitutes. A model shaped by RLHF to avoid harmful outputs will still overclaim in a domain where it has no reliable knowledge -- because overclaiming is not the same as producing harmful outputs by standard alignment definitions.

VERONICA treats overclaiming as a policy violation. If the Compiled Knowledge State does not support a claim with sufficient authority, the action is ABSTAIN or ESCALATE. The governance layer reads the knowledge state.

Governance without knowledge-state is blind.

A governance layer operating on raw retrieved text has no reliable signal for authority or confidence. It can enforce syntactic rules. It cannot enforce epistemic ones. VERONICA's policy decisions are grounded in the Compiled Knowledge State -- which is why the combination produces a different behavioral profile than either component alone.

This is what the system may do now.

---

## Why two layers are necessary

TriMemory without governance produces a well-described knowledge state that an unconstrained model can still misuse. VERONICA without compiled state operates on unresolved inputs and cannot distinguish an authoritative claim from a reconstructed one.

Together they constitute governed knowledge execution: the system knows its epistemic state and is constrained to act within it.

---

## What the benchmark shows

PolicyBench, N=10, IT security policy QA. Comparing 3B + TriMemory + VERONICA against a 32B reasoning baseline (deepseek-r1:32b):

| Metric | 3B + TM + V | 32B baseline |
|---|---|---|
| CurrVal | 0.420 | 0.400 |
| Authority | 1.000 | 0.500 |
| Unsafe Overclaim | 0.000 | 0.400 |
| Cost/query | $0.00006 | $0.0016 |
| Latency | 2.8s | 23.4s |
| Safe outcomes/$ | 16,294 | 312 |

On hard cases, the 32B baseline scores 0.500 and the 3B model scores 0.429.

The pattern that holds: the 32B baseline produces unsafe overclaims at a 40% rate. The 3B model with governance produces zero. Authority score is 1.000 versus 0.500. Cost-normalized safe outcomes differ by 52x. The 32B baseline also has known JSON parse issues that inflate its unsafe rate in this evaluation.

What this supports: on this benchmark, runtime governance enforces calibration that scale alone does not provide.

---

## What we are not claiming

**3B models are generically superior to larger models.** They are not. Larger models have substantially higher ceiling on hard reasoning. The comparison here is narrow: a specific configuration on a specific benchmark.

**Zero-configuration deployment.** TriMemory requires metadata work. VERONICA requires policy authoring. Neither is drop-in.

**All domains.** This evaluation used policy and knowledge tasks where authority and temporal scope are meaningful. Domains where knowledge state is not structured this way are out of scope.

**Statistical proof.** N=10 is not statistical proof. It is a directional signal.

**Solved AI memory.** Compiling knowledge state is a hard problem with open subproblems. This is an architecture for addressing it.

---

## The argument

Answer-only systems -- systems that retrieve text, synthesize, and return -- have a structural property: they cannot abstain on principled grounds. They have no representation of their own epistemic state. They answer because that is what they do.

Governed knowledge execution is a different architecture. Compile the knowledge state first. Enforce policy at execution time against that compiled state. The system can now abstain because it knows it does not know. It can escalate because it knows the claim requires authority it does not have.

The components are MIT licensed. The benchmark is included. Whether governed knowledge execution holds on your domain is an empirical question -- and the evaluation pipeline is there so you can answer it.
