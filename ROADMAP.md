# Roadmap

## Where things stand

Retrieval path works. Hidden-state search finds gold chunks 29% better than bag-of-token. Copy-mix integration confirmed at token level. Full model beats individual components in composite score under specific settings.

TRN pattern learning shows up at 3000 steps but seed dependence is too high. Decoder/mixer is the current bottleneck -- even correct chunks don't produce correct tokens through the existing projection.

---

## Phase 0 -- Architecture Proof (done)

Built the 4-way comparison system (KV / KV+TRN / KV+Ret / Full). Confirmed each path has a role. Copy-mix validated as the retrieval integration method.

- [x] Retrieval role validated
- [x] Full model > retrieval-only in key settings
- [x] TRN learning trajectory confirmed at 3000 steps

## Phase 1 -- Stabilize Retrieval Path (mostly done)

- [x] Hidden-state retrieval
- [x] Marker query for chunk selection
- [x] Token matrix copy head
- [x] Additive copy-mix
- [ ] Copy-mix as default integration
- [ ] Alpha / chunk selection / copy metrics in standard logs
- [ ] Re-evaluate token-wise gating

Gold containment: 0.323 (bag) -> 0.415 (hidden). Type A failures: 67.9% -> 58.5%.

The search side is in reasonable shape. The remaining gap is on the decode side.

## Phase 2 -- Fix Decoder/Mixer (next)

Current problem: decode success = 0.000 across all search modes. The ret_proj and mixer representation can't convert retrieved hidden states into correct output tokens.

Three things to try:

1. Check if ret_proj output actually reaches logit space
2. Add auxiliary head for direct retrieval-to-logit projection
3. Training-time retrieval ground truth supervision

This is the single biggest blocker right now. Search works. Gating works (Type B is only 17%). The model just can't use what it found.

## Phase 3 -- Stabilize TRN Path

Pattern learning appears at 3000 steps but seed dependence is high. B standalone sometimes doesn't fire.

- [ ] Shorter pattern period (5 -> 3)
- [ ] Higher pattern loss weight
- [ ] Episode ratio tuning
- [ ] Seed firing rate comparison

Target: `pattern_B_minus_A > 0` reproducibly.

## Phase 4 -- Main Benchmark Confirmation

4-way benchmark with 5 seeds. All metrics: old_fact, pattern, salient, recent, composite.

Success means:
- `old_fact_C_minus_A > 0`
- `pattern_B_minus_A > 0`
- `D_minus_maxABC > 0`

## Phase 5 -- Speed

Chunked TRN scan gave ~2x speedup. bf16 ruled out for toy config. batch=128 works.

- [ ] Standardize batch=128
- [ ] Copy loss batching
- [ ] Profile at larger config

Not doing serving optimization yet.

## Phase 6 -- OSS Packaging

- [ ] README / ROADMAP finalized
- [ ] Architecture diagram
- [ ] Reproducibility section
- [ ] Clean config entrypoints
- [ ] Example commands
- [ ] License / contribution policy

## Phase 7 -- Scale Validation

Everything so far is toy-scale (1--100M params). Need to check if the architecture holds at larger configs.

- [ ] Bigger d_model / more layers
- [ ] Longer context
- [ ] Harder old_fact space
- [ ] bf16 retest at scale

---

## Right now

1. Fix decoder/mixer (Phase 2)
2. TRN stability (Phase 3)
3. Benchmark confirmation (Phase 4)

## Not now

- Serving optimization
- 1B+ training
- SaaS anything
