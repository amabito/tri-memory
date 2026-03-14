"""Adversarial tests for generate.py — attacker mindset.

These tests verify robustness against corrupted input, boundary abuse,
and state correctness issues that happy-path tests miss.
"""
from __future__ import annotations

import torch

from trimemory.config import TRNConfig
from trimemory.model import TRNModel
from trimemory.generate import (
    generate,
    stream_generate,
    sample_token,
    _apply_top_p,
    _apply_repetition_penalty,
    GenerationConfig,
)


def _toy_cfg() -> TRNConfig:
    return TRNConfig.toy()


def _toy_model() -> TRNModel:
    torch.manual_seed(0)
    model = TRNModel(_toy_cfg())
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Corrupted input tests
# ---------------------------------------------------------------------------


class TestCorruptedInput:
    """Adversarial tests for corrupted / invalid inputs."""

    def test_top_p_zero_no_crash(self) -> None:
        """_apply_top_p(top_p=0) must not crash and must return finite logits."""
        logits = torch.randn(1, 100)
        result = _apply_top_p(logits.clone(), top_p=0)
        assert not torch.isnan(result).any(), "NaN found in output"
        # At least one finite value must exist (top-1 is kept)
        finite_mask = result != float("-inf")
        assert finite_mask.any(), "All tokens filtered to -inf"

    def test_top_p_negative_no_crash(self) -> None:
        """_apply_top_p(top_p=-1.0) must not crash and must return finite logits."""
        logits = torch.randn(1, 100)
        result = _apply_top_p(logits.clone(), top_p=-1.0)
        assert not torch.isnan(result).any(), "NaN found in output"
        finite_mask = result != float("-inf")
        assert finite_mask.any(), "All tokens filtered to -inf"

    def test_nan_logits_in_sample_token(self) -> None:
        """logits with NaN positions must not return NaN token id silently.

        Either raise an exception (acceptable) or return a valid int.
        """
        logits = torch.full((1, 100), float("nan"))
        logits[0, 42] = 10.0  # one valid position
        cfg = GenerationConfig(do_sample=False)
        try:
            result = sample_token(logits, cfg)
            # If it succeeds, result must be a valid integer token id
            assert isinstance(result.item(), int), "Result must be an int"
            assert not torch.isnan(result.float()), "Result must not be NaN"
        except (RuntimeError, ValueError):
            pass  # Raising is acceptable for NaN input

    def test_inf_logits_valid_sample(self) -> None:
        """All-inf logits except one position must always sample that position."""
        logits = torch.full((1, 100), float("-inf"))
        logits[0, 7] = 0.0
        cfg = GenerationConfig(do_sample=False)
        result = sample_token(logits, cfg)
        assert result.item() == 7, f"Expected 7, got {result.item()}"

    def test_temperature_near_zero_no_nan(self) -> None:
        """temperature=1e-7 must not produce NaN token id."""
        torch.manual_seed(0)
        logits = torch.randn(1, 100)
        cfg = GenerationConfig(temperature=1e-7, do_sample=True)
        result = sample_token(logits, cfg)
        assert not torch.isnan(result.float()), "NaN token id from near-zero temperature"
        assert 0 <= result.item() < 100, "Token id out of vocab range"

    def test_repetition_penalty_one_no_change(self) -> None:
        """penalty=1.0 must leave logits exactly unchanged."""
        logits = torch.randn(1, 100)
        input_ids = torch.tensor([[5, 10, 20]])
        result = _apply_repetition_penalty(logits.clone(), input_ids, penalty=1.0)
        assert torch.allclose(logits, result), "penalty=1.0 must not modify logits"


# ---------------------------------------------------------------------------
# Boundary abuse tests
# ---------------------------------------------------------------------------


class TestBoundaryAbuse:
    """Tests for edge cases at parameter boundaries."""

    def test_top_p_exactly_one_no_filter(self) -> None:
        """top_p=1.0 must return logits unchanged."""
        logits = torch.randn(1, 100)
        result = _apply_top_p(logits.clone(), top_p=1.0)
        assert torch.equal(logits, result), "top_p=1.0 must be a no-op"

    def test_top_k_one_always_argmax(self) -> None:
        """top_k=1 with do_sample=True must always pick the argmax token."""
        torch.manual_seed(0)
        logits = torch.randn(1, 100)
        argmax = logits.argmax(dim=-1).item()
        cfg = GenerationConfig(top_k=1, do_sample=True)
        for _ in range(10):
            result = sample_token(logits.clone(), cfg)
            assert result.item() == argmax, f"top_k=1 must pick argmax={argmax}, got {result.item()}"

    def test_top_k_larger_than_vocab(self) -> None:
        """top_k=99999 with vocab=100 must not crash."""
        logits = torch.randn(1, 100)
        cfg = GenerationConfig(top_k=99999, do_sample=True)
        result = sample_token(logits, cfg)
        assert 0 <= result.item() < 100, "Token id out of vocab range"

    def test_top_p_tiny_keeps_one_or_two(self) -> None:
        """top_p=1e-9 (very restrictive) must not crash; result must be valid."""
        logits = torch.randn(1, 100)
        cfg = GenerationConfig(top_p=1e-9, do_sample=True)
        result = sample_token(logits, cfg)
        assert 0 <= result.item() < 100, "Token id out of vocab range"

    def test_max_new_tokens_zero(self) -> None:
        """generate(max_new_tokens=0) must return shape (B, 0) without crash."""
        model = _toy_model()
        cfg = _toy_cfg()
        prompt = torch.randint(0, cfg.vocab_size, (2, 4))
        gen_cfg = GenerationConfig(max_new_tokens=0)
        out = generate(model, prompt, gen_cfg=gen_cfg)
        assert out.shape == (2, 0), f"Expected shape (2, 0), got {out.shape}"

    def test_prompt_len_one(self) -> None:
        """Prompt with single token must not crash and must produce valid output."""
        model = _toy_model()
        cfg = _toy_cfg()
        prompt = torch.randint(0, cfg.vocab_size, (2, 1))
        gen_cfg = GenerationConfig(max_new_tokens=5, do_sample=False)
        out = generate(model, prompt, gen_cfg=gen_cfg)
        assert out.shape == (2, 5), f"Expected shape (2, 5), got {out.shape}"
        assert (out >= 0).all(), "Negative token ids found"
        assert (out < cfg.vocab_size).all(), "Token ids exceed vocab size"

    def test_repetition_penalty_extreme(self) -> None:
        """penalty=100.0 must heavily penalize repeated tokens without NaN/inf."""
        logits = torch.zeros(1, 100)
        logits[0, 42] = 2.0  # positive logit for seen token
        input_ids = torch.tensor([[42]])
        result = _apply_repetition_penalty(logits.clone(), input_ids, penalty=100.0)
        assert not torch.isnan(result).any(), "NaN in output with extreme penalty"
        assert not torch.isinf(result).any(), "Inf in output with extreme penalty"
        assert result[0, 42] < logits[0, 42], "Extreme penalty must reduce positive logit"


# ---------------------------------------------------------------------------
# State correctness tests
# ---------------------------------------------------------------------------


class TestStateCorrectness:
    """Tests for state isolation and reproducibility."""

    def test_generate_state_reset_between_calls(self) -> None:
        """Two calls with same seed must produce identical results (no state leakage)."""
        model = _toy_model()
        cfg = _toy_cfg()
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        gen_cfg = GenerationConfig(max_new_tokens=5, do_sample=True, temperature=1.0)

        torch.manual_seed(0)
        out1 = generate(model, prompt, gen_cfg=gen_cfg)
        torch.manual_seed(0)
        out2 = generate(model, prompt, gen_cfg=gen_cfg)
        assert torch.equal(out1, out2), "Sequential calls with same seed must be identical"

    def test_all_uniform_logits_valid(self) -> None:
        """Uniform logits (zeros) must sample a valid token without crash."""
        logits = torch.zeros(1, 100)
        cfg = GenerationConfig(do_sample=True)
        for _ in range(20):
            result = sample_token(logits.clone(), cfg)
            assert 0 <= result.item() < 100, "Token id out of vocab range"

    def test_stream_single_token_prompt(self) -> None:
        """stream_generate with prompt_len=1 must yield exactly max_new_tokens tokens."""
        model = _toy_model()
        cfg = _toy_cfg()
        prompt = torch.randint(0, cfg.vocab_size, (1, 1))
        gen_cfg = GenerationConfig(max_new_tokens=5, do_sample=False)
        tokens = list(stream_generate(model, prompt, gen_cfg=gen_cfg))
        assert len(tokens) == 5, f"Expected 5 tokens, got {len(tokens)}"
        assert all(0 <= t < cfg.vocab_size for t in tokens), "Token ids out of vocab range"

    def test_generate_greedy_reproducible(self) -> None:
        """Greedy (do_sample=False) must be fully deterministic without seed."""
        model = _toy_model()
        cfg = _toy_cfg()
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        gen_cfg = GenerationConfig(max_new_tokens=8, do_sample=False)
        out1 = generate(model, prompt, gen_cfg=gen_cfg)
        out2 = generate(model, prompt, gen_cfg=gen_cfg)
        assert torch.equal(out1, out2), "Greedy must be deterministic"

    def test_stream_matches_batch_greedy(self) -> None:
        """stream_generate greedy must match batch generate greedy token-by-token."""
        model = _toy_model()
        cfg = _toy_cfg()
        torch.manual_seed(0)
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        gen_cfg = GenerationConfig(max_new_tokens=6, do_sample=False)

        batch_out = generate(model, prompt, gen_cfg=gen_cfg)  # (1, 6)
        stream_out = list(stream_generate(model, prompt, gen_cfg=gen_cfg))

        assert len(stream_out) == 6
        for step, tok in enumerate(stream_out):
            expected = batch_out[0, step].item()
            assert tok == expected, f"Step {step}: stream={tok} != batch={expected}"
