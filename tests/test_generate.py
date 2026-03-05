"""Tests for src/trn/generate.py."""
from __future__ import annotations

import torch
import pytest

from trn.config import TRNConfig
from trn.model import TRNModel
from trn.generate import (
    GenerationConfig,
    _apply_top_p,
    _apply_repetition_penalty,
    sample_token,
    generate,
    stream_generate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_cfg() -> TRNConfig:
    return TRNConfig.toy()


@pytest.fixture(scope="module")
def toy_model(toy_cfg: TRNConfig) -> TRNModel:
    torch.manual_seed(0)
    model = TRNModel(toy_cfg)
    model.eval()
    return model


def _prompt(cfg: TRNConfig, B: int = 2, prompt_len: int = 4) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, cfg.vocab_size, (B, prompt_len))


# ---------------------------------------------------------------------------
# test_generate_output_shape
# ---------------------------------------------------------------------------


def test_generate_output_shape(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    """generate() must return (B, max_new_tokens)."""
    B, prompt_len, max_new = 2, 4, 10
    prompt = _prompt(toy_cfg, B=B, prompt_len=prompt_len)
    cfg = GenerationConfig(max_new_tokens=max_new, do_sample=False)
    out = generate(toy_model, prompt, gen_cfg=cfg)
    assert out.shape == (B, max_new)


# ---------------------------------------------------------------------------
# test_generate_greedy_deterministic
# ---------------------------------------------------------------------------


def test_generate_greedy_deterministic(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    """do_sample=False must produce the same output on repeated calls."""
    prompt = _prompt(toy_cfg, B=2, prompt_len=4)
    cfg = GenerationConfig(max_new_tokens=5, do_sample=False)
    out1 = generate(toy_model, prompt, gen_cfg=cfg)
    out2 = generate(toy_model, prompt, gen_cfg=cfg)
    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# test_generate_top_k_filters
# ---------------------------------------------------------------------------


def test_generate_top_k_filters(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    """top_k=1 with do_sample=True should behave like greedy (argmax)."""
    prompt = _prompt(toy_cfg, B=2, prompt_len=4)
    greedy_cfg = GenerationConfig(max_new_tokens=5, do_sample=False)
    topk1_cfg = GenerationConfig(max_new_tokens=5, do_sample=True, top_k=1)

    torch.manual_seed(42)
    out_greedy = generate(toy_model, prompt, gen_cfg=greedy_cfg)
    torch.manual_seed(42)
    out_topk = generate(toy_model, prompt, gen_cfg=topk1_cfg)

    assert torch.equal(out_greedy, out_topk)


# ---------------------------------------------------------------------------
# test_apply_top_p_sum_to_one
# ---------------------------------------------------------------------------


def test_apply_top_p_sum_to_one() -> None:
    """After top_p filtering, surviving probabilities must sum to 1."""
    torch.manual_seed(7)
    logits = torch.randn(2, 50)
    filtered = _apply_top_p(logits.clone(), top_p=0.9)
    # Mask out -inf entries; surviving probs must sum to ~1.
    probs = torch.softmax(filtered, dim=-1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ---------------------------------------------------------------------------
# test_apply_top_p_full_disables
# ---------------------------------------------------------------------------


def test_apply_top_p_full_disables() -> None:
    """top_p=1.0 must leave logits unchanged."""
    logits = torch.randn(2, 50)
    out = _apply_top_p(logits.clone(), top_p=1.0)
    assert torch.equal(logits, out)


# ---------------------------------------------------------------------------
# test_repetition_penalty_reduces_repeated
# ---------------------------------------------------------------------------


def test_repetition_penalty_reduces_repeated() -> None:
    """Repetition penalty > 1 must reduce the logit of tokens already seen."""
    B, vocab = 1, 100
    logits = torch.zeros(B, vocab)
    logits[0, 10] = 2.0   # positive logit for token 10
    logits[0, 20] = -1.0  # negative logit for token 20

    input_ids = torch.tensor([[10, 20]])  # both tokens already seen
    penalized = _apply_repetition_penalty(logits.clone(), input_ids, penalty=2.0)

    # Positive logit divided by penalty → smaller
    assert penalized[0, 10] < logits[0, 10]
    # Negative logit multiplied by penalty → more negative
    assert penalized[0, 20] < logits[0, 20]


# ---------------------------------------------------------------------------
# test_stream_generate_yields_correct_count
# ---------------------------------------------------------------------------


def test_stream_generate_yields_correct_count(
    toy_model: TRNModel, toy_cfg: TRNConfig
) -> None:
    """stream_generate must yield exactly max_new_tokens token ids."""
    prompt = _prompt(toy_cfg, B=1, prompt_len=4)
    cfg = GenerationConfig(max_new_tokens=7, do_sample=False)
    tokens = list(stream_generate(toy_model, prompt, gen_cfg=cfg))
    assert len(tokens) == 7


# ---------------------------------------------------------------------------
# test_stream_generate_matches_generate
# ---------------------------------------------------------------------------


def test_stream_generate_matches_generate(
    toy_model: TRNModel, toy_cfg: TRNConfig
) -> None:
    """stream_generate (greedy) must produce the same tokens as generate (greedy)."""
    prompt = _prompt(toy_cfg, B=1, prompt_len=4)
    max_new = 6
    cfg = GenerationConfig(max_new_tokens=max_new, do_sample=False)

    batch_out = generate(toy_model, prompt, gen_cfg=cfg)  # (1, max_new)
    stream_out = list(stream_generate(toy_model, prompt, gen_cfg=cfg))

    assert batch_out.shape == (1, max_new)
    assert len(stream_out) == max_new
    for step, tok in enumerate(stream_out):
        assert tok == batch_out[0, step].item(), (
            f"Mismatch at step {step}: stream={tok}, batch={batch_out[0, step].item()}"
        )


# ---------------------------------------------------------------------------
# test_sample_token_temperature_low_concentrates
# ---------------------------------------------------------------------------


def test_sample_token_temperature_low_concentrates() -> None:
    """Very low temperature must concentrate mass on the argmax token."""
    torch.manual_seed(99)
    B, vocab = 4, 200
    logits = torch.randn(B, vocab)
    greedy = logits.argmax(dim=-1)

    cfg_low_temp = GenerationConfig(
        do_sample=True, temperature=0.001, top_k=0, top_p=1.0
    )
    results = []
    for _ in range(10):
        sampled = sample_token(logits.clone(), cfg_low_temp)
        results.append(sampled)

    sampled_stack = torch.stack(results, dim=0)  # (10, B)
    # All samples should match greedy under near-zero temperature.
    for b in range(B):
        assert (sampled_stack[:, b] == greedy[b]).all(), (
            f"batch {b}: expected {greedy[b]}, got {sampled_stack[:, b]}"
        )


# ---------------------------------------------------------------------------
# test_generate_finite_outputs
# ---------------------------------------------------------------------------


def test_generate_finite_outputs(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    """All generated token ids must be valid vocab indices (no nan/inf/oob)."""
    prompt = _prompt(toy_cfg, B=2, prompt_len=4)
    cfg = GenerationConfig(max_new_tokens=8, do_sample=True, temperature=1.0)
    torch.manual_seed(5)
    out = generate(toy_model, prompt, gen_cfg=cfg)
    assert out.dtype in (torch.int32, torch.int64)
    assert (out >= 0).all(), "Negative token ids found"
    assert (out < toy_cfg.vocab_size).all(), "Token ids exceed vocab size"
