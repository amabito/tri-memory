"""Tests for compare.py — happy path + adversarial."""
from __future__ import annotations

import math

import pytest

from trimemory.compare import ComparisonResult, print_comparison_report, run_comparison
from trimemory.config import TRNConfig


@pytest.fixture
def toy_cfg() -> TRNConfig:
    """Tiny config for fast tests."""
    return TRNConfig(
        vocab_size=32,
        d_model=32,
        n_layers=1,
        n_oscillators=16,
        d_ff=64,
        max_seq_len=32,
        dropout=0.0,
        tie_weights=False,
        use_parallel_scan=False,
    )


# ---------------------------------------------------------------------------
# Happy path tests (5)
# ---------------------------------------------------------------------------


def test_comparison_result_fields(toy_cfg: TRNConfig) -> None:
    result = run_comparison(
        toy_cfg, n_train_steps=5, n_bench_steps=5, batch_size=2, seq_len=8
    )
    assert isinstance(result, ComparisonResult)
    assert hasattr(result, "trn_loss")
    assert hasattr(result, "transformer_loss")
    assert hasattr(result, "trn_params")
    assert hasattr(result, "transformer_params")
    assert hasattr(result, "trn_forward_tps")
    assert hasattr(result, "transformer_forward_tps")
    assert hasattr(result, "trn_gen_tps")
    assert hasattr(result, "transformer_gen_tps")
    assert hasattr(result, "n_steps")
    assert result.n_steps == 5


def test_trn_and_transformer_both_train(toy_cfg: TRNConfig) -> None:
    result = run_comparison(
        toy_cfg, n_train_steps=5, n_bench_steps=5, batch_size=2, seq_len=8
    )
    assert math.isfinite(result.trn_loss)
    assert math.isfinite(result.transformer_loss)
    assert result.trn_loss > 0.0
    assert result.transformer_loss > 0.0


def test_parameter_counts_positive(toy_cfg: TRNConfig) -> None:
    result = run_comparison(
        toy_cfg, n_train_steps=5, n_bench_steps=5, batch_size=2, seq_len=8
    )
    assert result.trn_params > 0
    assert result.transformer_params > 0


def test_gen_tps_positive(toy_cfg: TRNConfig) -> None:
    result = run_comparison(
        toy_cfg, n_train_steps=5, n_bench_steps=5, batch_size=2, seq_len=8
    )
    assert result.trn_gen_tps > 0.0
    assert result.transformer_gen_tps > 0.0


def test_print_comparison_no_crash(toy_cfg: TRNConfig) -> None:
    result = run_comparison(
        toy_cfg, n_train_steps=5, n_bench_steps=5, batch_size=2, seq_len=8
    )
    # Should not raise
    print_comparison_report(result)


# ---------------------------------------------------------------------------
# Adversarial tests (2)
# ---------------------------------------------------------------------------


def test_comparison_zero_steps(toy_cfg: TRNConfig) -> None:
    """n_train_steps=0 must not crash — loss may be inf, no exception."""
    result = run_comparison(
        toy_cfg, n_train_steps=0, n_bench_steps=3, batch_size=2, seq_len=8
    )
    # Should not raise; loss is either finite (computed) or inf
    assert isinstance(result.trn_loss, float)
    assert isinstance(result.transformer_loss, float)


def test_trn_gen_tps_advantage(toy_cfg: TRNConfig) -> None:
    """TRN step_single and Transformer gen must both be > 0 (O(1) vs O(n^2))."""
    result = run_comparison(
        toy_cfg, n_train_steps=2, n_bench_steps=5, batch_size=1, seq_len=8
    )
    assert result.trn_gen_tps > 0.0
    assert result.transformer_gen_tps > 0.0
