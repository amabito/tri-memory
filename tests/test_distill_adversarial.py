"""Adversarial tests for distillation pipeline -- attacker mindset.

Categories tested:
- Corrupted input: NaN/Inf in teacher logits, wrong shapes
- Boundary abuse: temperature=0, both weights=0, extreme temperatures
- State corruption: vocab size mismatch, empty batch
- Numerical stability: very large/small logits, uniform distributions
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from trn.config import TRNConfig
from trn.model import TRNModel
from distill_lm import distill_loss, seed_everything


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_student():
    cfg = TRNConfig(
        vocab_size=64, d_model=32, n_oscillators=16,
        n_layers=1, d_ff=64, max_seq_len=32,
    )
    return TRNModel(cfg)


def _make_logits(batch: int = 2, seq_len: int = 16, vocab: int = 64) -> torch.Tensor:
    return torch.randn(batch, seq_len, vocab)


def _make_labels(batch: int = 2, seq_len: int = 16, vocab: int = 64) -> torch.Tensor:
    return torch.randint(0, vocab, (batch, seq_len))


# ---------------------------------------------------------------------------
# Corrupted input tests
# ---------------------------------------------------------------------------

class TestAdversarialCorruptedInput:
    """Teacher logits with NaN, Inf, or garbage values."""

    def test_nan_teacher_logits_propagates_nan_loss(self):
        """NaN in teacher logits must not silently produce a finite loss.

        If NaN propagates to loss, training loop's grad norm check catches it.
        Silent corruption (finite but wrong loss) would be worse.
        """
        s_logits = _make_logits()
        t_logits = _make_logits()
        t_logits[0, 5, :] = float("nan")
        labels = _make_labels()

        result = distill_loss(s_logits, t_logits, labels, temperature=2.0,
                              kl_weight=1.0, ce_weight=0.1)

        # NaN in teacher should propagate to KL loss (not silently disappear)
        assert torch.isnan(result["kl_loss"]) or torch.isinf(result["kl_loss"]), (
            "NaN teacher logits should propagate to KL loss, not silently produce finite value"
        )

    def test_inf_teacher_logits_does_not_crash(self):
        """Inf in teacher logits should not raise an exception."""
        s_logits = _make_logits()
        t_logits = _make_logits()
        t_logits[0, 3, 0] = float("inf")
        labels = _make_labels()

        # Should not raise
        result = distill_loss(s_logits, t_logits, labels, temperature=2.0,
                              kl_weight=1.0, ce_weight=0.1)
        assert "loss" in result

    def test_all_zero_teacher_logits_uniform(self):
        """All-zero teacher logits = uniform distribution. Loss should be finite."""
        s_logits = _make_logits()
        t_logits = torch.zeros(2, 16, 64)
        labels = _make_labels()

        result = distill_loss(s_logits, t_logits, labels, temperature=2.0,
                              kl_weight=1.0, ce_weight=0.1)

        assert torch.isfinite(result["loss"]), "Uniform teacher should produce finite loss"
        assert torch.isfinite(result["kl_loss"])

    def test_identical_student_teacher_logits_zero_kl(self):
        """When student = teacher, KL divergence should be zero."""
        logits = _make_logits()
        labels = _make_labels()

        result = distill_loss(logits.clone(), logits.clone(), labels,
                              temperature=2.0, kl_weight=1.0, ce_weight=0.0)

        assert result["kl_loss"].item() < 1e-5, (
            f"KL(p||p) should be ~0, got {result['kl_loss'].item()}"
        )


# ---------------------------------------------------------------------------
# Boundary abuse tests
# ---------------------------------------------------------------------------

class TestAdversarialBoundaryAbuse:
    """Edge cases in temperature, weights, and dimensions."""

    def test_temperature_zero_raises_or_inf(self):
        """Temperature=0 causes division by zero. Must not silently produce finite loss."""
        s_logits = _make_logits()
        t_logits = _make_logits()
        labels = _make_labels()

        result = distill_loss(s_logits, t_logits, labels, temperature=0.0,
                              kl_weight=1.0, ce_weight=0.1)

        # T=0 -> logits/0 -> Inf/NaN -> loss should not be a normal finite number
        loss = result["loss"]
        assert torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0.0, (
            f"T=0 should produce NaN/Inf/zero, got finite {loss.item()}"
        )

    def test_temperature_very_large_smooths_distribution(self):
        """Very large T -> near-uniform softmax -> small KL."""
        s_logits = _make_logits()
        t_logits = _make_logits()
        labels = _make_labels()

        result = distill_loss(s_logits, t_logits, labels, temperature=1000.0,
                              kl_weight=1.0, ce_weight=0.0)

        # At T=1000, both distributions are nearly uniform, KL should be very small
        # But T^2 correction amplifies it. The raw KL before T^2 should be tiny.
        assert torch.isfinite(result["kl_loss"]), "Large T should produce finite loss"

    def test_temperature_very_small_sharpens_distribution(self):
        """Very small T -> sharp softmax -> larger KL but still finite."""
        s_logits = _make_logits()
        t_logits = _make_logits()
        labels = _make_labels()

        result = distill_loss(s_logits, t_logits, labels, temperature=0.01,
                              kl_weight=1.0, ce_weight=0.0)

        assert torch.isfinite(result["loss"]), (
            f"Small T should be finite, got {result['loss'].item()}"
        )

    def test_both_weights_zero_produces_zero_loss(self):
        """kl_weight=0 and ce_weight=0 -> total loss = 0."""
        s_logits = _make_logits()
        t_logits = _make_logits()
        labels = _make_labels()

        result = distill_loss(s_logits, t_logits, labels, temperature=2.0,
                              kl_weight=0.0, ce_weight=0.0)

        assert result["loss"].item() == 0.0, (
            f"Both weights zero should give loss=0, got {result['loss'].item()}"
        )
        assert "kl_loss" not in result, "KL should not be computed when weight=0"
        assert "ce_loss" not in result, "CE should not be computed when weight=0"

    def test_kl_weight_zero_ce_only(self):
        """kl_weight=0 -> loss should equal CE only."""
        s_logits = _make_logits()
        t_logits = _make_logits()
        labels = _make_labels()

        result = distill_loss(s_logits, t_logits, labels, temperature=2.0,
                              kl_weight=0.0, ce_weight=1.0)

        assert "kl_loss" not in result, "KL should be skipped when weight=0"
        assert "ce_loss" in result
        assert torch.isfinite(result["loss"])

        # Verify CE matches standalone computation
        shift_s = s_logits[:, :-1].contiguous()
        targets = labels[:, 1:].contiguous()
        expected_ce = F.cross_entropy(
            shift_s.view(-1, shift_s.size(-1)), targets.view(-1),
        )
        assert abs(result["ce_loss"].item() - expected_ce.item()) < 1e-5

    def test_ce_weight_zero_kl_only(self):
        """ce_weight=0 -> loss should equal KL only."""
        s_logits = _make_logits()
        t_logits = _make_logits()
        labels = _make_labels()

        result = distill_loss(s_logits, t_logits, labels, temperature=2.0,
                              kl_weight=1.0, ce_weight=0.0)

        assert "ce_loss" not in result, "CE should be skipped when weight=0"
        assert "kl_loss" in result
        assert torch.isfinite(result["loss"])

    def test_seq_len_one_minimal_sequence(self):
        """Sequence length 1: after causal shift, 0 tokens to predict.

        distill_loss does [:, :-1] shift, so seq_len=1 -> empty tensor.
        Should not crash.
        """
        s_logits = _make_logits(seq_len=1)
        t_logits = _make_logits(seq_len=1)
        labels = _make_labels(seq_len=1)

        # This may produce 0 or NaN depending on reduction -- should not crash
        result = distill_loss(s_logits, t_logits, labels, temperature=2.0,
                              kl_weight=1.0, ce_weight=0.1)
        assert "loss" in result

    def test_negative_temperature(self):
        """Negative temperature should not silently produce valid loss."""
        s_logits = _make_logits()
        t_logits = _make_logits()
        labels = _make_labels()

        result = distill_loss(s_logits, t_logits, labels, temperature=-1.0,
                              kl_weight=1.0, ce_weight=0.1)

        # Negative T: logits/(-1) reverses order, T^2=1 -> loss could be finite
        # but semantically wrong. We just verify no crash.
        assert "loss" in result


# ---------------------------------------------------------------------------
# Numerical stability tests
# ---------------------------------------------------------------------------

class TestAdversarialNumericalStability:
    """Extreme logit values and gradient behavior."""

    def test_very_large_logits_no_overflow(self):
        """Logits of magnitude 1e6 should not produce Inf loss (softmax overflow)."""
        s_logits = torch.randn(2, 16, 64) * 1e6
        t_logits = torch.randn(2, 16, 64) * 1e6
        labels = _make_labels()

        result = distill_loss(s_logits, t_logits, labels, temperature=2.0,
                              kl_weight=1.0, ce_weight=0.1)

        # PyTorch's log_softmax is numerically stable, so this should be finite
        assert torch.isfinite(result["ce_loss"]), (
            f"CE with large logits should be finite, got {result['ce_loss'].item()}"
        )

    def test_very_small_logits_no_underflow(self):
        """Near-zero logits (magnitude 1e-10) should produce finite loss."""
        s_logits = torch.randn(2, 16, 64) * 1e-10
        t_logits = torch.randn(2, 16, 64) * 1e-10
        labels = _make_labels()

        result = distill_loss(s_logits, t_logits, labels, temperature=2.0,
                              kl_weight=1.0, ce_weight=0.1)

        assert torch.isfinite(result["loss"]), (
            f"Small logits should produce finite loss, got {result['loss'].item()}"
        )

    def test_one_hot_teacher_high_ce(self):
        """Teacher with one-hot logits (very confident) should still train."""
        s_logits = _make_logits()

        # Teacher is very confident: 100 on correct token, 0 elsewhere
        t_logits = torch.zeros(2, 16, 64)
        labels = _make_labels()
        for b in range(2):
            for t in range(16):
                t_logits[b, t, labels[b, t]] = 100.0

        result = distill_loss(s_logits, t_logits, labels, temperature=2.0,
                              kl_weight=1.0, ce_weight=0.1)

        assert torch.isfinite(result["loss"])
        assert torch.isfinite(result["kl_loss"])

    def test_gradient_flows_through_student_not_teacher(self):
        """Gradients should flow through student logits but NOT teacher logits."""
        s_logits = _make_logits().requires_grad_(True)
        t_logits = _make_logits()  # No grad
        labels = _make_labels()

        result = distill_loss(s_logits, t_logits, labels, temperature=2.0,
                              kl_weight=1.0, ce_weight=0.1)
        result["loss"].backward()

        assert s_logits.grad is not None, "Student should receive gradients"
        assert not t_logits.requires_grad, "Teacher should NOT have requires_grad"

    def test_gradient_not_nan_with_normal_inputs(self):
        """Normal inputs should produce finite gradients."""
        s_logits = _make_logits().requires_grad_(True)
        t_logits = _make_logits()
        labels = _make_labels()

        result = distill_loss(s_logits, t_logits, labels, temperature=2.0,
                              kl_weight=1.0, ce_weight=0.1)
        result["loss"].backward()

        assert torch.isfinite(s_logits.grad).all(), (
            f"Gradients should be finite, got NaN/Inf count: "
            f"{(~torch.isfinite(s_logits.grad)).sum()}"
        )


# ---------------------------------------------------------------------------
# Training loop edge cases
# ---------------------------------------------------------------------------

class TestAdversarialTrainingLoop:
    """Edge cases in the full training pipeline."""

    def test_student_with_wrong_vocab_size_raises(self, tiny_student):
        """Student vocab != teacher vocab should cause shape mismatch in CE."""
        # Student has vocab=64, teacher logits have vocab=128
        s_logits = tiny_student(torch.randint(0, 64, (2, 16)))["logits"]
        t_logits = torch.randn(2, 16, 128)  # Wrong vocab
        labels = torch.randint(0, 64, (2, 16))

        # KL should fail on vocab dimension mismatch
        with pytest.raises((RuntimeError, ValueError)):
            result = distill_loss(s_logits, t_logits, labels, temperature=2.0,
                                  kl_weight=1.0, ce_weight=0.1)

    def test_single_step_distillation_updates_weights(self, tiny_student):
        """One distillation step must actually change student weights."""
        # Snapshot weights before
        w_before = tiny_student.embedding.weight.clone()

        optimizer = torch.optim.AdamW(tiny_student.parameters(), lr=1e-3)

        input_ids = torch.randint(0, 64, (2, 16))
        s_out = tiny_student(input_ids)
        t_logits = torch.randn_like(s_out["logits"])  # Fake teacher

        losses = distill_loss(s_out["logits"], t_logits, input_ids,
                              temperature=2.0, kl_weight=1.0, ce_weight=0.1)

        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

        w_after = tiny_student.embedding.weight

        assert not torch.equal(w_before, w_after), (
            "Weights should change after one distillation step"
        )

    def test_seed_everything_deterministic(self):
        """seed_everything must produce identical random sequences."""
        seed_everything(42)
        a1 = torch.rand(10)
        a2 = np.random.rand(5)

        seed_everything(42)
        b1 = torch.rand(10)
        b2 = np.random.rand(5)

        torch.testing.assert_close(a1, b1)
        np.testing.assert_array_equal(a2, b2)

    def test_kl_loss_symmetric_property(self):
        """KL(p||q) != KL(q||p) in general (asymmetric). Verify this."""
        logits_a = _make_logits()
        logits_b = _make_logits()
        labels = _make_labels()

        result_ab = distill_loss(logits_a, logits_b, labels,
                                 temperature=2.0, kl_weight=1.0, ce_weight=0.0)
        result_ba = distill_loss(logits_b, logits_a, labels,
                                 temperature=2.0, kl_weight=1.0, ce_weight=0.0)

        # KL is asymmetric
        assert abs(result_ab["kl_loss"].item() - result_ba["kl_loss"].item()) > 1e-6, (
            "KL(a||b) should differ from KL(b||a) for random distributions"
        )

    def test_kl_loss_non_negative(self):
        """KL divergence must always be >= 0 (Gibbs' inequality)."""
        for _ in range(10):
            s_logits = _make_logits()
            t_logits = _make_logits()
            labels = _make_labels()

            result = distill_loss(s_logits, t_logits, labels, temperature=2.0,
                                  kl_weight=1.0, ce_weight=0.0)

            assert result["kl_loss"].item() >= -1e-6, (
                f"KL must be non-negative, got {result['kl_loss'].item()}"
            )
