"""P0 Stabilization regression tests.

Validates:
  1. Gradient norms stay below 1e4 (median) over 50 training steps.
  2. Training loss decreases.
  3. No NaN/Inf in loss or gradients.
  4. Alpha (gate) init mean is within [0.5, 0.85].
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from trn.config import TRNConfig
from trn.model import TRNModel


def _make_p0_config() -> TRNConfig:
    """Return a small config with P0 stabilization defaults."""
    return TRNConfig(
        vocab_size=128,
        d_model=64,
        n_oscillators=32,
        n_layers=2,
        d_ff=256,
        max_seq_len=64,
        dropout=0.0,
        use_parallel_scan=False,
        tie_weights=True,
        amplitude_max=3.0,
        state_norm=True,
        res_scale_init=0.05,
        gate_bias_init=0.65,
        phase_mode="log",
    )


class TestAlphaInit:
    """Alpha (decay gate) initialization range."""

    def test_alpha_mean_in_range(self) -> None:
        cfg = _make_p0_config()
        model = TRNModel(cfg)

        alpha_means: list[float] = []
        for name, param in model.named_parameters():
            if "proj.bias" in name:
                K = param.shape[0] // 4
                gate_bias = param.data[3 * K:]
                alpha_vals = torch.sigmoid(gate_bias)
                alpha_means.append(alpha_vals.mean().item())

        assert len(alpha_means) > 0, "No gate bias parameters found"
        for layer_idx, mean in enumerate(alpha_means):
            assert 0.5 <= mean <= 0.85, (
                f"Layer {layer_idx}: alpha mean = {mean:.3f}, expected [0.5, 0.85]"
            )

    def test_alpha_not_saturated(self) -> None:
        cfg = _make_p0_config()
        model = TRNModel(cfg)

        for name, param in model.named_parameters():
            if "proj.bias" in name:
                K = param.shape[0] // 4
                gate_bias = param.data[3 * K:]
                alpha_vals = torch.sigmoid(gate_bias)
                # No oscillator should start above 0.95
                assert alpha_vals.max().item() < 0.95, (
                    f"{name}: max alpha = {alpha_vals.max().item():.3f}, too close to 1.0"
                )


class TestResScaleInit:
    """Resonance scale initialization."""

    def test_res_scale_small(self) -> None:
        cfg = _make_p0_config()
        model = TRNModel(cfg)

        for name, param in model.named_parameters():
            if "res_scale" in name:
                assert param.item() == pytest.approx(0.05, abs=1e-6), (
                    f"{name}: res_scale = {param.item()}, expected 0.05"
                )


class TestGradientStability:
    """Run 50 training steps and verify gradient norms stay bounded."""

    @pytest.fixture()
    def train_result(self) -> dict:
        torch.manual_seed(42)
        cfg = _make_p0_config()
        model = TRNModel(cfg)

        param_groups = model.configure_optimizer_param_groups(weight_decay=0.1)
        optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95))

        n_steps = 50
        losses: list[float] = []
        grad_norms: list[float] = []
        has_nan = False
        has_inf = False

        model.train()
        for step in range(n_steps):
            input_ids = torch.randint(0, cfg.vocab_size, (4, cfg.max_seq_len))
            optimizer.zero_grad()
            out = model(input_ids, labels=input_ids)
            loss = out["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                has_nan = True
                break

            loss.backward()

            # Compute unclipped grad norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        has_inf = True
                    total_norm += p.grad.data.float().norm().item() ** 2
            grad_norms.append(total_norm ** 0.5)

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        return {
            "losses": losses,
            "grad_norms": grad_norms,
            "has_nan": has_nan,
            "has_inf": has_inf,
        }

    def test_no_nan(self, train_result: dict) -> None:
        assert not train_result["has_nan"], "NaN detected in loss"

    def test_no_inf_grads(self, train_result: dict) -> None:
        assert not train_result["has_inf"], "Inf detected in gradients"

    def test_grad_norm_median_bounded(self, train_result: dict) -> None:
        gn = sorted(train_result["grad_norms"])
        median = gn[len(gn) // 2]
        assert median < 1e4, f"Median grad norm = {median:.1f}, expected < 1e4"

    def test_loss_decreases(self, train_result: dict) -> None:
        losses = train_result["losses"]
        assert len(losses) == 50
        first_5 = sum(losses[:5]) / 5
        last_5 = sum(losses[-5:]) / 5
        assert last_5 < first_5, (
            f"Loss did not decrease: first_5={first_5:.4f}, last_5={last_5:.4f}"
        )

    def test_loss_is_finite(self, train_result: dict) -> None:
        for i, loss in enumerate(train_result["losses"]):
            assert math.isfinite(loss), f"Loss at step {i} is not finite: {loss}"


class TestStateNorm:
    """State normalization bounds resonance state magnitude."""

    def test_state_bounded_after_forward(self) -> None:
        torch.manual_seed(42)
        cfg = _make_p0_config()
        model = TRNModel(cfg)

        input_ids = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
        _ = model(input_ids)

        # Verify resonance states in each layer are bounded
        # (state_norm ensures max_abs <= 1.0 per channel)
        # We can't directly inspect states from forward, but we verify
        # the output doesn't explode
        out = model(input_ids)
        logits = out["logits"]
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        # With state norm, logit magnitudes should be reasonable
        assert logits.abs().max().item() < 1000.0, (
            f"Logit magnitude too large: {logits.abs().max().item():.1f}"
        )


class TestAmplitudeClamp:
    """Amplitude is bounded by amplitude_max."""

    def test_amplitude_bounded(self) -> None:
        torch.manual_seed(42)
        cfg = _make_p0_config()
        cfg.amplitude_max = 3.0
        model = TRNModel(cfg)

        # Feed large inputs to push amplitude logits high
        x = torch.randn(2, 16, cfg.d_model) * 10.0
        for block in model.blocks:
            A, omega, phi, alpha = block.resonance.proj(x)
            assert A.max().item() <= 3.0 + 1e-6, (
                f"Amplitude exceeded max: {A.max().item():.4f}"
            )


class TestPhaseMode:
    """Phase mode config switches between log and linear."""

    def test_log_phase_default(self) -> None:
        cfg = _make_p0_config()
        assert cfg.phase_mode == "log"
        model = TRNModel(cfg)
        for block in model.blocks:
            assert block.resonance.phase_mode == "log"

    def test_linear_phase_override(self) -> None:
        cfg = _make_p0_config()
        cfg.phase_mode = "linear"
        model = TRNModel(cfg)
        for block in model.blocks:
            assert block.resonance.phase_mode == "linear"
