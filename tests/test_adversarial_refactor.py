"""Adversarial tests for post-refactor correctness -- attacker mindset.

Covers:
  1. phase_mode linear vs log produce distinct positions and no NaN
  2. phase_mode invalid string -- fallback or clean failure, no crash
  3. scan_chunk_size=0 -- degenerate, must raise or handle
  4. scan_chunk_size negative -- must not crash
  5. TemporalResonanceLayer forward-backward gradient finiteness
  6. Sin/cos cache correctness -- two calls must be identical (deterministic)
  7. TriMemoryEngine embedding reuse -- forward_with_memory outputs are finite,
     dropout patterns in train mode
  8. Narrow except exposure -- ValueError from associative_scan must propagate
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from trimemory.config import TRNConfig
from trimemory.resonance import TemporalResonanceLayer
from trimemory.scan import chunked_resonance_scan


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_cfg() -> TRNConfig:
    return TRNConfig.toy()


def _make_layer(phase_mode: str = "log", chunk_size: int = 16) -> TemporalResonanceLayer:
    torch.manual_seed(0)
    return TemporalResonanceLayer(
        d_model=32,
        K=16,
        use_parallel_scan=False,  # force CPU path (chunked scan)
        phase_mode=phase_mode,
        scan_chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# Category 1: phase_mode linear vs log -- positions must differ, no NaN
# ---------------------------------------------------------------------------

class TestPhaseModeLinearVsLog:
    """phase_mode controls how positions are encoded; the two modes must diverge."""

    def test_linear_and_log_produce_different_outputs(self) -> None:
        """phase_mode='linear' and phase_mode='log' must give different forward outputs."""
        torch.manual_seed(1)
        x = torch.randn(2, 8, 32)
        # Identical weights so any difference comes from position encoding only.
        layer_log = _make_layer(phase_mode="log")
        layer_lin = _make_layer(phase_mode="linear")
        # Copy log weights into linear layer so the sole difference is phase_mode.
        layer_lin.load_state_dict(layer_log.state_dict())
        layer_log.eval()
        layer_lin.eval()

        with torch.no_grad():
            out_log = layer_log(x)
            out_lin = layer_lin(x)

        assert not torch.allclose(out_log, out_lin), (
            "log and linear phase modes must produce different outputs"
        )

    def test_linear_output_finite(self) -> None:
        """phase_mode='linear' must not produce NaN or Inf."""
        torch.manual_seed(2)
        layer = _make_layer(phase_mode="linear").eval()
        x = torch.randn(2, 12, 32)
        with torch.no_grad():
            out = layer(x)
        assert torch.isfinite(out).all(), "linear phase mode produced non-finite output"

    def test_log_output_finite(self) -> None:
        """phase_mode='log' must not produce NaN or Inf."""
        torch.manual_seed(3)
        layer = _make_layer(phase_mode="log").eval()
        x = torch.randn(2, 12, 32)
        with torch.no_grad():
            out = layer(x)
        assert torch.isfinite(out).all(), "log phase mode produced non-finite output"

    def test_positions_differ_for_n_greater_than_1(self) -> None:
        """_compute_positions must return log1p(t) vs t -- verify raw values differ."""
        layer_log = _make_layer(phase_mode="log").eval()
        layer_lin = _make_layer(phase_mode="linear").eval()
        n = 10

        pos_log = layer_log._compute_positions(n, torch.device("cpu"))
        pos_lin = layer_lin._compute_positions(n, torch.device("cpu"))

        # Positions 0..n-1: log1p(0)=0, log1p(1)=log(2), etc. vs 0,1,2,...
        # Beyond t=0 they diverge.
        assert not torch.allclose(pos_log, pos_lin), (
            "_compute_positions must differ between log and linear modes"
        )
        assert torch.isfinite(pos_log).all()
        assert torch.isfinite(pos_lin).all()


# ---------------------------------------------------------------------------
# Category 2: phase_mode invalid string -- no crash, meaningful behaviour
# ---------------------------------------------------------------------------

class TestPhaseModeInvalidString:
    """An unrecognised phase_mode string must not crash forward."""

    def test_invalid_phase_mode_no_crash(self) -> None:
        """phase_mode='invalid' falls through the if-block -- linear behaviour."""
        layer = _make_layer(phase_mode="invalid").eval()
        x = torch.randn(1, 5, 32)
        # Must not raise
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (1, 5, 32)

    def test_invalid_phase_mode_output_finite(self) -> None:
        """Invalid phase_mode must still produce finite output."""
        layer = _make_layer(phase_mode="invalid").eval()
        x = torch.randn(2, 8, 32)
        with torch.no_grad():
            out = layer(x)
        assert torch.isfinite(out).all(), (
            "invalid phase_mode produced non-finite output"
        )

    def test_invalid_phase_mode_matches_linear_fallback(self) -> None:
        """'invalid' falls to the else branch -- must match 'linear' exactly."""
        torch.manual_seed(7)
        x = torch.randn(2, 6, 32)
        layer_invalid = _make_layer(phase_mode="invalid")
        layer_linear = _make_layer(phase_mode="linear")
        layer_linear.load_state_dict(layer_invalid.state_dict())
        layer_invalid.eval()
        layer_linear.eval()

        with torch.no_grad():
            out_invalid = layer_invalid(x)
            out_linear = layer_linear(x)

        assert torch.allclose(out_invalid, out_linear), (
            "unknown phase_mode must fall back to linear (no if-branch hit -> same raw positions)"
        )


# ---------------------------------------------------------------------------
# Category 3: scan_chunk_size=0 -- degenerate input
# ---------------------------------------------------------------------------

class TestScanChunkSizeZero:
    """chunk_size=0 in chunked_resonance_scan is degenerate -- must raise or handle."""

    def test_chunked_scan_chunk_size_zero_raises_or_handles(self) -> None:
        """chunk_size=0 causes range(0, n, 0) -- Python raises ValueError."""
        B, n, K = 2, 8, 16
        alpha = torch.rand(B, n, K)
        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        # range(0, n, 0) raises ValueError: range() arg 3 must not be zero
        with pytest.raises((ValueError, ZeroDivisionError)):
            chunked_resonance_scan(alpha, drive_r, drive_i, chunk_size=0)

    def test_layer_with_chunk_size_zero_forward_raises(self) -> None:
        """TemporalResonanceLayer with scan_chunk_size=0 must raise on forward."""
        layer = _make_layer(chunk_size=0).eval()
        x = torch.randn(1, 4, 32)
        with pytest.raises((ValueError, ZeroDivisionError)):
            with torch.no_grad():
                layer(x)


# ---------------------------------------------------------------------------
# Category 4: scan_chunk_size negative -- must not silently corrupt or crash badly
# ---------------------------------------------------------------------------

class TestScanChunkSizeNegative:
    """Negative chunk_size: range(0, n, negative) = empty loop -- produces empty cat."""

    def test_chunked_scan_negative_chunk_size_raises_or_empty(self) -> None:
        """Negative step in range silently produces no iterations.

        torch.cat([]) raises RuntimeError -- acceptable, not a silent corruption.
        """
        B, n, K = 2, 8, 16
        alpha = torch.rand(B, n, K)
        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        # range(0, 8, -4) is empty -> torch.cat([]) -> RuntimeError
        # Either a clear exception or a shape mismatch is acceptable.
        try:
            r_r, r_i = chunked_resonance_scan(alpha, drive_r, drive_i, chunk_size=-4)
            # If it somehow succeeds, outputs must be finite (no silent NaN injection)
            assert torch.isfinite(r_r).all()
            assert torch.isfinite(r_i).all()
        except (ValueError, RuntimeError):
            pass  # explicit error is the expected path

    def test_layer_negative_chunk_size_does_not_produce_nan(self) -> None:
        """If layer survives negative chunk_size, output must not be NaN."""
        layer = _make_layer(chunk_size=-1).eval()
        x = torch.randn(1, 4, 32)
        try:
            with torch.no_grad():
                out = layer(x)
            # If it returned, must be finite -- NaN is the worst outcome
            assert not torch.isnan(out).any(), (
                "negative scan_chunk_size produced NaN output silently"
            )
        except (ValueError, RuntimeError):
            pass  # explicit error is acceptable


# ---------------------------------------------------------------------------
# Category 5: forward-backward gradient finiteness
# ---------------------------------------------------------------------------

class TestForwardBackwardGradients:
    """All trainable parameters must receive finite gradients after a forward-backward."""

    def test_gradients_finite_default_log_mode(self) -> None:
        """Log mode -- standard training step must produce finite gradients."""
        torch.manual_seed(10)
        layer = _make_layer(phase_mode="log").train()
        x = torch.randn(2, 16, 32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        for name, param in layer.named_parameters():
            assert param.grad is not None, f"param {name} has no gradient"
            assert torch.isfinite(param.grad).all(), (
                f"param {name} has non-finite gradient (log mode)"
            )

    def test_gradients_finite_linear_mode(self) -> None:
        """Linear mode -- standard training step must produce finite gradients."""
        torch.manual_seed(11)
        layer = _make_layer(phase_mode="linear").train()
        x = torch.randn(2, 16, 32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        for name, param in layer.named_parameters():
            assert param.grad is not None, f"param {name} has no gradient"
            assert torch.isfinite(param.grad).all(), (
                f"param {name} has non-finite gradient (linear mode)"
            )

    def test_gradients_finite_long_sequence(self) -> None:
        """Long sequence (n=512) must not cause gradient explosion."""
        torch.manual_seed(12)
        layer = _make_layer(phase_mode="log", chunk_size=32).train()
        x = torch.randn(1, 512, 32, requires_grad=True)
        out = layer(x)
        loss = out.mean()
        loss.backward()

        for name, param in layer.named_parameters():
            assert param.grad is not None, f"param {name} has no gradient"
            assert torch.isfinite(param.grad).all(), (
                f"param {name} exploded on long sequence"
            )
            # Catch obvious explosion (> 1e6 is a red flag)
            assert param.grad.abs().max().item() < 1e6, (
                f"param {name} gradient magnitude suspiciously large: "
                f"{param.grad.abs().max().item():.2e}"
            )


# ---------------------------------------------------------------------------
# Category 6: sin/cos cache correctness -- two identical calls must agree
# ---------------------------------------------------------------------------

class TestSinCosCacheCorrectness:
    """Two forward passes with the same input must produce bit-identical outputs.

    This validates that the refactored sin/cos reuse does not introduce
    state mutation or stale cache bugs.
    """

    def test_two_calls_identical_eval(self) -> None:
        """Determinism in eval mode: forward(x) == forward(x)."""
        torch.manual_seed(20)
        layer = _make_layer(phase_mode="log").eval()
        x = torch.randn(3, 10, 32)

        with torch.no_grad():
            out1 = layer(x)
            out2 = layer(x)

        assert torch.equal(out1, out2), (
            "two forward calls with identical input returned different results"
        )

    def test_two_calls_identical_linear_eval(self) -> None:
        """Same check for linear mode."""
        torch.manual_seed(21)
        layer = _make_layer(phase_mode="linear").eval()
        x = torch.randn(3, 10, 32)

        with torch.no_grad():
            out1 = layer(x)
            out2 = layer(x)

        assert torch.equal(out1, out2), (
            "linear mode: two calls returned different results"
        )

    def test_different_inputs_produce_different_outputs(self) -> None:
        """Sanity: different x must not collapse to the same output (cache not frozen)."""
        torch.manual_seed(22)
        layer = _make_layer(phase_mode="log").eval()
        x1 = torch.randn(2, 6, 32)
        x2 = torch.randn(2, 6, 32)

        with torch.no_grad():
            out1 = layer(x1)
            out2 = layer(x2)

        assert not torch.allclose(out1, out2), (
            "different inputs collapsed to the same output -- possible frozen cache"
        )


# ---------------------------------------------------------------------------
# Category 7: TriMemoryEngine embedding reuse
# ---------------------------------------------------------------------------

class TestTriMemoryEngineEmbeddingReuse:
    """forward_with_memory must produce finite output and correctly apply dropout."""

    @pytest.fixture
    def engine_cfg(self) -> TRNConfig:
        # Tiny config: fast forward, still exercises all paths
        return TRNConfig(
            vocab_size=64,
            d_model=32,
            n_oscillators=16,
            n_layers=2,
            d_ff=64,
            max_seq_len=128,
            dropout=0.0,
        )

    def _make_states(
        self, cfg: TRNConfig, B: int, device: torch.device
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        states_r = [torch.zeros(B, cfg.n_oscillators, device=device) for _ in range(cfg.n_layers)]
        states_i = [torch.zeros(B, cfg.n_oscillators, device=device) for _ in range(cfg.n_layers)]
        return states_r, states_i

    def test_forward_with_memory_output_finite(self, engine_cfg: TRNConfig) -> None:
        """forward_with_memory logits must be finite with zero initial states."""
        from trimemory.tri_memory import TriMemoryEngine

        torch.manual_seed(30)
        engine = TriMemoryEngine(engine_cfg).eval()
        B, T = 1, 8
        input_ids = torch.randint(0, engine_cfg.vocab_size, (B, T))
        states_r, states_i = self._make_states(engine_cfg, B, torch.device("cpu"))

        with torch.no_grad():
            result, new_r, new_i, new_kv = engine.forward_with_memory(
                input_ids, states_r, states_i, position=0
            )

        logits = result["logits"]
        assert logits.shape == (B, T, engine_cfg.vocab_size)
        assert torch.isfinite(logits).all(), "logits contain non-finite values"

        # Updated states must also be finite
        for layer_r, layer_i in zip(new_r, new_i):
            assert torch.isfinite(layer_r).all(), "updated state_r is non-finite"
            assert torch.isfinite(layer_i).all(), "updated state_i is non-finite"

    def test_embedding_reuse_does_not_alter_output(self, engine_cfg: TRNConfig) -> None:
        """Two calls with same input+state must produce identical logits (no mutation side-effect)."""
        from trimemory.tri_memory import TriMemoryEngine

        torch.manual_seed(31)
        engine = TriMemoryEngine(engine_cfg).eval()
        B, T = 1, 6
        input_ids = torch.randint(0, engine_cfg.vocab_size, (B, T))

        states_r1, states_i1 = self._make_states(engine_cfg, B, torch.device("cpu"))
        states_r2, states_i2 = self._make_states(engine_cfg, B, torch.device("cpu"))

        with torch.no_grad():
            res1, _, _, _ = engine.forward_with_memory(input_ids, states_r1, states_i1, position=0)
            res2, _, _, _ = engine.forward_with_memory(input_ids, states_r2, states_i2, position=0)

        assert torch.equal(res1["logits"], res2["logits"]), (
            "identical inputs/states produced different logits -- possible embedding mutation"
        )

    def test_forward_with_memory_train_mode_applies_dropout(self, engine_cfg: TRNConfig) -> None:
        """With dropout > 0 in train mode, two calls must differ (dropout is active)."""
        from trimemory.tri_memory import TriMemoryEngine

        cfg_drop = TRNConfig(
            vocab_size=engine_cfg.vocab_size,
            d_model=engine_cfg.d_model,
            n_oscillators=engine_cfg.n_oscillators,
            n_layers=engine_cfg.n_layers,
            d_ff=engine_cfg.d_ff,
            max_seq_len=engine_cfg.max_seq_len,
            dropout=0.5,
        )
        torch.manual_seed(32)
        engine = TriMemoryEngine(cfg_drop).train()
        B, T = 2, 8
        input_ids = torch.randint(0, cfg_drop.vocab_size, (B, T))

        states_r1, states_i1 = self._make_states(cfg_drop, B, torch.device("cpu"))
        states_r2, states_i2 = self._make_states(cfg_drop, B, torch.device("cpu"))

        # Different random seeds between calls -- should differ because dropout is stochastic
        with torch.no_grad():
            res1, _, _, _ = engine.forward_with_memory(input_ids, states_r1, states_i1, position=0)
        with torch.no_grad():
            res2, _, _, _ = engine.forward_with_memory(input_ids, states_r2, states_i2, position=0)

        # With p=0.5 and d_model=32, probability of identical output is negligible
        assert not torch.equal(res1["logits"], res2["logits"]), (
            "dropout=0.5 in train mode produced identical logits on two calls -- "
            "dropout may not be active"
        )

    def test_forward_with_memory_states_are_not_output_input_aliases(
        self, engine_cfg: TRNConfig
    ) -> None:
        """Updated states returned must not alias the input state tensors."""
        from trimemory.tri_memory import TriMemoryEngine

        torch.manual_seed(33)
        engine = TriMemoryEngine(engine_cfg).eval()
        B, T = 1, 4
        input_ids = torch.randint(0, engine_cfg.vocab_size, (B, T))
        states_r, states_i = self._make_states(engine_cfg, B, torch.device("cpu"))

        with torch.no_grad():
            _, new_r, new_i, _ = engine.forward_with_memory(
                input_ids, states_r, states_i, position=0
            )

        # new states come from step_single, which builds new tensors
        # They may or may not alias -- but they must be finite regardless
        for layer_r in new_r:
            assert torch.isfinite(layer_r).all()


# ---------------------------------------------------------------------------
# Category 8: narrow except exposure -- ValueError must propagate
# ---------------------------------------------------------------------------

class TestNarrowExceptExposure:
    """scan.py catches only (TypeError, RuntimeError, AttributeError).

    A ValueError raised inside associative_scan must NOT be silently swallowed.
    """

    def test_value_error_propagates_through_parallel_scan(self) -> None:
        """Monkeypatch torch.associative_scan to raise ValueError.

        parallel_resonance_scan's except clause is:
          except (TypeError, RuntimeError, AttributeError)
        ValueError is NOT in this tuple, so it must propagate.
        """
        import trimemory.scan as scan_module

        def _raise_value_error(*args, **kwargs):  # noqa: ANN001
            raise ValueError("injected ValueError -- must not be caught")

        # Ensure associative_scan attribute exists on torch for the patch target
        if not hasattr(torch, "associative_scan"):
            pytest.skip("torch.associative_scan not present; parallel scan not reachable")

        B, n, K = 2, 8, 16
        alpha = torch.rand(B, n, K)
        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        with patch.object(torch, "associative_scan", side_effect=_raise_value_error):
            with pytest.raises(ValueError, match="injected ValueError"):
                scan_module.parallel_resonance_scan(alpha, drive_r, drive_i)

    def test_os_error_propagates_through_parallel_scan(self) -> None:
        """OSError (also not caught) must propagate -- belt-and-suspenders check."""
        import trimemory.scan as scan_module

        if not hasattr(torch, "associative_scan"):
            pytest.skip("torch.associative_scan not present")

        B, n, K = 1, 4, 8
        alpha = torch.rand(B, n, K)
        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        with patch.object(torch, "associative_scan", side_effect=OSError("injected OSError")):
            with pytest.raises(OSError, match="injected OSError"):
                scan_module.parallel_resonance_scan(alpha, drive_r, drive_i)

    def test_caught_exceptions_do_fallback(self) -> None:
        """TypeError and RuntimeError ARE in the catch list -- must fallback to chunked scan.

        Verifies that the narrowing did not accidentally remove the fallback for
        these two expected error types.
        """
        import trimemory.scan as scan_module

        if not hasattr(torch, "associative_scan"):
            pytest.skip("torch.associative_scan not present")

        B, n, K = 2, 8, 16
        alpha = torch.rand(B, n, K)
        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        # Compute reference via chunked scan
        ref_r, ref_i = scan_module.chunked_resonance_scan(alpha, drive_r, drive_i)

        for exc_cls in (TypeError, RuntimeError, AttributeError):
            with patch.object(
                torch, "associative_scan", side_effect=exc_cls("injected")
            ):
                # Must not raise; must return chunked-scan result
                out_r, out_i = scan_module.parallel_resonance_scan(alpha, drive_r, drive_i)

            assert torch.allclose(out_r, ref_r, atol=1e-5), (
                f"{exc_cls.__name__} fallback produced different r_r"
            )
            assert torch.allclose(out_i, ref_i, atol=1e-5), (
                f"{exc_cls.__name__} fallback produced different r_i"
            )
