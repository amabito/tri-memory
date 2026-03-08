"""Cross-module adversarial tests for TRN implementation."""
from __future__ import annotations

import threading
import pytest
import torch
import torch.nn as nn
from trn.config import TRNConfig
from trn.scan import _combine, sequential_resonance_scan
from trn.oscillator import OscillatorProjection
from trn.resonance import TemporalResonanceLayer
from trn.block import TRNBlock, SwiGLU
from trn.model import TRNModel
from trn.utils import build_rms_norm


@pytest.fixture
def toy_cfg() -> TRNConfig:
    return TRNConfig.toy()


@pytest.fixture
def toy_model(toy_cfg: TRNConfig) -> TRNModel:
    torch.manual_seed(0)
    model = TRNModel(toy_cfg)
    model.eval()
    return model


@pytest.fixture
def toy_layer(toy_cfg: TRNConfig) -> TemporalResonanceLayer:
    torch.manual_seed(1)
    return TemporalResonanceLayer(
        d_model=toy_cfg.d_model,
        K=toy_cfg.n_oscillators,
        use_parallel_scan=False,
    )


# ---------------------------------------------------------------------------
# 1. CORRUPTED INPUT
# ---------------------------------------------------------------------------

def test_nan_in_embedding_propagates(toy_cfg: TRNConfig) -> None:
    """NaN in embedding weight propagates to logits (not silently swallowed)."""
    model = TRNModel(toy_cfg)
    model.eval()
    model.embedding.weight.data[0] = float("nan")
    input_ids = torch.zeros(1, 4, dtype=torch.long)  # all token 0
    out = model(input_ids)
    assert torch.isnan(out["logits"]).any(), (
        "Expected NaN to propagate through embedding, but logits are clean"
    )


def test_vocab_boundary_tokens(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    """First and last vocab tokens must not crash, outputs must be finite."""
    ids = torch.tensor([[0, toy_cfg.vocab_size - 1]])
    out = toy_model(ids)
    assert torch.isfinite(out["logits"]).all()


def test_empty_sequence_zero_n(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    """n=0 input — either works or raises cleanly, no segfault."""
    ids = torch.zeros(1, 0, dtype=torch.long)
    try:
        out = toy_model(ids)
        # If it doesn't crash, logits should have correct shape
        assert out["logits"].shape[0] == 1
        assert out["logits"].shape[2] == toy_cfg.vocab_size
    except (RuntimeError, ValueError, IndexError):
        pass  # Clean failure is acceptable


# ---------------------------------------------------------------------------
# 2. SHAPE EDGE CASES
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_name", ["resonance", "block", "model"])
def test_n1_all_modules(toy_cfg: TRNConfig, module_name: str) -> None:
    """n=1 must work for all core modules."""
    B, d = 2, toy_cfg.d_model
    if module_name == "resonance":
        layer = TemporalResonanceLayer(d, toy_cfg.n_oscillators, use_parallel_scan=False)
        x = torch.randn(B, 1, d)
        out = layer(x)
        assert out.shape == (B, 1, d)
    elif module_name == "block":
        block = TRNBlock(toy_cfg)
        x = torch.randn(B, 1, d)
        out = block(x)
        assert out.shape == (B, 1, d)
    else:
        model = TRNModel(toy_cfg)
        ids = torch.randint(0, toy_cfg.vocab_size, (B, 1))
        out = model(ids)
        assert out["logits"].shape == (B, 1, toy_cfg.vocab_size)


def test_odd_n(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    """Odd sequence length n=7 must produce correct shapes."""
    B, n = 2, 7
    ids = torch.randint(0, toy_cfg.vocab_size, (B, n))
    out = toy_model(ids)
    assert out["logits"].shape == (B, n, toy_cfg.vocab_size)


def test_b1(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    """B=1 must not crash."""
    ids = torch.randint(0, toy_cfg.vocab_size, (1, 16))
    out = toy_model(ids)
    assert out["logits"].shape == (1, 16, toy_cfg.vocab_size)


def test_large_batch_cpu(toy_cfg: TRNConfig) -> None:
    """B=32, n=16 on CPU — no crash, finite output."""
    model = TRNModel(toy_cfg)
    model.eval()
    ids = torch.randint(0, toy_cfg.vocab_size, (32, 16))
    out = model(ids)
    assert torch.isfinite(out["logits"]).all()


# ---------------------------------------------------------------------------
# 3. MIXED PRECISION
# ---------------------------------------------------------------------------

def test_bf16_block_output_dtype(toy_cfg: TRNConfig) -> None:
    """Block with bf16 input should return bf16 output."""
    block = TRNBlock(toy_cfg).bfloat16()
    x = torch.randn(2, 8, toy_cfg.d_model, dtype=torch.bfloat16)
    out = block(x)
    assert out.dtype == torch.bfloat16


def test_fp32_block_output_dtype(toy_cfg: TRNConfig) -> None:
    """Block with fp32 input should return fp32 output."""
    block = TRNBlock(toy_cfg)
    x = torch.randn(2, 8, toy_cfg.d_model, dtype=torch.float32)
    out = block(x)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# 4. CONCURRENT FORWARD (read-only model state)
# ---------------------------------------------------------------------------

def test_concurrent_model_forward_readonly(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    """8 threads call model.forward() simultaneously — all finite, no crash."""
    N_THREADS = 8
    B_per_thread = 2
    n = 16
    results: list[torch.Tensor | Exception] = [None] * N_THREADS  # type: ignore[list-item]

    def worker(idx: int) -> None:
        ids = torch.randint(0, toy_cfg.vocab_size, (B_per_thread, n))
        try:
            with torch.inference_mode():
                out = toy_model(ids)
            results[idx] = out["logits"]
        except Exception as e:
            results[idx] = e

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i, res in enumerate(results):
        assert not isinstance(res, Exception), f"Thread {i} raised: {res}"
        assert torch.isfinite(res).all(), f"Thread {i} produced non-finite logits"


# ---------------------------------------------------------------------------
# 5. MATHEMATICAL INVARIANTS
# ---------------------------------------------------------------------------

def test_rms_norm_output(toy_cfg: TRNConfig) -> None:
    """With weight=ones, RMSNorm output rows should have RMS ≈ 1.0."""
    norm = build_rms_norm(toy_cfg.d_model)
    # Ensure weight is ones
    with torch.no_grad():
        if hasattr(norm, "weight"):
            norm.weight.fill_(1.0)
    x = torch.randn(4, 16, toy_cfg.d_model)
    y = norm(x)
    rms = y.float().pow(2).mean(-1).sqrt()
    # Should be close to 1.0 (tolerance 0.1 to account for eps)
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1), (
        f"RMS of normalized output deviates from 1.0: mean={rms.mean():.4f}"
    )


def test_swiglu_formula_manual(toy_cfg: TRNConfig) -> None:
    """Verify SwiGLU(x) == down(silu(gate(x)) * up(x)) with fixed weights."""
    torch.manual_seed(7)
    d, h = 8, 16
    ffn = SwiGLU(d, h)
    x = torch.randn(2, 4, d)
    out = ffn(x)
    expected = ffn.down(torch.nn.functional.silu(ffn.gate(x)) * ffn.up(x))
    assert torch.allclose(out, expected, atol=1e-6)


def test_combine_left_identity() -> None:
    """_combine((ones, zeros), (a, b)) == (a, b)."""
    a = torch.tensor([2.0, 3.0])
    b = torch.tensor([4.0, 5.0])
    identity = (torch.ones_like(a), torch.zeros_like(b))
    ra, rb = _combine(identity, (a, b))
    assert torch.allclose(ra, a)
    assert torch.allclose(rb, b)


def test_combine_composition() -> None:
    """Manual 3-step scan matches sequential scan output."""
    B, n, K = 2, 6, 4
    torch.manual_seed(3)
    alpha   = torch.rand(B, n, K) * 0.9     # (0, 0.9)
    drive_r = torch.randn(B, n, K)
    drive_i = torch.randn(B, n, K)
    out_r, out_i = sequential_resonance_scan(alpha, drive_r, drive_i)
    # Manually compute first 3 steps for batch 0
    r = torch.zeros(K)
    for t in range(3):
        r = alpha[0, t] * r + drive_r[0, t]
    assert torch.allclose(r, out_r[0, 2], atol=1e-5)


# ---------------------------------------------------------------------------
# 6. WEIGHT TYING GRADIENT
# ---------------------------------------------------------------------------

def test_weight_tying_gradient_flows_both_ways(toy_cfg: TRNConfig) -> None:
    """With tie_weights=True, embedding.weight.grad must be non-None after backward."""
    cfg   = TRNConfig.toy()
    model = TRNModel(cfg)
    B, n  = 2, 16
    ids    = torch.randint(0, cfg.vocab_size, (B, n))
    labels = torch.randint(0, cfg.vocab_size, (B, n))
    out   = model(ids, labels=labels)
    out["loss"].backward()
    assert model.embedding.weight.grad is not None


# ---------------------------------------------------------------------------
# 7. STEP_SINGLE ADVERSARIAL
# ---------------------------------------------------------------------------

def test_step_single_large_position_no_overflow(
    toy_layer: TemporalResonanceLayer, toy_cfg: TRNConfig
) -> None:
    """Large position index (1_000_000) should produce finite output."""
    B, d = 2, toy_cfg.d_model
    x  = torch.randn(B, d)
    r  = torch.zeros(B, toy_cfg.n_oscillators)
    i_ = torch.zeros(B, toy_cfg.n_oscillators)
    out, r_new, i_new = toy_layer.step_single(x, r, i_, position=1_000_000)
    assert torch.isfinite(out).all(),   "Output not finite at large position"
    assert torch.isfinite(r_new).all(), "r_real not finite at large position"
    assert torch.isfinite(i_new).all(), "r_imag not finite at large position"


def test_step_single_state_not_mutated_inplace(
    toy_layer: TemporalResonanceLayer, toy_cfg: TRNConfig
) -> None:
    """step_single must not mutate the input state tensors in-place."""
    B, d = 2, toy_cfg.d_model
    x  = torch.randn(B, d)
    r_orig  = torch.randn(B, toy_cfg.n_oscillators)
    i_orig  = torch.randn(B, toy_cfg.n_oscillators)
    r_copy  = r_orig.clone()
    i_copy  = i_orig.clone()
    _, r_new, i_new = toy_layer.step_single(x, r_orig, i_orig, position=0)
    assert torch.allclose(r_orig, r_copy), "r_real was mutated in-place"
    assert torch.allclose(i_orig, i_copy), "r_imag was mutated in-place"


# ---------------------------------------------------------------------------
# 8. ADDITIONAL ADVERSARIAL (items not covered above)
# ---------------------------------------------------------------------------

def test_inf_input_to_oscillator(toy_cfg: TRNConfig) -> None:
    """Very large input (1e38) -> softplus clamp keeps A <= 10, all outputs finite."""
    from trn.oscillator import OscillatorProjection
    torch.manual_seed(20)
    proj = OscillatorProjection(toy_cfg.d_model, toy_cfg.n_oscillators).eval()
    B, n = 2, 3
    x = torch.full((B, n, toy_cfg.d_model), 1e38)
    with torch.no_grad():
        A, omega, phi, alpha = proj(x)
    assert torch.all(A <= 10.0 + 1e-4), f"A exceeded clamp max: A.max()={A.max().item()}"
    assert torch.all(torch.isfinite(omega)), "omega is not finite"
    assert torch.all(torch.isfinite(phi)),   "phi is not finite"
    assert torch.all(torch.isfinite(alpha)), "alpha is not finite"


def test_zero_sequence(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    """All-zero input_ids -> output must be finite (zero embeddings are valid)."""
    B, n = 2, 8
    input_ids = torch.zeros(B, n, dtype=torch.long)
    with torch.no_grad():
        result = toy_model(input_ids)
    logits = result["logits"]
    assert logits.shape == (B, n, toy_cfg.vocab_size)
    assert torch.all(torch.isfinite(logits)), "all-zero input produced non-finite logits"


def test_negative_token_ids(toy_model: TRNModel) -> None:
    """Negative token ids — undefined semantics but must not segfault."""
    B, n = 1, 2
    input_ids = torch.tensor([[-1, -2]], dtype=torch.long)
    try:
        with torch.no_grad():
            result = toy_model(input_ids)
        # If it runs, output may be garbage but must have correct leading shape.
        assert result["logits"].shape[0] == B
    except (IndexError, RuntimeError):
        pass  # Clean failure is also acceptable


def test_fp32_state_in_bf16_forward(toy_cfg: TRNConfig) -> None:
    """alpha passed to the scan must be float32 even in a bf16 forward pass.

    Patches trn.resonance.chunked_resonance_scan (the CPU path used by forward).
    """
    import trn.resonance as resonance_module
    from trn.scan import chunked_resonance_scan as _orig_chunked
    torch.manual_seed(21)
    B, n = 1, 4

    layer = TemporalResonanceLayer(
        d_model=toy_cfg.d_model,
        K=toy_cfg.n_oscillators,
        use_parallel_scan=False,
    ).to(torch.bfloat16).eval()

    observed: list[torch.dtype] = []

    def _spy(alpha: torch.Tensor, drive_r: torch.Tensor, drive_i: torch.Tensor, **kwargs):
        observed.append(alpha.dtype)
        return _orig_chunked(alpha, drive_r, drive_i, **kwargs)

    # Patch the name as seen by resonance.py (its local import binding).
    _orig_in_resonance = resonance_module.chunked_resonance_scan
    resonance_module.chunked_resonance_scan = _spy
    try:
        x = torch.randn(B, n, toy_cfg.d_model, dtype=torch.bfloat16)
        with torch.no_grad():
            layer(x)
    finally:
        resonance_module.chunked_resonance_scan = _orig_in_resonance

    assert len(observed) > 0, "scan was never called"
    for dt in observed:
        assert dt == torch.float32, f"alpha dtype in scan was {dt}, expected float32"


def test_combine_absorbing_zero() -> None:
    """_combine((0, b), (a2, b2)) -> first component of result is 0."""
    torch.manual_seed(22)
    shape = (3, 5)
    b_left  = torch.rand(shape)
    a_right = torch.rand(shape)
    b_right = torch.rand(shape)
    zeros   = torch.zeros(shape)
    out_a, _ = _combine((zeros, b_left), (a_right, b_right))
    assert torch.allclose(out_a, zeros, atol=1e-6), (
        f"absorbing zero failed: out_a.max()={out_a.abs().max().item()}"
    )


def test_nan_propagates_through_model(toy_cfg: TRNConfig) -> None:
    """NaN in embedding weight -> logits contain NaN (alias for broader coverage)."""
    model = TRNModel(toy_cfg).eval()
    model.embedding.weight.data[0] = float("nan")
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    out = model(input_ids)
    assert torch.isnan(out["logits"]).any(), (
        "Expected NaN to propagate from corrupted embedding, but logits are clean"
    )
