"""Tests for baseline.py — GPT-style Transformer baseline."""
from __future__ import annotations
import torch
import pytest
from trimemory.config import TRNConfig
from trimemory.baseline import TransformerModel


@pytest.fixture
def toy_cfg() -> TRNConfig:
    return TRNConfig.toy()


@pytest.fixture
def model(toy_cfg: TRNConfig) -> TransformerModel:
    m = TransformerModel(toy_cfg)
    m.eval()
    return m


def make_ids(cfg: TRNConfig, B: int = 2, T: int = 16) -> torch.Tensor:
    return torch.randint(0, cfg.vocab_size, (B, T))


# ── Happy Path Tests ─────────────────────────────────────────────────────────


def test_forward_shape(model: TransformerModel, toy_cfg: TRNConfig) -> None:
    """logits shape must be (B, T, vocab_size)."""
    ids = make_ids(toy_cfg)
    result = model(ids)
    assert result["logits"].shape == (2, 16, toy_cfg.vocab_size)


def test_forward_with_labels(model: TransformerModel, toy_cfg: TRNConfig) -> None:
    """forward with labels returns 'loss' key; loss > 0."""
    ids = make_ids(toy_cfg)
    result = model(ids, labels=ids)
    assert "loss" in result
    assert result["loss"].item() > 0.0


def test_num_parameters_positive(model: TransformerModel) -> None:
    """num_parameters() must be positive."""
    assert model.num_parameters() > 0


def test_causal_mask(toy_cfg: TRNConfig) -> None:
    """logits at position 0 must not depend on tokens at positions 1+."""
    model = TransformerModel(toy_cfg)
    model.eval()
    ids = make_ids(toy_cfg, B=1, T=8)
    ids.requires_grad_(False)

    # Check via grad: perturb position 1 embedding and see if logits[:,0] changes
    # We do this by comparing two runs with different future tokens
    ids_a = ids.clone()
    ids_b = ids.clone()
    ids_b[0, 1:] = (ids_b[0, 1:] + 1) % toy_cfg.vocab_size

    with torch.no_grad():
        logits_a = model(ids_a)["logits"]
        logits_b = model(ids_b)["logits"]

    # Position 0 logits must be identical regardless of future tokens
    assert torch.allclose(logits_a[:, 0, :], logits_b[:, 0, :], atol=1e-5), (
        "Causal mask violated: logits at pos 0 differ when future tokens change"
    )


def test_configure_param_groups_two_groups(model: TransformerModel) -> None:
    """configure_optimizer_param_groups must return list of exactly 2 dicts."""
    groups = model.configure_optimizer_param_groups(0.1)
    assert isinstance(groups, list)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0


def test_loss_decreases_after_step(toy_cfg: TRNConfig) -> None:
    """One Adam step must decrease the loss."""
    model = TransformerModel(toy_cfg)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ids = make_ids(toy_cfg)

    result = model(ids, labels=ids)
    loss_before = result["loss"].item()

    result["loss"].backward()
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        result2 = model(ids, labels=ids)
    loss_after = result2["loss"].item()

    assert loss_after < loss_before, (
        f"Loss did not decrease: {loss_before:.4f} -> {loss_after:.4f}"
    )


def test_bf16_compatible(toy_cfg: TRNConfig) -> None:
    """Model in bfloat16 must produce finite logits without crashing."""
    model = TransformerModel(toy_cfg).to(torch.bfloat16)
    model.eval()
    ids = make_ids(toy_cfg)
    with torch.no_grad():
        result = model(ids)
    logits = result["logits"]
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all(), "Non-finite logits in bfloat16 mode"


def test_weight_tying(model: TransformerModel) -> None:
    """embed.weight and lm_head.weight must be the same tensor object."""
    assert model.embed.weight is model.lm_head.weight


# ── Adversarial Tests ────────────────────────────────────────────────────────


def test_single_token_sequence(toy_cfg: TRNConfig) -> None:
    """T=1 input: forward must not crash; 'logits' key must be present."""
    model = TransformerModel(toy_cfg)
    model.eval()
    ids = torch.randint(0, toy_cfg.vocab_size, (1, 1))
    # With T=1, shift produces empty tensors — no RuntimeError expected
    result = model(ids, labels=ids)
    assert "logits" in result
    assert result["logits"].shape == (1, 1, toy_cfg.vocab_size)


def test_zero_input_ids(toy_cfg: TRNConfig) -> None:
    """All-zero input_ids must produce finite logits without crashing."""
    model = TransformerModel(toy_cfg)
    model.eval()
    ids = torch.zeros(2, 16, dtype=torch.long)
    with torch.no_grad():
        result = model(ids)
    assert torch.isfinite(result["logits"]).all()


def test_large_seq_len(toy_cfg: TRNConfig) -> None:
    """Sequence of length max_seq_len must not crash."""
    model = TransformerModel(toy_cfg)
    model.eval()
    ids = make_ids(toy_cfg, B=1, T=toy_cfg.max_seq_len)
    with torch.no_grad():
        result = model(ids)
    assert result["logits"].shape == (1, toy_cfg.max_seq_len, toy_cfg.vocab_size)


def test_num_parameters_non_embedding_less(model: TransformerModel) -> None:
    """num_parameters(non_embedding=True) must be less than num_parameters(non_embedding=False)."""
    total = model.num_parameters(non_embedding=False)
    non_emb = model.num_parameters(non_embedding=True)
    assert non_emb < total, (
        f"non_embedding={non_emb} must be < total={total}"
    )


def test_deterministic_eval(toy_cfg: TRNConfig) -> None:
    """Same input twice in eval mode must yield identical logits."""
    model = TransformerModel(toy_cfg)
    model.eval()
    ids = make_ids(toy_cfg)
    with torch.no_grad():
        out1 = model(ids)["logits"]
        out2 = model(ids)["logits"]
    assert torch.equal(out1, out2), "Eval mode produced different logits for same input"


def test_gradient_flow(toy_cfg: TRNConfig) -> None:
    """loss.backward() must assign non-None grad to all trainable parameters."""
    model = TransformerModel(toy_cfg)
    model.train()
    ids = make_ids(toy_cfg)
    result = model(ids, labels=ids)
    result["loss"].backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for parameter: {name}"
