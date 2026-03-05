"""Advanced generation utilities for TRNModel.

Provides top-p (nucleus) sampling, temperature scaling, streaming
generation, and a stateful GenerationConfig for multi-turn inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import TRNModel


@dataclass
class GenerationConfig:
    """Sampling parameters for generation."""

    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 0           # 0 = disabled
    top_p: float = 1.0       # 1.0 = disabled (nucleus sampling)
    do_sample: bool = True    # False = greedy
    repetition_penalty: float = 1.0  # 1.0 = disabled


def _apply_top_p(logits: Tensor, top_p: float) -> Tensor:
    """Apply nucleus (top-p) filtering to logits.

    Keeps the smallest set of tokens whose cumulative probability >= top_p.
    All other token logits are set to -inf.

    Args:
        logits: (B, vocab_size) float logits
        top_p:  Nucleus threshold in (0, 1]. Values >= 1.0 are a no-op.
                Values <= 0.0 keep only the single highest-probability token.

    Returns:
        Filtered logits of the same shape.
    """
    if top_p >= 1.0:
        return logits
    # Guard against top_p=0 or negative: treat as "keep top-1 only" (greedy-like).
    if top_p <= 0.0:
        top_p = 1e-8

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens whose cumulative probability already exceeds top_p.
    # Shift one position right so the first token that pushes cumprob above
    # top_p is still retained.
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[sorted_indices_to_remove] = float("-inf")

    # Scatter filtered logits back to the original vocab ordering.
    return logits.scatter(-1, sorted_indices, sorted_logits)


def _apply_repetition_penalty(
    logits: Tensor,
    input_ids: Tensor,
    penalty: float,
) -> Tensor:
    """Penalize tokens that have already appeared in the sequence.

    Positive logits are divided by the penalty; negative logits are multiplied
    by it. This consistently reduces the probability of already-seen tokens.

    Args:
        logits:    (B, vocab_size) float logits
        input_ids: (B, T) integer ids of the context (seen tokens)
        penalty:   Values > 1.0 reduce probability of repeated tokens.

    Returns:
        Modified logits of the same shape.
    """
    if penalty == 1.0:
        return logits

    # Gather current scores for tokens in the context.
    score = logits.gather(-1, input_ids)
    # Reduce positive scores, increase negative scores.
    score = torch.where(score < 0, score * penalty, score / penalty)
    return logits.scatter(-1, input_ids, score)


def sample_token(
    logits: Tensor,
    cfg: GenerationConfig,
    generated_ids: Optional[Tensor] = None,
) -> Tensor:
    """Apply sampling strategy and return next-token ids.

    Processing order: repetition penalty → greedy/temperature → top-k → top-p
    → multinomial sample.

    Args:
        logits:        (B, vocab_size) raw logits
        cfg:           Sampling hyper-parameters
        generated_ids: (B, T) already-generated token ids for repetition penalty

    Returns:
        next_token: (B,) integer tensor of selected token ids
    """
    logits = logits.clone().float()

    if generated_ids is not None and cfg.repetition_penalty != 1.0:
        logits = _apply_repetition_penalty(logits, generated_ids, cfg.repetition_penalty)

    if not cfg.do_sample:
        return logits.argmax(dim=-1)

    if cfg.temperature != 1.0:
        logits = logits / cfg.temperature

    if cfg.top_k > 0:
        top_vals, _ = torch.topk(logits, min(cfg.top_k, logits.size(-1)))
        logits[logits < top_vals[:, -1:]] = float("-inf")

    if cfg.top_p < 1.0:
        logits = _apply_top_p(logits, cfg.top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.inference_mode()
def generate(
    model: TRNModel,
    prompt_ids: Tensor,
    gen_cfg: GenerationConfig = GenerationConfig(),
    device: str = "cpu",
) -> Tensor:
    """Generate tokens autoregressively using step_single (O(1) memory per token).

    The prompt is processed token-by-token via step_single to build an accurate
    resonance state before generation begins.

    Args:
        model:      Trained TRNModel (eval mode enforced internally)
        prompt_ids: (B, prompt_len) integer prompt tensor
        gen_cfg:    Sampling configuration
        device:     Target device for tensors

    Returns:
        generated: (B, max_new_tokens) — only the newly generated tokens
    """
    model.eval()

    B = prompt_ids.size(0)
    K = model.cfg.n_oscillators
    n_layers = model.cfg.n_layers
    prompt_len = prompt_ids.size(1)

    prompt_ids = prompt_ids.to(device)
    param_dtype = next(model.parameters()).dtype

    # Resonance states are always kept in fp32 (see TemporalResonanceLayer docstring).
    states_r = [torch.zeros(B, K, device=device) for _ in range(n_layers)]
    states_i = [torch.zeros(B, K, device=device) for _ in range(n_layers)]

    # Warm up resonance state by stepping through all prompt tokens except the last.
    # The last prompt token is the seed for the first generation step.
    for pos in range(prompt_len - 1):
        token = prompt_ids[:, pos]  # (B,)
        x = model.drop_emb(model.embedding(token).to(param_dtype))
        for layer_idx, block in enumerate(model.blocks):
            x_n = block.norm1(x)
            res_out, states_r[layer_idx], states_i[layer_idx] = (
                block.resonance.step_single(
                    x_n, states_r[layer_idx], states_i[layer_idx], pos
                )
            )
            x = x + block.drop(res_out)
            x = x + block.drop(block.ffn(block.norm2(x)))

    generated: list[Tensor] = []
    current_ids = prompt_ids  # (B, T) — grows as we generate

    for step in range(gen_cfg.max_new_tokens):
        pos = prompt_len - 1 + step
        # First step uses the last prompt token; subsequent steps use the last
        # generated token.
        token = prompt_ids[:, -1] if step == 0 else generated[-1]  # (B,)

        x = model.drop_emb(model.embedding(token).to(param_dtype))
        for layer_idx, block in enumerate(model.blocks):
            x_n = block.norm1(x)
            res_out, states_r[layer_idx], states_i[layer_idx] = (
                block.resonance.step_single(
                    x_n, states_r[layer_idx], states_i[layer_idx], pos
                )
            )
            x = x + block.drop(res_out)
            x = x + block.drop(block.ffn(block.norm2(x)))

        logits = model.lm_head(model.norm_out(x))  # (B, vocab_size)
        next_tok = sample_token(logits, gen_cfg, generated_ids=current_ids)
        generated.append(next_tok)
        current_ids = torch.cat([current_ids, next_tok.unsqueeze(1)], dim=1)

    if not generated:
        return torch.empty(B, 0, dtype=torch.long, device=device)
    return torch.stack(generated, dim=1)  # (B, max_new_tokens)


@torch.inference_mode()
def stream_generate(
    model: TRNModel,
    prompt_ids: Tensor,
    gen_cfg: GenerationConfig = GenerationConfig(),
    device: str = "cpu",
) -> Iterator[int]:
    """Stream generated token ids one at a time (single-sequence only).

    Memory-efficient: only one token is processed per step, and the resonance
    state is updated in-place.

    Args:
        model:      Trained TRNModel (eval mode enforced internally)
        prompt_ids: (1, prompt_len) single-sequence prompt
        gen_cfg:    Sampling configuration
        device:     Target device

    Yields:
        token_id: int for each generated token
    """
    assert prompt_ids.size(0) == 1, "stream_generate only supports batch_size=1"

    model.eval()

    K = model.cfg.n_oscillators
    n_layers = model.cfg.n_layers
    prompt_len = prompt_ids.size(1)

    prompt_ids = prompt_ids.to(device)
    param_dtype = next(model.parameters()).dtype

    states_r = [torch.zeros(1, K, device=device) for _ in range(n_layers)]
    states_i = [torch.zeros(1, K, device=device) for _ in range(n_layers)]

    # Warm up resonance state on prompt (all but last token).
    for pos in range(prompt_len - 1):
        token = prompt_ids[:, pos]  # (1,)
        x = model.drop_emb(model.embedding(token).to(param_dtype))
        for layer_idx, block in enumerate(model.blocks):
            x_n = block.norm1(x)
            res_out, states_r[layer_idx], states_i[layer_idx] = (
                block.resonance.step_single(
                    x_n, states_r[layer_idx], states_i[layer_idx], pos
                )
            )
            x = x + block.drop(res_out)
            x = x + block.drop(block.ffn(block.norm2(x)))

    current_ids = prompt_ids  # (1, T)
    last_tok: int = -1

    for step in range(gen_cfg.max_new_tokens):
        pos = prompt_len - 1 + step

        if step == 0:
            token = prompt_ids[:, -1]  # (1,)
        else:
            token = torch.tensor([last_tok], device=device)  # (1,)

        x = model.drop_emb(model.embedding(token).to(param_dtype))
        for layer_idx, block in enumerate(model.blocks):
            x_n = block.norm1(x)
            res_out, states_r[layer_idx], states_i[layer_idx] = (
                block.resonance.step_single(
                    x_n, states_r[layer_idx], states_i[layer_idx], pos
                )
            )
            x = x + block.drop(res_out)
            x = x + block.drop(block.ffn(block.norm2(x)))

        logits = model.lm_head(model.norm_out(x))  # (1, vocab_size)
        next_tok = sample_token(logits, gen_cfg, generated_ids=current_ids)
        last_tok = next_tok.item()
        current_ids = torch.cat([current_ids, next_tok.unsqueeze(1)], dim=1)
        yield last_tok
