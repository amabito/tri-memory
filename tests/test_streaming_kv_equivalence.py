"""Test that streaming with cross-chunk KV cache produces equivalent
results to batch forward for the recent-context window.

This is the critical verification that the KV fix works:
  - Batch forward on full sequence uses _make_window_mask(T=full_len)
  - Streaming forward processes chunk-by-chunk with past_kv cache
  - For positions within the KV window at the end, logits must match.
"""
from __future__ import annotations

import torch

from trimemory.config import TRNConfig
from trimemory.tri_memory import TriMemoryEngine


VOCAB_SIZE = 256
D_MODEL = 64
N_LAYERS = 2
N_OSC = 32
D_FF = 128
WINDOW = 16
CHUNK = 8


def _make_cfg() -> TRNConfig:
    return TRNConfig(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_oscillators=N_OSC,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=256,
    )


def _make_model(enable_trn=True, enable_retrieval=True) -> TriMemoryEngine:
    cfg = _make_cfg()
    return TriMemoryEngine(
        cfg, window_size=WINDOW, chunk_size=CHUNK,
        max_retrieval_chunks=64,
        enable_trn=enable_trn, enable_retrieval=enable_retrieval,
    )


class TestStreamingKVEquivalence:
    """Verify that streaming + KV cache matches batch forward."""

    def test_last_window_logits_match(self):
        """Batch and streaming logits within the last KV window must match.

        The batch forward sees the full sequence with window masking.
        The streaming forward sees chunks with past_kv cache.
        For positions in the last window_size tokens, the KV attention
        input is identical, so logits must be close.
        """
        torch.manual_seed(42)
        model = _make_model(enable_trn=False, enable_retrieval=False)
        model.eval()

        seq_len = 64  # 4 chunks of 8, well beyond window=16
        input_ids = torch.randint(0, VOCAB_SIZE, (1, seq_len))

        # Batch forward
        with torch.inference_mode():
            batch_result = model(input_ids)
        batch_logits = batch_result["logits"]  # (1, seq_len, vocab)

        # Streaming forward
        model.reset_memory()
        B = 1
        K = model.cfg.n_oscillators
        states_r = [torch.zeros(B, K) for _ in range(N_LAYERS)]
        states_i = [torch.zeros(B, K) for _ in range(N_LAYERS)]
        past_kv = None

        all_logits = []
        pos = 0
        with torch.inference_mode():
            for start in range(0, seq_len, CHUNK):
                end = min(start + CHUNK, seq_len)
                chunk = input_ids[:, start:end]
                result, states_r, states_i, past_kv = model.forward_with_memory(
                    chunk, states_r, states_i, pos, past_kv=past_kv,
                )
                all_logits.append(result["logits"])
                pos += (end - start)

        stream_logits = torch.cat(all_logits, dim=1)

        # Compare last WINDOW positions (where KV cache covers full context)
        last_w = WINDOW
        batch_last = batch_logits[0, -last_w:, :]
        stream_last = stream_logits[0, -last_w:, :]

        # They should match closely (same model, same weights, same data)
        max_diff = (batch_last - stream_last).abs().max().item()
        assert max_diff < 0.01, (
            f"Batch vs streaming logits diverge in last window: max_diff={max_diff:.6f}"
        )

    def test_kv_cache_shape_sanity(self):
        """Verify KV cache shapes are correct after streaming."""
        torch.manual_seed(0)
        model = _make_model()
        model.eval()

        B = 1
        K = model.cfg.n_oscillators
        states_r = [torch.zeros(B, K) for _ in range(N_LAYERS)]
        states_i = [torch.zeros(B, K) for _ in range(N_LAYERS)]

        seq_len = 48  # 6 chunks
        input_ids = torch.randint(0, VOCAB_SIZE, (1, seq_len))
        past_kv = None
        pos = 0

        with torch.inference_mode():
            for start in range(0, seq_len, CHUNK):
                end = min(start + CHUNK, seq_len)
                chunk = input_ids[:, start:end]
                _, states_r, states_i, past_kv = model.forward_with_memory(
                    chunk, states_r, states_i, pos, past_kv=past_kv,
                )
                pos += (end - start)

        assert past_kv is not None
        assert len(past_kv) == N_LAYERS

        for layer_idx, (pk, pv) in enumerate(past_kv):
            assert pk.dim() == 4, f"Layer {layer_idx}: past_k should be 4D"
            assert pv.dim() == 4, f"Layer {layer_idx}: past_v should be 4D"
            assert pk.shape[0] == B
            assert pk.shape[2] <= WINDOW, (
                f"Layer {layer_idx}: past_k has {pk.shape[2]} tokens, exceeds window={WINDOW}"
            )
            assert pk.shape == pv.shape

    def test_kv_cache_truncation(self):
        """Verify KV cache is truncated to window_size."""
        torch.manual_seed(1)
        model = _make_model(enable_trn=False, enable_retrieval=False)
        model.eval()

        B = 1
        K = model.cfg.n_oscillators
        states_r = [torch.zeros(B, K) for _ in range(N_LAYERS)]
        states_i = [torch.zeros(B, K) for _ in range(N_LAYERS)]

        # Process many chunks to ensure truncation happens
        seq_len = 128  # 16 chunks, much larger than window=16
        input_ids = torch.randint(0, VOCAB_SIZE, (1, seq_len))
        past_kv = None
        pos = 0

        with torch.inference_mode():
            for start in range(0, seq_len, CHUNK):
                end = min(start + CHUNK, seq_len)
                chunk = input_ids[:, start:end]
                _, states_r, states_i, past_kv = model.forward_with_memory(
                    chunk, states_r, states_i, pos, past_kv=past_kv,
                )
                pos += (end - start)

                # After each chunk, cache must not exceed window_size
                for pk, pv in past_kv:
                    assert pk.shape[2] <= WINDOW

    def test_batch_path_unchanged(self):
        """Batch forward (no past_kv) still works and returns same results."""
        torch.manual_seed(99)
        model = _make_model()
        model.eval()

        input_ids = torch.randint(0, VOCAB_SIZE, (2, 32))

        with torch.inference_mode():
            result = model(input_ids, labels=input_ids)

        assert "logits" in result
        assert "loss" in result
        assert result["logits"].shape == (2, 32, VOCAB_SIZE)
        assert torch.isfinite(result["loss"])

    def test_streaming_with_retrieval_enabled(self):
        """Streaming with retrieval + KV cache produces valid output."""
        torch.manual_seed(7)
        model = _make_model(enable_trn=True, enable_retrieval=True)
        model.eval()

        B = 1
        K = model.cfg.n_oscillators
        states_r = [torch.zeros(B, K) for _ in range(N_LAYERS)]
        states_i = [torch.zeros(B, K) for _ in range(N_LAYERS)]

        seq_len = 64
        input_ids = torch.randint(0, VOCAB_SIZE, (1, seq_len))
        past_kv = None
        pos = 0
        all_logits = []

        with torch.inference_mode():
            for start in range(0, seq_len, CHUNK):
                end = min(start + CHUNK, seq_len)
                chunk = input_ids[:, start:end]
                result, states_r, states_i, past_kv = model.forward_with_memory(
                    chunk, states_r, states_i, pos, past_kv=past_kv,
                )
                all_logits.append(result["logits"])
                pos += (end - start)

        full_logits = torch.cat(all_logits, dim=1)
        assert full_logits.shape == (1, seq_len, VOCAB_SIZE)
        assert not torch.isnan(full_logits).any()
        assert not torch.isinf(full_logits).any()
