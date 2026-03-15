"""Adversarial tests for deque-bounded data structures in TRN.

Targets three structures refactored from list to deque(maxlen=...):
  - RetrievalIndex._chunks          deque(maxlen=max_chunks)
  - TriMemoryEngine._eviction_buffer deque(maxlen=chunk_size*2)
  - TriMemoryBlock._window_mask_cache dict with LRU eviction at 16 entries

Test categories (attacker mindset):
  1. Retrieval overflow: capacity is silently enforced, oldest entries vanish
  2. Search after overflow: deque index access on shifted contents is valid
  3. Empty / top_k > len edge cases: no crash, sane return values
  4. Window mask cache LRU eviction: bounded at 16, correctness preserved
  5. Deque maxlen boundary (off-by-one): exactly maxlen+1 causes first eviction
  6. Concurrent read/write: no crash, no invalid tensor references
"""
from __future__ import annotations

import threading
import time

import pytest
import torch

from trimemory.config import TRNConfig
from trimemory.retrieval import ChunkRecord, RetrievalIndex
from trimemory.tri_memory import TriMemoryBlock, TriMemoryEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index(max_chunks: int = 8, vocab_size: int = 256, d_model: int = 32) -> RetrievalIndex:
    return RetrievalIndex(vocab_size=vocab_size, max_chunks=max_chunks, d_model=d_model)


def _make_hidden(d_model: int = 32) -> torch.Tensor:
    return torch.randn(d_model)


def _add_chunks(index: RetrievalIndex, n: int, d_model: int = 32) -> list[int]:
    """Add n chunks and return assigned chunk_ids."""
    ids = []
    for i in range(n):
        cid = index.add_chunk(
            token_ids=[i % 256],
            hidden_mean=_make_hidden(d_model),
            step=i,
            saliency=float(i),
        )
        ids.append(cid)
    return ids


# ---------------------------------------------------------------------------
# Category 1: Retrieval overflow -- capacity never exceeded
# ---------------------------------------------------------------------------

class TestRetrievalOverflow:
    """deque(maxlen=max_chunks) silently drops oldest when full."""

    def test_len_never_exceeds_max_chunks(self) -> None:
        """Adding max_chunks+10 items must not grow the deque past max_chunks."""
        max_chunks = 8
        index = _make_index(max_chunks=max_chunks)
        _add_chunks(index, max_chunks + 10)
        assert len(index) == max_chunks, (
            f"Expected len={max_chunks}, got {len(index)}"
        )

    def test_oldest_chunk_ids_are_evicted(self) -> None:
        """After overflow, the oldest chunk_ids (lowest) must no longer exist."""
        max_chunks = 4
        index = _make_index(max_chunks=max_chunks)
        all_ids = _add_chunks(index, max_chunks + 3)

        surviving_ids = {c.chunk_id for c in index.get_all_chunks()}
        # The first (max_chunks+3) - max_chunks = 3 IDs must be gone
        evicted_ids = set(all_ids[: len(all_ids) - max_chunks])
        for cid in evicted_ids:
            assert cid not in surviving_ids, f"chunk_id={cid} should have been evicted"

    def test_newest_chunks_survive_overflow(self) -> None:
        """The most recently added chunks must all be present after overflow."""
        max_chunks = 4
        index = _make_index(max_chunks=max_chunks)
        all_ids = _add_chunks(index, max_chunks + 3)

        surviving_ids = {c.chunk_id for c in index.get_all_chunks()}
        newest_ids = set(all_ids[-max_chunks:])
        for cid in newest_ids:
            assert cid in surviving_ids, f"chunk_id={cid} should survive"

    def test_to_summary_reflects_overflow(self) -> None:
        """to_summary() must report num_chunks == max_chunks after overflow."""
        max_chunks = 6
        index = _make_index(max_chunks=max_chunks)
        _add_chunks(index, max_chunks + 5)
        summary = index.to_summary()
        assert summary["num_chunks"] == max_chunks
        assert summary["oldest_step"] >= 0
        assert summary["newest_step"] >= summary["oldest_step"]


# ---------------------------------------------------------------------------
# Category 2: Search after overflow -- no stale references
# ---------------------------------------------------------------------------

class TestSearchAfterOverflow:
    """After FIFO eviction, all deque indices used in search must be valid."""

    def test_bag_search_after_overflow_returns_results(self) -> None:
        """Bag-of-token search must succeed and return valid ChunkRecords."""
        max_chunks = 4
        index = _make_index(max_chunks=max_chunks, vocab_size=64)
        _add_chunks(index, max_chunks + 6, d_model=32)

        results = index.search(query_token_ids=[0, 1, 2], top_k=2, mode="bag")
        assert len(results) <= 2
        for r in results:
            assert isinstance(r, ChunkRecord)
            assert r.token_bag is not None
            assert r.hidden_mean.shape == (32,)

    def test_hidden_search_after_overflow_returns_results(self) -> None:
        """Hidden-state search after overflow must return valid ChunkRecords."""
        max_chunks = 5
        d_model = 32
        index = _make_index(max_chunks=max_chunks, d_model=d_model)
        _add_chunks(index, max_chunks + 4, d_model=d_model)

        query_hidden = torch.randn(d_model)
        results = index.search(
            query_token_ids=[1, 2],
            top_k=3,
            query_hidden=query_hidden,
            mode="hidden",
        )
        assert len(results) <= min(3, max_chunks)
        for r in results:
            assert r.hidden_mean.shape == (d_model,)

    def test_search_with_scores_after_overflow_indices_in_range(self) -> None:
        """Internal deque indexing in search_with_scores must stay in bounds."""
        max_chunks = 4
        d_model = 16
        index = _make_index(max_chunks=max_chunks, vocab_size=64, d_model=d_model)
        _add_chunks(index, max_chunks + 8, d_model=d_model)

        query_hidden = torch.randn(d_model)
        results, scores = index.search_with_scores(
            query_token_ids=[5, 10],
            top_k=4,
            query_hidden=query_hidden,
            mode="hybrid",
        )
        assert len(results) == len(scores)
        for score_dict in scores:
            assert "hidden_score" in score_dict
            assert "bag_score" in score_dict
            assert "combined_score" in score_dict
            # Scores are finite (no NaN from evicted garbage references)
            assert not (score_dict["combined_score"] != score_dict["combined_score"]), \
                "NaN combined_score -- possible stale tensor reference"

    def test_metadata_search_after_overflow_never_crashes(self) -> None:
        """search_by_metadata must not crash when deque was recently truncated."""
        max_chunks = 3
        index = _make_index(max_chunks=max_chunks)
        for i in range(max_chunks + 5):
            index.add_chunk(
                token_ids=[i],
                hidden_mean=_make_hidden(),
                step=i,
                saliency=0.9 if i % 2 == 0 else 0.1,
                tool_name="tool_A" if i < 4 else "tool_B",
            )
        # All of the tool_A chunks may have been evicted; must not crash
        results = index.search_by_metadata(tool_name="tool_A", top_k=4)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Category 3: Empty / top_k > len edge cases
# ---------------------------------------------------------------------------

class TestRetrievalEmptyEdge:
    """Degenerate inputs must never crash and must return sensible values."""

    def test_search_on_empty_index_returns_empty(self) -> None:
        index = _make_index()
        results = index.search(query_token_ids=[1, 2, 3], top_k=4, mode="bag")
        assert results == []

    def test_search_with_scores_on_empty_returns_empty_pair(self) -> None:
        index = _make_index()
        results, scores = index.search_with_scores(
            query_token_ids=[0], top_k=8, mode="hidden",
            query_hidden=torch.randn(32),
        )
        assert results == []
        assert scores == []

    def test_top_k_greater_than_len_returns_all(self) -> None:
        """When top_k > actual chunks, return all available chunks."""
        index = _make_index(max_chunks=10)
        _add_chunks(index, 3)
        results = index.search(query_token_ids=[0], top_k=99, mode="bag")
        assert len(results) == 3

    def test_search_empty_query_tokens_no_crash(self) -> None:
        """Empty query_token_ids should not crash bag search."""
        index = _make_index(max_chunks=8)
        _add_chunks(index, 4)
        results = index.search(query_token_ids=[], top_k=2, mode="bag")
        assert isinstance(results, list)

    def test_search_zero_top_k_returns_empty(self) -> None:
        """top_k=0 must return empty list."""
        index = _make_index()
        _add_chunks(index, 4)
        results = index.search(query_token_ids=[1], top_k=0, mode="bag")
        assert results == []

    def test_reset_then_search_returns_empty(self) -> None:
        """After reset(), deque is cleared; search must return empty."""
        index = _make_index()
        _add_chunks(index, 6)
        assert len(index) == 6
        index.reset()
        assert len(index) == 0
        results = index.search(query_token_ids=[1], top_k=4, mode="bag")
        assert results == []


# ---------------------------------------------------------------------------
# Category 4: Window mask cache LRU eviction
# ---------------------------------------------------------------------------

class TestWindowMaskCacheEviction:
    """_window_mask_cache is a plain dict with manual LRU eviction at 16."""

    @pytest.fixture
    def block(self) -> TriMemoryBlock:
        cfg = TRNConfig.toy()
        return TriMemoryBlock(cfg, window_size=8, enable_trn=False, enable_retrieval=False)

    def _unique_masks(self, block: TriMemoryBlock, n: int) -> list[torch.Tensor]:
        """Generate n masks for distinct (T, W) pairs."""
        masks = []
        for i in range(n):
            T = 4 + i        # T in [4, 4+n)
            W = 2 + (i % 4)  # W varies but stays < T
            mask = block._make_window_mask(T, W, torch.device("cpu"))
            masks.append((T, W, mask))
        return masks

    def test_cache_size_stays_at_most_16_after_20_entries(self, block: TriMemoryBlock) -> None:
        """Cache must never exceed _MASK_CACHE_MAX=16 entries."""
        self._unique_masks(block, 20)
        assert len(block._window_mask_cache) <= TriMemoryBlock._MASK_CACHE_MAX

    def test_cache_size_exactly_16_after_filling(self, block: TriMemoryBlock) -> None:
        """After inserting exactly _MASK_CACHE_MAX entries the cache is full."""
        self._unique_masks(block, TriMemoryBlock._MASK_CACHE_MAX)
        assert len(block._window_mask_cache) == TriMemoryBlock._MASK_CACHE_MAX

    def test_mask_correctness_after_eviction(self, block: TriMemoryBlock) -> None:
        """Masks computed after cache eviction must still be numerically correct."""
        # Fill cache past capacity so early keys are evicted
        self._unique_masks(block, 20)
        # Request a fresh mask and verify its correctness manually
        T, W = 10, 4
        mask = block._make_window_mask(T, W, torch.device("cpu"))
        assert mask.shape == (T, T)
        # Causal + window: row i, col j is 0 iff j <= i and j >= i - W + 1
        for i in range(T):
            for j in range(T):
                expected = 0.0 if (j <= i and j >= i - W + 1) else float("-inf")
                actual = mask[i, j].item()
                assert actual == expected, (
                    f"mask[{i},{j}] expected {expected}, got {actual}"
                )

    def test_evicted_entry_is_recomputed_correctly(self, block: TriMemoryBlock) -> None:
        """A key evicted by LRU must be recomputed correctly on re-access."""
        # The first key we insert will be evicted once cache is full
        T_first, W_first = 4, 2
        first_mask = block._make_window_mask(T_first, W_first, torch.device("cpu")).clone()

        # Overflow the cache to evict first entry
        for i in range(TriMemoryBlock._MASK_CACHE_MAX + 2):
            block._make_window_mask(5 + i, 3, torch.device("cpu"))

        # Recompute the first mask (cache miss -> fresh computation)
        recomputed = block._make_window_mask(T_first, W_first, torch.device("cpu"))
        assert torch.allclose(first_mask, recomputed), (
            "Recomputed mask after eviction differs from original -- computation is not deterministic"
        )

    def test_cache_hit_returns_same_tensor_object(self, block: TriMemoryBlock) -> None:
        """Second call with same (T, W) must return the exact cached tensor."""
        T, W = 8, 4
        m1 = block._make_window_mask(T, W, torch.device("cpu"))
        m2 = block._make_window_mask(T, W, torch.device("cpu"))
        assert m1 is m2, "Expected cache hit to return identical tensor object"


# ---------------------------------------------------------------------------
# Category 5: Deque maxlen boundary (off-by-one)
# ---------------------------------------------------------------------------

class TestDequeBoundary:
    """Exactly maxlen items fills the deque; one more evicts the first."""

    def test_exactly_maxlen_items_no_eviction(self) -> None:
        """Filling to exactly maxlen must retain all items."""
        max_chunks = 5
        index = _make_index(max_chunks=max_chunks)
        ids = _add_chunks(index, max_chunks)
        assert len(index) == max_chunks
        surviving = {c.chunk_id for c in index.get_all_chunks()}
        for cid in ids:
            assert cid in surviving

    def test_maxlen_plus_one_evicts_exactly_one(self) -> None:
        """maxlen+1 additions must evict exactly the first entry."""
        max_chunks = 5
        index = _make_index(max_chunks=max_chunks)
        ids = _add_chunks(index, max_chunks + 1)
        assert len(index) == max_chunks
        surviving = {c.chunk_id for c in index.get_all_chunks()}
        # The very first id must be gone
        assert ids[0] not in surviving, "First chunk must be evicted at maxlen+1"
        # All others must be present
        for cid in ids[1:]:
            assert cid in surviving

    def test_eviction_buffer_bounded_at_chunk_size_x2(self) -> None:
        """_eviction_buffer deque must never exceed maxlen=chunk_size*2."""
        cfg = TRNConfig.toy()
        chunk_size = 16
        engine = TriMemoryEngine(
            cfg,
            window_size=32,
            chunk_size=chunk_size,
            max_retrieval_chunks=64,
            enable_trn=False,
            enable_retrieval=False,
        )
        maxlen_expected = chunk_size * 2
        # Append many tokens manually (bypassing forward pass) to stress the buffer
        for i in range(maxlen_expected + 50):
            engine._eviction_buffer.append(i % cfg.vocab_size)
        assert len(engine._eviction_buffer) <= maxlen_expected, (
            f"_eviction_buffer exceeded maxlen: {len(engine._eviction_buffer)} > {maxlen_expected}"
        )

    def test_router_log_bounded_at_1024(self) -> None:
        """_router_log must stay bounded at its declared maxlen=1024."""
        from trimemory.router import RouterDecision
        cfg = TRNConfig.toy()
        engine = TriMemoryEngine(
            cfg, window_size=32, chunk_size=16,
            enable_trn=False, enable_retrieval=False,
        )
        # Inject synthetic RouterDecision objects directly
        for i in range(1200):
            engine._router_log.append(
                RouterDecision(
                    g_kv=0.0,
                    g_trn=0.0,
                    g_ret=1.0,
                    reason="test",
                )
            )
        assert len(engine._router_log) == 1024, (
            f"_router_log exceeded 1024: {len(engine._router_log)}"
        )


# ---------------------------------------------------------------------------
# Category 6: Concurrent read/write -- no crash, no invalid state
# ---------------------------------------------------------------------------

class TestConcurrentReadWrite:
    """Multiple reader threads searching while a writer adds chunks.

    Python's deque raises RuntimeError("deque mutated during iteration")
    when a concurrent append happens inside a for-loop iteration.
    RetrievalIndex.search_with_scores iterates self._chunks without a lock,
    so concurrent add_chunk() during search IS a real bug.

    These tests document the existing unsafe behaviour.
    They are marked xfail(strict=True) so:
      - if the bug is still present: xfail (expected, test suite green)
      - if the bug is FIXED (e.g. by adding a lock): xpass -> strict=True
        turns it into a failure, forcing removal of the xfail marker.
    """

    def test_concurrent_search_while_adding_no_crash(self) -> None:
        """10 reader threads searching while 1 writer adds chunks: no exception.

        Fixed: search_with_scores now snapshots deque to list before iteration.
        """
        max_chunks = 16
        d_model = 32
        index = _make_index(max_chunks=max_chunks, vocab_size=128, d_model=d_model)
        # Seed the index so readers have something to work with
        _add_chunks(index, max_chunks // 2, d_model=d_model)

        errors: list[Exception] = []

        def reader() -> None:
            for _ in range(20):
                try:
                    query_hidden = torch.randn(d_model)
                    index.search(
                        query_token_ids=[1, 2, 3],
                        top_k=4,
                        query_hidden=query_hidden,
                        mode="hidden",
                    )
                except Exception as exc:
                    errors.append(exc)

        def writer() -> None:
            for i in range(40):
                index.add_chunk(
                    token_ids=[i % 128],
                    hidden_mean=torch.randn(d_model),
                    step=i + 100,
                    saliency=0.5,
                )
                time.sleep(0.0)  # yield GIL

        threads = [threading.Thread(target=reader) for _ in range(10)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Exceptions during concurrent access: {errors}"

    def test_concurrent_results_contain_valid_records(self) -> None:
        """All ChunkRecords returned during concurrent ops must be coherent.

        Fixed: search_with_scores now snapshots deque to list before iteration.
        """
        max_chunks = 8
        d_model = 16
        index = _make_index(max_chunks=max_chunks, vocab_size=64, d_model=d_model)
        _add_chunks(index, max_chunks, d_model=d_model)

        bad_records: list[str] = []

        def reader() -> None:
            for _ in range(15):
                try:
                    recs = index.search(
                        query_token_ids=[0],
                        top_k=4,
                        query_hidden=torch.randn(d_model),
                        mode="hidden",
                    )
                    for rec in recs:
                        if rec is None:
                            bad_records.append("None record returned")
                        elif rec.hidden_mean is None:
                            bad_records.append(f"chunk_id={rec.chunk_id} has None hidden_mean")
                        elif rec.hidden_mean.shape != (d_model,):
                            bad_records.append(
                                f"chunk_id={rec.chunk_id} bad shape {rec.hidden_mean.shape}"
                            )
                except Exception as exc:
                    bad_records.append(str(exc))

        def writer() -> None:
            for i in range(30):
                index.add_chunk(
                    token_ids=[i % 64],
                    hidden_mean=torch.randn(d_model),
                    step=i + 200,
                    saliency=0.7,
                )

        threads = [threading.Thread(target=reader) for _ in range(8)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert bad_records == [], f"Invalid records found: {bad_records}"
