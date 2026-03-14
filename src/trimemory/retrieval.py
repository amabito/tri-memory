"""RetrievalIndex: sparse exact-fact archive for Tri-Memory LLM.

Stores salient evicted chunks with metadata and supports
multiple search modes: bag-of-token cosine, hidden-state cosine,
or a weighted hybrid of both.

Design rationale:
  - No vector DB dependency (standard library + torch only)
  - Chunk-level storage (not token-level) to bound memory
  - hidden-state cosine for semantic retrieval (default)
  - bag-of-token cosine for lexical retrieval (original v1)
  - hybrid mode for weighted combination
  - Metadata enables rule-based pre-filtering before similarity

Limitations:
  - Not a learned retriever -- rule-based saliency + cosine
  - Max chunks bounded to prevent unbounded growth
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


@dataclass
class ChunkRecord:
    """One archived chunk with metadata."""
    chunk_id: int
    token_ids: list[int]          # raw token IDs in this chunk
    hidden_mean: Tensor           # (d_model,) mean-pooled hidden state
    step: int                     # global step when this chunk was evicted
    saliency: float               # saliency score at eviction time
    tool_name: str = ""           # tool boundary tag if any
    entity_tags: list[str] = field(default_factory=list)
    token_bag: Optional[Tensor] = None  # (vocab_size,) bag-of-tokens vector


class RetrievalIndex:
    """Fixed-capacity chunk archive with bag-of-token cosine search.

    Args:
        vocab_size: vocabulary size for bag-of-token vectors
        max_chunks: maximum stored chunks (FIFO eviction of oldest)
        d_model: hidden dimension for stored embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        max_chunks: int = 256,
        d_model: int = 128,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_chunks = max_chunks
        self.d_model = d_model
        self._chunks: deque[ChunkRecord] = deque(maxlen=max_chunks)
        self._next_id: int = 0

    def __len__(self) -> int:
        return len(self._chunks)

    def _make_token_bag(self, token_ids: list[int]) -> Tensor:
        """Build normalized bag-of-tokens vector from token ID list."""
        bag = torch.zeros(self.vocab_size, dtype=torch.float32)
        for t in token_ids:
            if 0 <= t < self.vocab_size:
                bag[t] += 1.0
        norm = bag.norm()
        if norm > 0:
            bag = bag / norm
        return bag

    def add_chunk(
        self,
        token_ids: list[int],
        hidden_mean: Tensor,
        step: int,
        saliency: float,
        tool_name: str = "",
        entity_tags: Optional[list[str]] = None,
    ) -> int:
        """Archive a chunk. Returns the assigned chunk_id.

        If at capacity, evicts the oldest (lowest chunk_id) entry.
        """
        token_bag = self._make_token_bag(token_ids)

        record = ChunkRecord(
            chunk_id=self._next_id,
            token_ids=token_ids,
            hidden_mean=hidden_mean.detach().cpu().clone(),
            step=step,
            saliency=saliency,
            tool_name=tool_name,
            entity_tags=entity_tags or [],
            token_bag=token_bag,
        )
        self._next_id += 1

        self._chunks.append(record)
        return record.chunk_id

    def _hidden_cosine(
        self, query_hidden: Tensor, chunk: ChunkRecord,
    ) -> float:
        """Cosine similarity between query hidden and chunk hidden_mean."""
        q = query_hidden.flatten().float().cpu()
        c = chunk.hidden_mean.flatten().float()
        q_norm = q.norm()
        c_norm = c.norm()
        if q_norm == 0 or c_norm == 0:
            return 0.0
        return (torch.dot(q, c) / (q_norm * c_norm)).item()

    def search(
        self,
        query_token_ids: list[int],
        top_k: int = 4,
        query_hidden: Optional[Tensor] = None,
        mode: str = "hidden",
        w_hidden: float = 0.7,
        w_bag: float = 0.3,
    ) -> list[ChunkRecord]:
        """Retrieve top-k chunks by similarity.

        Args:
            query_token_ids: current context token IDs for bag-of-token match
            top_k: number of chunks to return
            query_hidden: (d_model,) hidden state for hidden/hybrid modes
            mode: "bag", "hidden", or "hybrid"
            w_hidden: weight for hidden cosine in hybrid mode
            w_bag: weight for bag cosine in hybrid mode

        Returns:
            list of ChunkRecord, sorted by descending similarity
        """
        results, _ = self.search_with_scores(
            query_token_ids, top_k, query_hidden, mode, w_hidden, w_bag,
        )
        return results

    def search_with_scores(
        self,
        query_token_ids: list[int],
        top_k: int = 4,
        query_hidden: Optional[Tensor] = None,
        mode: str = "hidden",
        w_hidden: float = 0.7,
        w_bag: float = 0.3,
    ) -> tuple[list[ChunkRecord], list[dict]]:
        """Retrieve top-k chunks with per-chunk score breakdown.

        Returns:
            (results, score_dicts) where each score_dict has keys:
            hidden_score, bag_score, combined_score
        """
        if not self._chunks:
            return [], []

        query_bag = self._make_token_bag(query_token_ids)
        use_hidden = mode in ("hidden", "hybrid") and query_hidden is not None
        use_bag = mode in ("bag", "hybrid")

        # Fallback: if hidden mode requested but no query_hidden, use bag
        if mode == "hidden" and query_hidden is None:
            use_bag = True

        scored: list[tuple[float, float, float, int]] = []
        for idx, chunk in enumerate(self._chunks):
            h_score = 0.0
            b_score = 0.0
            if use_hidden:
                h_score = self._hidden_cosine(query_hidden, chunk)
            if use_bag or mode == "bag":
                if chunk.token_bag is not None:
                    b_score = torch.dot(query_bag, chunk.token_bag).item()

            if mode == "hybrid":
                combined = w_hidden * h_score + w_bag * b_score
            elif mode == "hidden" and query_hidden is not None:
                combined = h_score
            else:
                combined = b_score

            scored.append((combined, h_score, b_score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        results = [self._chunks[idx] for _, _, _, idx in top]
        score_dicts = [
            {"hidden_score": hs, "bag_score": bs, "combined_score": cs}
            for cs, hs, bs, _ in top
        ]
        return results, score_dicts

    def search_by_metadata(
        self,
        tool_name: Optional[str] = None,
        entity_tag: Optional[str] = None,
        min_saliency: float = 0.0,
        top_k: int = 4,
    ) -> list[ChunkRecord]:
        """Filter chunks by metadata fields."""
        candidates = self._chunks
        if tool_name:
            candidates = [c for c in candidates if c.tool_name == tool_name]
        if entity_tag:
            candidates = [c for c in candidates if entity_tag in c.entity_tags]
        if min_saliency > 0:
            candidates = [c for c in candidates if c.saliency >= min_saliency]
        # Sort by saliency descending
        candidates.sort(key=lambda c: c.saliency, reverse=True)
        return candidates[:top_k]

    def reset(self) -> None:
        """Clear all stored chunks."""
        self._chunks.clear()
        self._next_id = 0

    def get_all_chunks(self) -> list[ChunkRecord]:
        """Return a copy of all stored chunks."""
        return list(self._chunks)

    def remove_chunks(self, keep_fn) -> int:
        """Remove chunks where keep_fn(chunk) returns False.

        Returns number of chunks removed.
        """
        before = len(self._chunks)
        kept = deque((c for c in self._chunks if keep_fn(c)), maxlen=self.max_chunks)
        self._chunks = kept
        return before - len(self._chunks)

    def update_chunk(self, chunk_id: int, **kwargs) -> bool:
        """Update fields on a chunk by chunk_id. Returns True if found."""
        for chunk in self._chunks:
            if chunk.chunk_id == chunk_id:
                for key, value in kwargs.items():
                    if hasattr(chunk, key):
                        setattr(chunk, key, value)
                return True
        return False

    def memory_bytes(self) -> int:
        """Estimate memory usage of stored chunks."""
        if not self._chunks:
            return 0
        per_chunk = (
            4 * 32  # token_ids (avg 32 ints)
            + self.d_model * 4  # hidden_mean fp32
            + self.vocab_size * 4  # token_bag fp32
            + 64  # metadata overhead
        )
        return len(self._chunks) * per_chunk

    def to_summary(self) -> dict:
        """Return summary stats for artifacts."""
        return {
            "num_chunks": len(self._chunks),
            "max_chunks": self.max_chunks,
            "memory_bytes": self.memory_bytes(),
            "oldest_step": self._chunks[0].step if self._chunks else -1,
            "newest_step": self._chunks[-1].step if self._chunks else -1,
        }
