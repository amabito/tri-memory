"""RetrievalIndex: sparse exact-fact archive for Tri-Memory LLM.

Stores salient evicted chunks with metadata and supports
bag-of-token-id cosine similarity search.

Design rationale:
  - No vector DB dependency (standard library + torch only)
  - Chunk-level storage (not token-level) to bound memory
  - Bag-of-token cosine for search: works well for exact fact lookup
    (numbers, IDs, URLs) without requiring hidden state storage
  - Metadata enables rule-based pre-filtering before similarity

Limitations:
  - Not a learned retriever -- rule-based saliency + cosine
  - No semantic similarity (bag-of-token is lexical)
  - Max chunks bounded to prevent unbounded growth
"""
from __future__ import annotations

import math
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
        self._chunks: list[ChunkRecord] = []
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

        if len(self._chunks) >= self.max_chunks:
            self._chunks.pop(0)  # FIFO evict oldest

        self._chunks.append(record)
        return record.chunk_id

    def search(
        self,
        query_token_ids: list[int],
        top_k: int = 4,
        query_hidden: Optional[Tensor] = None,
    ) -> list[ChunkRecord]:
        """Retrieve top-k chunks by bag-of-token cosine similarity.

        Args:
            query_token_ids: current context token IDs for bag-of-token match
            top_k: number of chunks to return
            query_hidden: optional hidden state for tiebreaking (unused in v1)

        Returns:
            list of ChunkRecord, sorted by descending similarity
        """
        if not self._chunks:
            return []

        query_bag = self._make_token_bag(query_token_ids)
        query_norm = query_bag.norm()
        if query_norm == 0:
            return self._chunks[:top_k]

        scores: list[tuple[float, int]] = []
        for idx, chunk in enumerate(self._chunks):
            if chunk.token_bag is not None:
                sim = torch.dot(query_bag, chunk.token_bag).item()
            else:
                sim = 0.0
            scores.append((sim, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = [self._chunks[idx] for _, idx in scores[:top_k]]
        return results

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
