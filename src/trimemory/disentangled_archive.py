"""Disentangled archive metadata extension for RetrievalIndex.

Extends ChunkRecord with structured metadata (doc status, provenance,
span references, entity-value pairs) without breaking the existing
RetrievalIndex API.

Design rationale:
  - Metadata stored as a dict on ChunkRecord (backward compatible)
  - Rule-based parser extracts metadata from chunk text at archive time
  - Missing metadata degrades gracefully (defaults to 'unknown')
  - No learned components -- heuristic extraction only
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Patterns for status detection -- order does NOT determine priority.
# _detect_status evaluates all and picks by priority order:
# current > superseded > draft > provisional > unknown.
_STATUS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("superseded", re.compile(r"(?:superseded|obsolete|replaced|withdrawn|旧版)", re.IGNORECASE)),
    ("draft", re.compile(r"(?:draft|案|素案|暫定案|下書き)", re.IGNORECASE)),
    ("provisional", re.compile(r"(?:provisional|暫定|仮|未確定|予定)", re.IGNORECASE)),
    ("current", re.compile(r"(?:current|final|approved|正式|確定|最終|承認済)", re.IGNORECASE)),
]

# Priority order for status resolution when multiple patterns match.
# Higher index = higher priority (current wins over superseded).
_STATUS_PRIORITY_ORDER: list[str] = [
    "unknown", "provisional", "draft", "superseded", "current",
]

# Patterns for provenance detection
_PROVENANCE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("spec", re.compile(r"(?:spec|specification|仕様書|設計書|基準書|Rev\.\d+)", re.IGNORECASE)),
    ("calc", re.compile(r"(?:calculation|計算書|算定|照査)", re.IGNORECASE)),
    ("meeting", re.compile(r"(?:meeting|minutes|議事録|打合せ|会議)", re.IGNORECASE)),
    ("table", re.compile(r"(?:table|一覧|表\s*\d|matrix|台帳)", re.IGNORECASE)),
    ("faq", re.compile(r"(?:faq|q\s*&\s*a|よくある質問)", re.IGNORECASE)),
    ("note", re.compile(r"(?:note|memo|注記|備考|覚書|メモ)", re.IGNORECASE)),
]

# Pattern for "key: value" or "key = value" extraction
_KV_PATTERN = re.compile(
    r"^[\s\-\*]*([A-Za-z\u3000-\u9FFF\uF900-\uFAFF][\w\u3000-\u9FFF\uF900-\uFAFF\s]{1,40}?)"
    r"\s*[:=]\s*(.{1,500})$",
    re.MULTILINE,
)

# Pattern for numeric values with units
_NUMERIC_UNIT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(kN|MPa|mm|cm|m|m2|m3|kg|kgf|t|N|Pa|kPa|kN/m|kN/m2|mm/h|%|度|個|本|枚|式)",
)

# Pattern for before/after change pairs
_CHANGE_PATTERN = re.compile(
    r"(?:変更前|旧|before|old)[\s:：]*(.{1,200}?)[\s]*(?:→|->|⇒|変更後|新|after|new)[\s:：]*(.{1,200})",
    re.IGNORECASE | re.MULTILINE,
)

# Pattern for revision numbers
_REVISION_PATTERN = re.compile(r"Rev\.?\s*(\d+)", re.IGNORECASE)


@dataclass
class ChunkMetadata:
    """Extended metadata for a retrieval chunk.

    All fields are optional -- missing metadata defaults to 'unknown'
    or empty lists. This ensures backward compatibility with chunks
    archived before metadata extraction was added.
    """

    doc_id: str = ""
    span_id: str = ""
    title: str = ""
    status: str = "unknown"
    provenance: str = "unknown"
    recency_bucket: str = "unknown"
    exact_fact_candidates: list[str] = field(default_factory=list)
    entity_value_pairs: list[dict[str, str]] = field(default_factory=list)
    anomaly_tags: list[str] = field(default_factory=list)
    goal_tags: list[str] = field(default_factory=list)
    source_trust: float = 0.5
    references: list[str] = field(default_factory=list)
    revision: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "doc_id": self.doc_id,
            "span_id": self.span_id,
            "title": self.title,
            "status": self.status,
            "provenance": self.provenance,
            "recency_bucket": self.recency_bucket,
            "exact_fact_candidates": self.exact_fact_candidates,
            "entity_value_pairs": self.entity_value_pairs,
            "anomaly_tags": self.anomaly_tags,
            "goal_tags": self.goal_tags,
            "source_trust": self.source_trust,
            "references": self.references,
            "revision": self.revision,
        }

    @staticmethod
    def _safe_trust(raw: Any) -> float:
        """Coerce source_trust to a finite float in [0, 1]."""
        try:
            val = float(raw)
        except (TypeError, ValueError, OverflowError):
            return 0.5
        if not math.isfinite(val):
            return 0.5
        return max(0.0, min(1.0, val))

    @staticmethod
    def _safe_revision(raw: Any) -> int | None:
        """Coerce revision to int or None."""
        if raw is None:
            return None
        if isinstance(raw, bool):
            return None
        try:
            val = int(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        return val

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChunkMetadata:
        """Deserialize from dict."""
        return cls(
            doc_id=d.get("doc_id", ""),
            span_id=d.get("span_id", ""),
            title=d.get("title", ""),
            status=d.get("status", "unknown"),
            provenance=d.get("provenance", "unknown"),
            recency_bucket=d.get("recency_bucket", "unknown"),
            exact_fact_candidates=list(d.get("exact_fact_candidates") or []),
            entity_value_pairs=list(d.get("entity_value_pairs") or []),
            anomaly_tags=[str(x) for x in (d.get("anomaly_tags") or []) if not isinstance(x, (dict, list))],
            goal_tags=list(d.get("goal_tags") or []),
            source_trust=cls._safe_trust(d.get("source_trust", 0.5)),
            references=list(d.get("references") or []),
            revision=cls._safe_revision(d.get("revision")),
        )


class MetadataParser:
    """Rule-based parser that extracts ChunkMetadata from raw text.

    Heuristic-only. No learned components. Works with Japanese and
    English text. Degrades gracefully on unstructured input.
    """

    def parse(
        self,
        text: str,
        doc_id: str = "",
        span_id: str = "",
        title: str = "",
    ) -> ChunkMetadata:
        """Extract metadata from raw chunk text.

        Args:
            text: raw chunk text (may be tokenized then decoded)
            doc_id: external document ID if known
            span_id: span identifier if known
            title: document title if known

        Returns:
            ChunkMetadata with best-effort field population
        """
        text = str(text)[:50_000] if text is not None else ""
        title = str(title)[:1_000] if title is not None else ""
        combined = f"{title} {text}"

        status = self._detect_status(combined)
        provenance = self._detect_provenance(combined)
        fact_candidates = self._extract_kv_pairs_as_strings(text)
        entity_value_pairs = self._extract_entity_values(text)
        revision = self._detect_revision(combined)

        # Trust scoring: formal docs get higher trust
        trust = self._compute_trust(status, provenance)

        return ChunkMetadata(
            doc_id=doc_id,
            span_id=span_id,
            title=title,
            status=status,
            provenance=provenance,
            exact_fact_candidates=fact_candidates,
            entity_value_pairs=entity_value_pairs,
            source_trust=trust,
            revision=revision,
        )

    def _detect_status(self, text: str) -> str:
        """Detect document status from text content.

        Evaluates all patterns and picks by priority order when
        multiple statuses match (current > superseded > draft > provisional).
        """
        matched: list[str] = []
        for status, pattern in _STATUS_PATTERNS:
            if pattern.search(text):
                matched.append(status)
        if not matched:
            return "unknown"
        if len(matched) == 1:
            return matched[0]
        # Pick highest priority among matched
        best = matched[0]
        for s in matched[1:]:
            if _STATUS_PRIORITY_ORDER.index(s) > _STATUS_PRIORITY_ORDER.index(best):
                best = s
        return best

    def _detect_provenance(self, text: str) -> str:
        """Detect document provenance type from text content."""
        for provenance, pattern in _PROVENANCE_PATTERNS:
            if pattern.search(text):
                return provenance
        return "unknown"

    def _extract_kv_pairs_as_strings(self, text: str) -> list[str]:
        """Extract 'key: value' patterns as fact candidate strings."""
        candidates: list[str] = []
        for match in _KV_PATTERN.finditer(text):
            key = match.group(1).strip()
            value = match.group(2).strip()
            if len(key) > 1 and len(value) > 0:
                candidates.append(f"{key}: {value}")
        # Also extract before/after change pairs
        for match in _CHANGE_PATTERN.finditer(text):
            old_val = match.group(1).strip()
            new_val = match.group(2).strip()
            candidates.append(f"[change] {old_val} -> {new_val}")
        return candidates[:20]  # cap to prevent bloat

    def _extract_entity_values(self, text: str) -> list[dict[str, str]]:
        """Extract numeric values with units as entity-value pairs."""
        pairs: list[dict[str, str]] = []
        seen: set[str] = set()
        for match in _NUMERIC_UNIT_PATTERN.finditer(text):
            number = match.group(1)
            unit = match.group(2)
            pair_key = f"{number}{unit}"
            if pair_key not in seen:
                seen.add(pair_key)
                pairs.append({"value": number, "unit": unit})
        return pairs[:20]

    def _detect_revision(self, text: str) -> int | None:
        """Extract revision number if present."""
        match = _REVISION_PATTERN.search(text)
        if match:
            return int(match.group(1))
        return None

    def _compute_trust(self, status: str, provenance: str) -> float:
        """Heuristic trust score based on status and provenance."""
        base = 0.5

        status_bonus = {
            "current": 0.3,
            "draft": -0.1,
            "provisional": -0.05,
            "superseded": -0.2,
            "unknown": 0.0,
        }
        provenance_bonus = {
            "spec": 0.2,
            "calc": 0.15,
            "table": 0.1,
            "meeting": 0.0,
            "note": -0.05,
            "faq": -0.1,
            "unknown": 0.0,
        }

        score = base + status_bonus.get(status, 0.0) + provenance_bonus.get(provenance, 0.0)
        if not math.isfinite(score):
            score = base
        return max(0.0, min(1.0, score))
