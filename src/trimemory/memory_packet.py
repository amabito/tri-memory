"""CompactMemoryPacket: structured memory representation for real-data PoC.

Decomposes raw retrieved chunks into typed facts, state hints, and
source references. Designed for token-efficient prompt injection and
downstream conflict resolution.

Design rationale:
  - dataclass-based for readability and JSON round-trip
  - No learned components -- rule-based extraction only
  - Backward compatible: works without metadata via safe defaults
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

logger = logging.getLogger(__name__)

# Valid enum values for documentation and optional validation
VALID_STATUS = frozenset(
    {"current", "superseded", "draft", "provisional", "unknown"}
)
VALID_PROVENANCE = frozenset(
    {"spec", "note", "meeting", "table", "faq", "calc", "unknown"}
)
VALID_HINT_TYPE = frozenset(
    {"anomaly", "trend", "pending_change", "conflict", "unresolved"}
)


@dataclass
class MemoryFact:
    """A single extracted fact with provenance metadata."""

    key: str
    value: str
    confidence: float
    source_doc_id: str
    source_span_id: str
    source_title: str
    timestamp: str | None = None
    status: str | None = None
    provenance: str | None = None
    priority_score: float = 0.0


@dataclass
class StateHint:
    """A state-level observation (anomaly, trend, conflict, etc.)."""

    hint_type: str
    text: str
    confidence: float
    source_doc_id: str
    source_span_id: str


@dataclass
class SourceRef:
    """A reference to a source document/span."""

    doc_id: str
    span_id: str
    title: str
    status: str | None = None
    provenance: str | None = None


@dataclass
class CompactMemoryPacket:
    """Token-efficient structured memory packet.

    Contains typed facts, state hints, anomaly flags, provenance summary,
    source references, and a short textual summary.
    """

    exact_facts: list[MemoryFact] = field(default_factory=list)
    state_hints: list[StateHint] = field(default_factory=list)
    anomaly_flags: list[str] = field(default_factory=list)
    provenance_summary: list[str] = field(default_factory=list)
    source_refs: list[SourceRef] = field(default_factory=list)
    packet_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CompactMemoryPacket:
        """Deserialize from a dict produced by to_dict().

        Tolerates null lists and extra keys gracefully.
        """
        fact_names = {f.name for f in fields(MemoryFact)}
        hint_names = {f.name for f in fields(StateHint)}
        ref_names = {f.name for f in fields(SourceRef)}

        def _safe_fact(f: dict) -> MemoryFact | None:
            filtered = {k: v for k, v in f.items() if k in fact_names}
            # Fill required fields with defaults if missing
            filtered.setdefault("key", "")
            filtered.setdefault("value", "")
            filtered.setdefault("confidence", 0.0)
            filtered.setdefault("source_doc_id", "")
            filtered.setdefault("source_span_id", "")
            filtered.setdefault("source_title", "")
            try:
                filtered["confidence"] = float(filtered["confidence"])
                if not math.isfinite(filtered["confidence"]):
                    filtered["confidence"] = 0.0
            except (TypeError, ValueError, OverflowError):
                filtered["confidence"] = 0.0
            try:
                filtered["priority_score"] = float(filtered.get("priority_score", 0.0))
                if not math.isfinite(filtered["priority_score"]):
                    filtered["priority_score"] = 0.0
            except (TypeError, ValueError, OverflowError):
                filtered["priority_score"] = 0.0
            return MemoryFact(**filtered)

        def _safe_hint(h: dict) -> StateHint | None:
            filtered = {k: v for k, v in h.items() if k in hint_names}
            filtered.setdefault("hint_type", "")
            filtered.setdefault("text", "")
            filtered.setdefault("confidence", 0.0)
            filtered.setdefault("source_doc_id", "")
            filtered.setdefault("source_span_id", "")
            try:
                filtered["confidence"] = float(filtered["confidence"])
                if not math.isfinite(filtered["confidence"]):
                    filtered["confidence"] = 0.0
            except (TypeError, ValueError, OverflowError):
                filtered["confidence"] = 0.0
            return StateHint(**filtered)

        def _safe_ref(r: dict) -> SourceRef | None:
            filtered = {k: v for k, v in r.items() if k in ref_names}
            filtered.setdefault("doc_id", "")
            filtered.setdefault("span_id", "")
            filtered.setdefault("title", "")
            return SourceRef(**filtered)

        return cls(
            exact_facts=[
                _safe_fact(f)
                for f in (d.get("exact_facts") or [])
                if isinstance(f, dict)
            ],
            state_hints=[
                _safe_hint(h)
                for h in (d.get("state_hints") or [])
                if isinstance(h, dict)
            ],
            anomaly_flags=[
                str(x) for x in (d.get("anomaly_flags") or [])
                if not isinstance(x, (dict, list, set))
            ],
            provenance_summary=[str(x) for x in (d.get("provenance_summary") or [])],
            source_refs=[
                _safe_ref(r)
                for r in (d.get("source_refs") or [])
                if isinstance(r, dict)
            ],
            packet_summary=d.get("packet_summary", "") if isinstance(d.get("packet_summary"), str) else "",
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, s: str) -> CompactMemoryPacket:
        """Deserialize from JSON string."""
        parsed = json.loads(s)
        if not isinstance(parsed, dict):
            return cls()
        return cls.from_dict(parsed)

    @property
    def fact_count(self) -> int:
        return len(self.exact_facts)

    @property
    def hint_count(self) -> int:
        return len(self.state_hints)

    def has_conflicts(self) -> bool:
        """Check if any state hints indicate a conflict."""
        return any(h.hint_type == "conflict" for h in self.state_hints)
