"""SelectiveMemoryMessenger: builds CompactMemoryPacket from raw retrieval results.

Extracts query-relevant attributes from retrieved chunks, filtering by
query intent (value lookup, reason/history, conflict detection, etc.).
Produces a token-efficient packet instead of raw chunk text.

Design rationale:
  - Rule-based query classification (Japanese + English patterns)
  - No raw text in packet -- only structured fields
  - Deduplication by fact key
  - Status-aware ordering (current first, superseded last)
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

from trimemory.disentangled_archive import ChunkMetadata, MetadataParser
from trimemory.memory_packet import (
    CompactMemoryPacket,
    MemoryFact,
    SourceRef,
    StateHint,
)

logger = logging.getLogger(__name__)

# Query intent patterns
_VALUE_QUERY = re.compile(
    r"(?:値|数値|上限|下限|何個|何時|何mm|何kN|何m3|いくら|何倍|寸法|強度|耐力|支持力"
    r"|how\s+much|how\s+many|what\s+is\s+the\s+value|capacity|strength|limit)",
    re.IGNORECASE,
)
_REASON_QUERY = re.compile(
    r"(?:理由|なぜ|経緯|変更|why|reason|history|changed|背景|根拠|動機)",
    re.IGNORECASE,
)
_CONFLICT_QUERY = re.compile(
    r"(?:矛盾|整合|一致|違い|食い違い|不一致|conflict|inconsisten|discrepan|differ)",
    re.IGNORECASE,
)
_CURRENT_QUERY = re.compile(
    r"(?:現行|最新|正式|有効|effective|current|latest|formal|approved|確定)",
    re.IGNORECASE,
)
_PROVISIONAL_QUERY = re.compile(
    r"(?:暫定|議事録|案|予定|仮|provisional|draft|minutes|pending|tentative)",
    re.IGNORECASE,
)

# Status priority for sorting (lower = higher priority)
_STATUS_PRIORITY: dict[str, int] = {
    "current": 0,
    "unknown": 1,
    "draft": 2,
    "provisional": 3,
    "superseded": 4,
}


class SelectiveMemoryMessenger:
    """Builds CompactMemoryPacket from raw retrieved items.

    Extracted fields depend on query intent:
      - Value queries: prioritize exact_facts
      - Reason queries: prioritize provenance_summary and state_hints
      - Conflict queries: prioritize conflict state_hints
      - Current queries: prioritize status=current facts
      - Provisional queries: include provisional information
    """

    def __init__(self) -> None:
        self._parser = MetadataParser()

    def build_packet(
        self,
        retrieved_items: list[dict[str, Any]],
        query: str,
        max_exact_fact_fields: int = 8,
        max_state_hints: int = 4,
        max_source_refs: int = 6,
    ) -> CompactMemoryPacket:
        """Build a compact memory packet from retrieved items.

        Args:
            retrieved_items: list of dicts, each with at least 'text' key.
                Optional keys: 'doc_id', 'span_id', 'title', 'metadata' (ChunkMetadata dict).
            query: the user query string
            max_exact_fact_fields: max facts in packet
            max_state_hints: max state hints in packet
            max_source_refs: max source references in packet

        Returns:
            CompactMemoryPacket with query-relevant fields populated
        """
        # Classify query intent
        query = str(query) if query is not None else ""
        wants_value = bool(_VALUE_QUERY.search(query))
        wants_reason = bool(_REASON_QUERY.search(query))
        wants_conflict = bool(_CONFLICT_QUERY.search(query))
        wants_current = bool(_CURRENT_QUERY.search(query))
        wants_provisional = bool(_PROVISIONAL_QUERY.search(query))

        # If no specific intent detected, default to value + current
        if not any([wants_value, wants_reason, wants_conflict, wants_current, wants_provisional]):
            wants_value = True
            wants_current = True

        all_facts: list[MemoryFact] = []
        all_hints: list[StateHint] = []
        all_refs: list[SourceRef] = []
        all_anomalies: list[str] = []
        provenance_lines: list[str] = []

        if not retrieved_items:
            retrieved_items = []

        for item in retrieved_items:
            if not isinstance(item, dict):
                continue
            meta = self._get_or_parse_metadata(item)
            doc_id = meta.doc_id or item.get("doc_id", "")
            span_id = meta.span_id or item.get("span_id", "")
            title = meta.title or item.get("title", "")

            # Build source ref
            all_refs.append(SourceRef(
                doc_id=doc_id,
                span_id=span_id,
                title=title,
                status=meta.status,
                provenance=meta.provenance,
            ))

            # Extract facts from metadata
            for fact_str in meta.exact_fact_candidates:
                if not isinstance(fact_str, str) or not fact_str:
                    continue
                if ": " in fact_str:
                    key, value = fact_str.split(": ", 1)
                    key = key.strip()
                    value = value.strip()
                else:
                    key = fact_str.strip()
                    value = fact_str.strip()

                priority = self._compute_fact_priority(
                    meta, wants_value, wants_current, wants_provisional,
                )
                all_facts.append(MemoryFact(
                    key=key,
                    value=value,
                    confidence=meta.source_trust,
                    source_doc_id=doc_id,
                    source_span_id=span_id,
                    source_title=title,
                    status=meta.status,
                    provenance=meta.provenance,
                    priority_score=priority,
                ))

            # Extract entity-value pairs as facts
            for evp in meta.entity_value_pairs:
                if not isinstance(evp, dict):
                    continue
                evp_value = evp.get("value", "")
                evp_unit = evp.get("unit", "")
                val_str = f"{evp_value} {evp_unit}".strip()
                priority = self._compute_fact_priority(
                    meta, wants_value, wants_current, wants_provisional,
                )
                # Boost numeric facts when query wants values
                if wants_value:
                    priority += 0.3
                evp_key = f"numeric_{evp_unit}" if evp_unit else f"numeric_{evp_value}"
                all_facts.append(MemoryFact(
                    key=evp_key,
                    value=val_str,
                    confidence=meta.source_trust,
                    source_doc_id=doc_id,
                    source_span_id=span_id,
                    source_title=title,
                    status=meta.status,
                    provenance=meta.provenance,
                    priority_score=priority,
                ))

            # Build provenance summary line
            if meta.provenance and meta.provenance != "unknown":
                provenance_lines.append(
                    f"[{meta.provenance}] {title} (status={meta.status})"
                )

            # Anomaly tags
            all_anomalies.extend(meta.anomaly_tags)

        # Detect conflicts: same key with different values
        conflict_hints = self._detect_conflicts(all_facts)
        all_hints.extend(conflict_hints)

        # Add reason/change hints if query wants them
        if wants_reason:
            for item in retrieved_items:
                if not isinstance(item, dict):
                    continue
                meta = self._get_or_parse_metadata(item)
                for fact_str in meta.exact_fact_candidates:
                    if not isinstance(fact_str, str) or not fact_str:
                        continue
                    if fact_str.startswith("[change]"):
                        all_hints.append(StateHint(
                            hint_type="pending_change",
                            text=fact_str,
                            confidence=meta.source_trust,
                            source_doc_id=meta.doc_id,
                            source_span_id=meta.span_id,
                        ))

        # Sort facts: by priority_score desc, then status priority
        all_facts.sort(
            key=lambda f: (
                -f.priority_score,
                _STATUS_PRIORITY.get(f.status or "unknown", 9),
            ),
        )

        # Deduplicate facts by (key, value) tuple (keep highest priority)
        seen_keys: set[tuple[str, str]] = set()
        deduped_facts: list[MemoryFact] = []
        for fact in all_facts:
            dedup_key = (fact.key, fact.value)
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                deduped_facts.append(fact)

        # Trim to limits (guard negative)
        max_exact_fact_fields = max(0, max_exact_fact_fields)
        max_state_hints = max(0, max_state_hints)
        max_source_refs = max(0, max_source_refs)
        trimmed_facts = deduped_facts[:max_exact_fact_fields]
        trimmed_hints = all_hints[:max_state_hints]
        trimmed_refs = all_refs[:max_source_refs]

        # Build summary
        summary = self._build_summary(trimmed_facts, trimmed_hints, query)

        return CompactMemoryPacket(
            exact_facts=trimmed_facts,
            state_hints=trimmed_hints,
            anomaly_flags=list(dict.fromkeys(
                str(x) for x in all_anomalies
                if not isinstance(x, (dict, list, set))
            )),
            provenance_summary=provenance_lines[:6],
            source_refs=trimmed_refs,
            packet_summary=summary,
        )

    def _get_or_parse_metadata(self, item: dict[str, Any]) -> ChunkMetadata:
        """Get existing metadata or parse from text."""
        if "metadata" in item and isinstance(item["metadata"], dict):
            return ChunkMetadata.from_dict(item["metadata"])
        if "metadata" in item and isinstance(item["metadata"], ChunkMetadata):
            return item["metadata"]
        # Parse from raw text
        raw_text = item.get("text", "")
        text = str(raw_text) if raw_text is not None else ""
        return self._parser.parse(
            text=text,
            doc_id=item.get("doc_id", ""),
            span_id=item.get("span_id", ""),
            title=item.get("title", ""),
        )

    def _compute_fact_priority(
        self,
        meta: ChunkMetadata,
        wants_value: bool,
        wants_current: bool,
        wants_provisional: bool,
    ) -> float:
        """Compute priority score for a fact based on query intent and metadata."""
        score = meta.source_trust if math.isfinite(meta.source_trust) else 0.5

        if wants_current and meta.status == "current":
            score += 0.4
        elif wants_current and meta.status == "superseded":
            score -= 0.3

        if wants_provisional and meta.status == "provisional":
            score += 0.2

        if wants_value:
            score += 0.1  # mild boost for all facts when value query

        return max(0.0, score)

    def _detect_conflicts(self, facts: list[MemoryFact]) -> list[StateHint]:
        """Detect conflicting values for the same key across different sources."""
        key_values: dict[str, list[MemoryFact]] = {}
        for fact in facts:
            # Group by normalized key (strip numeric_ prefix for entity values)
            norm_key = fact.key
            if norm_key not in key_values:
                key_values[norm_key] = []
            key_values[norm_key].append(fact)

        hints: list[StateHint] = []
        for key, group in key_values.items():
            if len(group) < 2:
                continue
            # Check if values differ across different sources
            unique_values = set(
                f.value for f in group if isinstance(f.value, str)
            )
            unique_sources = set(f.source_doc_id for f in group if f.source_doc_id)
            if len(unique_values) > 1 and len(unique_sources) >= 2:
                seen_src: set[str] = set()
                sources: list[str] = []
                for f in group:
                    s = f"{f.source_title}({f.status})"
                    if s not in seen_src:
                        seen_src.add(s)
                        sources.append(s)
                values_str = ", ".join(f"'{v}'" for v in unique_values)
                hints.append(StateHint(
                    hint_type="conflict",
                    text=f"Key '{key}' has conflicting values: {values_str} from {', '.join(sources)}",
                    confidence=0.8,
                    source_doc_id=group[0].source_doc_id,
                    source_span_id=group[0].source_span_id,
                ))
        return hints

    def _build_summary(
        self,
        facts: list[MemoryFact],
        hints: list[StateHint],
        query: str,
    ) -> str:
        """Build a short textual summary of the packet."""
        lines: list[str] = []

        if facts:
            top_fact = facts[0]
            lines.append(f"Primary: {top_fact.key}={top_fact.value} ({top_fact.status})")

        conflict_count = sum(1 for h in hints if h.hint_type == "conflict")
        if conflict_count > 0:
            lines.append(f"Conflicts detected: {conflict_count}")

        pending_count = sum(1 for h in hints if h.hint_type == "pending_change")
        if pending_count > 0:
            lines.append(f"Pending changes: {pending_count}")

        n_current = sum(1 for f in facts if f.status == "current")
        n_superseded = sum(1 for f in facts if f.status == "superseded")
        if n_current > 0 or n_superseded > 0:
            lines.append(f"Sources: {n_current} current, {n_superseded} superseded")

        return "; ".join(lines) if lines else "No structured facts extracted"
