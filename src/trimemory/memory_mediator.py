"""MemoryMediator: deterministic conflict resolution for CompactMemoryPacket.

Resolves conflicts between retrieval results, TRN state hints, and
working memory using freshness, provenance, and status-based rules.
No learned components -- pure rule-based arbitration.

Design rationale:
  - Immutable input: never modifies the input packet
  - Freshness/status/provenance scoring is additive and transparent
  - Conflicts are preserved as hints, not silently dropped
  - Quota-aware trimming respects priority ordering
"""
from __future__ import annotations

import logging
import math
from typing import Any

from trimemory.memory_packet import (
    CompactMemoryPacket,
    MemoryFact,
    StateHint,
)

logger = logging.getLogger(__name__)

# Status weights: higher = more authoritative
_STATUS_WEIGHT: dict[str, float] = {
    "current": 1.0,
    "unknown": 0.5,
    "draft": 0.3,
    "provisional": 0.2,
    "superseded": 0.1,
}

# Provenance weights: higher = more authoritative
_PROVENANCE_WEIGHT: dict[str, float] = {
    "spec": 1.0,
    "calc": 0.9,
    "table": 0.7,
    "meeting": 0.5,
    "note": 0.3,
    "faq": 0.2,
    "unknown": 0.4,
}


class MemoryMediator:
    """Deterministic conflict resolution and re-scoring of memory packets.

    Operates on CompactMemoryPacket only. Does not access raw chunks
    or the retrieval index directly.
    """

    def resolve(
        self,
        packet: CompactMemoryPacket,
        trn_state: dict[str, Any] | None = None,
        working_memory: dict[str, Any] | None = None,
        prefer_current: bool = True,
        max_exact_facts: int = 8,
        max_state_hints: int = 4,
        max_source_refs: int = 6,
    ) -> CompactMemoryPacket:
        """Resolve conflicts and re-score a memory packet.

        Args:
            packet: input packet (not modified)
            trn_state: optional TRN state dict for additional signals
            working_memory: optional working memory dict for context
            prefer_current: if True, boost current/final status
            max_exact_facts: quota for output facts
            max_state_hints: quota for output hints
            max_source_refs: quota for output source refs

        Returns:
            New CompactMemoryPacket with re-scored and trimmed contents.
        """
        if packet is None:
            return CompactMemoryPacket()
        max_exact_facts = max(0, max_exact_facts)
        max_state_hints = max(0, max_state_hints)
        max_source_refs = max(0, max_source_refs)

        # Re-score facts
        scored_facts = self._rescore_facts(
            packet.exact_facts, prefer_current, trn_state, working_memory,
        )

        # Detect additional conflicts from TRN state / working memory
        extra_hints = self._cross_check_hints(
            scored_facts, trn_state, working_memory,
        )
        all_hints = list(packet.state_hints) + extra_hints

        # Suppress duplicates in hints
        all_hints = self._deduplicate_hints(all_hints)

        # Sort facts by priority, trim
        scored_facts.sort(key=lambda f: -f.priority_score)
        trimmed_facts = scored_facts[:max_exact_facts]

        # Trim hints (keep conflicts and unresolved first)
        all_hints.sort(key=lambda h: (
            0 if h.hint_type in ("conflict", "unresolved") else 1,
            -h.confidence,
        ))
        trimmed_hints = all_hints[:max_state_hints]

        # Trim source refs
        trimmed_refs = packet.source_refs[:max_source_refs]

        # Rebuild summary
        summary = self._build_mediated_summary(
            trimmed_facts, trimmed_hints, packet.packet_summary,
        )

        return CompactMemoryPacket(
            exact_facts=trimmed_facts,
            state_hints=trimmed_hints,
            anomaly_flags=list(dict.fromkeys(
                str(x) for x in packet.anomaly_flags
                if not isinstance(x, (dict, list, set))
            )),
            provenance_summary=list(packet.provenance_summary),
            source_refs=trimmed_refs,
            packet_summary=summary,
        )

    def _rescore_facts(
        self,
        facts: list[MemoryFact],
        prefer_current: bool,
        trn_state: dict[str, Any] | None,
        working_memory: dict[str, Any] | None,
    ) -> list[MemoryFact]:
        """Re-score facts based on status, provenance, and context."""
        # Pre-compute focus keys once
        wm_keys: set[str] = set()
        if working_memory is not None:
            raw_focus = working_memory.get("focus_keys", [])
            if isinstance(raw_focus, (list, tuple, set, frozenset)):
                wm_keys = {item for item in raw_focus if isinstance(item, str)}

        result: list[MemoryFact] = []
        for fact in facts:
            # Start from confidence (retrieval signal), not priority_score
            # to avoid double-scoring with messenger's additive score.
            conf = fact.confidence if math.isfinite(fact.confidence) else 0.0
            new_score = conf

            # Status weighting
            status = fact.status or "unknown"
            new_score += _STATUS_WEIGHT.get(status, 0.5)
            if prefer_current and status == "current":
                new_score += 0.5

            # Provenance weighting
            provenance = fact.provenance or "unknown"
            new_score += _PROVENANCE_WEIGHT.get(provenance, 0.4)

            # Working memory context boost
            if fact.key in wm_keys:
                new_score += 0.3

            # Create new fact with updated score (immutable input)
            result.append(MemoryFact(
                key=fact.key,
                value=fact.value,
                confidence=fact.confidence,
                source_doc_id=fact.source_doc_id,
                source_span_id=fact.source_span_id,
                source_title=fact.source_title,
                timestamp=fact.timestamp,
                status=fact.status,
                provenance=fact.provenance,
                priority_score=new_score,
            ))
        return result

    def _cross_check_hints(
        self,
        facts: list[MemoryFact],
        trn_state: dict[str, Any] | None,
        working_memory: dict[str, Any] | None,
    ) -> list[StateHint]:
        """Generate additional hints from cross-checking with TRN/working memory."""
        hints: list[StateHint] = []

        if trn_state is not None:
            # If TRN detected a trend change, add hint
            raw_anomalies = trn_state.get("anomalies", [])
            if isinstance(raw_anomalies, str):
                raw_anomalies = [raw_anomalies]
            elif not isinstance(raw_anomalies, (list, tuple)):
                raw_anomalies = []
            for anomaly in raw_anomalies:
                anomaly_str = str(anomaly)[:200]
                hints.append(StateHint(
                    hint_type="anomaly",
                    text=f"TRN anomaly: {anomaly_str}",
                    confidence=0.6,
                    source_doc_id="trn_state",
                    source_span_id="",
                ))

        if working_memory is not None:
            # Cross-check: if working memory has a value that differs from top fact
            wm_values = working_memory.get("known_values", {})
            if not isinstance(wm_values, dict):
                wm_values = {}
            for fact in facts:
                if fact.key in wm_values:
                    wm_val = str(wm_values[fact.key]).strip()[:200]
                    fact_val = (fact.value or "").strip()[:200]
                    if wm_val != fact_val:
                        hints.append(StateHint(
                            hint_type="conflict",
                            text=(
                                f"Working memory has '{fact.key}={wm_val}' "
                                f"but retrieval says '{fact.value}'"
                            ),
                            confidence=0.7,
                            source_doc_id=fact.source_doc_id,
                            source_span_id=fact.source_span_id,
                        ))

        return hints

    def _deduplicate_hints(self, hints: list[StateHint]) -> list[StateHint]:
        """Remove duplicate hints by (hint_type, text) pair."""
        seen: set[tuple[str, str]] = set()
        result: list[StateHint] = []
        for hint in hints:
            hint_type_str = hint.hint_type if isinstance(hint.hint_type, str) else str(hint.hint_type)
            hint_text_str = hint.text if isinstance(hint.text, str) else str(hint.text)
            key = (hint_type_str, hint_text_str)
            if key not in seen:
                seen.add(key)
                result.append(hint)
        return result

    def _build_mediated_summary(
        self,
        facts: list[MemoryFact],
        hints: list[StateHint],
        original_summary: str,
    ) -> str:
        """Build post-mediation summary."""
        lines: list[str] = []

        if facts:
            top = facts[0]
            lines.append(
                f"Top fact: {top.key}={top.value} "
                f"(status={top.status}, score={top.priority_score:.2f})"
            )

        conflicts = [h for h in hints if h.hint_type == "conflict"]
        if conflicts:
            lines.append(f"Unresolved conflicts: {len(conflicts)}")

        unresolved = [h for h in hints if h.hint_type == "unresolved"]
        if unresolved:
            lines.append(f"Unresolved items: {len(unresolved)}")

        if not lines:
            return original_summary

        return "; ".join(lines)
