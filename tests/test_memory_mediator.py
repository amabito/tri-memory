"""Tests for MemoryMediator -- conflict resolution and re-scoring."""
from __future__ import annotations

import pytest

from trimemory.memory_mediator import MemoryMediator
from trimemory.memory_packet import (
    CompactMemoryPacket,
    MemoryFact,
    SourceRef,
    StateHint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mediator() -> MemoryMediator:
    return MemoryMediator()


def _make_fact(
    key: str,
    value: str,
    status: str = "unknown",
    provenance: str = "unknown",
    priority_score: float = 0.5,
    source_title: str = "",
) -> MemoryFact:
    return MemoryFact(
        key=key,
        value=value,
        confidence=0.8,
        source_doc_id=f"doc_{key}",
        source_span_id=f"span_{key}",
        source_title=source_title or f"Title for {key}",
        status=status,
        provenance=provenance,
        priority_score=priority_score,
    )


def _make_packet(
    facts: list[MemoryFact],
    hints: list[StateHint] | None = None,
) -> CompactMemoryPacket:
    return CompactMemoryPacket(
        exact_facts=facts,
        state_hints=hints or [],
        anomaly_flags=[],
        provenance_summary=[],
        source_refs=[],
        packet_summary="",
    )


# ---------------------------------------------------------------------------
# Test 1: status-based re-scoring
# ---------------------------------------------------------------------------

class TestStatusScoring:
    """Current facts should rank above superseded after mediation."""

    def test_current_outranks_superseded(self, mediator: MemoryMediator) -> None:
        packet = _make_packet([
            _make_fact("capacity", "380 kN", status="superseded", provenance="spec", priority_score=0.8),
            _make_fact("capacity", "450 kN", status="current", provenance="spec", priority_score=0.5),
        ])
        result = mediator.resolve(packet, prefer_current=True)

        assert result.fact_count == 2
        top = result.exact_facts[0]
        assert "450" in top.value, f"Current 450kN should rank first, got {top.value}"

    def test_prefer_current_false(self, mediator: MemoryMediator) -> None:
        """Without prefer_current, status still helps but less dramatically."""
        packet = _make_packet([
            _make_fact("capacity", "380 kN", status="superseded", provenance="spec", priority_score=2.0),
            _make_fact("capacity", "450 kN", status="current", provenance="spec", priority_score=0.1),
        ])
        result = mediator.resolve(packet, prefer_current=False)
        # Without the 0.5 current boost, the higher base score may win
        assert result.fact_count == 2


# ---------------------------------------------------------------------------
# Test 2: provenance weighting
# ---------------------------------------------------------------------------

class TestProvenanceWeighting:
    """Spec/calc should outrank faq/note at equal status."""

    def test_spec_outranks_faq(self, mediator: MemoryMediator) -> None:
        packet = _make_packet([
            _make_fact("depth", "5.0 m", status="current", provenance="faq", priority_score=0.5),
            _make_fact("depth", "5.0 m", status="current", provenance="spec", priority_score=0.5),
        ])
        result = mediator.resolve(packet)

        # Both have same base but spec provenance should score higher
        top = result.exact_facts[0]
        assert top.provenance == "spec", f"Spec should rank first, got {top.provenance}"


# ---------------------------------------------------------------------------
# Test 3: working memory cross-check
# ---------------------------------------------------------------------------

class TestWorkingMemoryCrossCheck:
    """Working memory conflict should generate a conflict hint."""

    def test_wm_conflict_generates_hint(self, mediator: MemoryMediator) -> None:
        packet = _make_packet([
            _make_fact("capacity", "450 kN", status="current"),
        ])
        working_memory = {
            "known_values": {"capacity": "380 kN"},
        }
        result = mediator.resolve(packet, working_memory=working_memory)

        conflict_hints = [h for h in result.state_hints if h.hint_type == "conflict"]
        assert len(conflict_hints) > 0, "Should detect conflict with working memory"

    def test_wm_agreement_no_conflict(self, mediator: MemoryMediator) -> None:
        packet = _make_packet([
            _make_fact("capacity", "450 kN", status="current"),
        ])
        working_memory = {
            "known_values": {"capacity": "450 kN"},
        }
        result = mediator.resolve(packet, working_memory=working_memory)

        conflict_hints = [h for h in result.state_hints if h.hint_type == "conflict"]
        assert len(conflict_hints) == 0, "No conflict when values agree"


# ---------------------------------------------------------------------------
# Test 4: TRN state anomaly integration
# ---------------------------------------------------------------------------

class TestTRNStateIntegration:
    """TRN anomalies should propagate as hints."""

    def test_trn_anomaly_becomes_hint(self, mediator: MemoryMediator) -> None:
        packet = _make_packet([
            _make_fact("flow_rate", "150 mm/h", status="current"),
        ])
        trn_state = {
            "anomalies": ["rapid_value_change_detected"],
        }
        result = mediator.resolve(packet, trn_state=trn_state)

        anomaly_hints = [h for h in result.state_hints if h.hint_type == "anomaly"]
        assert len(anomaly_hints) > 0, "TRN anomaly should become a hint"


# ---------------------------------------------------------------------------
# Test 5: quota trimming
# ---------------------------------------------------------------------------

class TestQuotaTrimming:
    """Output should respect max limits."""

    def test_facts_trimmed(self, mediator: MemoryMediator) -> None:
        facts = [_make_fact(f"k{i}", f"v{i}") for i in range(20)]
        packet = _make_packet(facts)
        result = mediator.resolve(packet, max_exact_facts=3)
        assert result.fact_count <= 3

    def test_hints_trimmed(self, mediator: MemoryMediator) -> None:
        hints = [
            StateHint(
                hint_type="conflict",
                text=f"conflict_{i}",
                confidence=0.5,
                source_doc_id=f"d{i}",
                source_span_id=f"s{i}",
            )
            for i in range(10)
        ]
        packet = CompactMemoryPacket(state_hints=hints)
        result = mediator.resolve(packet, max_state_hints=2)
        assert result.hint_count <= 2


# ---------------------------------------------------------------------------
# Test 6: immutability -- input packet unchanged
# ---------------------------------------------------------------------------

class TestImmutability:
    """Resolve must not modify the input packet."""

    def test_input_packet_unchanged(self, mediator: MemoryMediator) -> None:
        original_fact = _make_fact("k", "v", priority_score=0.5)
        packet = _make_packet([original_fact])
        original_score = original_fact.priority_score

        _ = mediator.resolve(packet)

        assert packet.exact_facts[0].priority_score == original_score


# ---------------------------------------------------------------------------
# Test 7: hint deduplication
# ---------------------------------------------------------------------------

class TestHintDeduplication:
    def test_duplicate_hints_removed(self, mediator: MemoryMediator) -> None:
        hint = StateHint(
            hint_type="conflict",
            text="same conflict",
            confidence=0.8,
            source_doc_id="d1",
            source_span_id="s1",
        )
        packet = CompactMemoryPacket(state_hints=[hint, hint])
        result = mediator.resolve(packet)
        assert result.hint_count == 1


# ---------------------------------------------------------------------------
# Test 8: empty packet
# ---------------------------------------------------------------------------

class TestEmptyPacket:
    def test_empty_packet_resolves(self, mediator: MemoryMediator) -> None:
        packet = CompactMemoryPacket()
        result = mediator.resolve(packet)
        assert result.fact_count == 0
        assert result.hint_count == 0

    def test_empty_with_trn_state(self, mediator: MemoryMediator) -> None:
        packet = CompactMemoryPacket()
        result = mediator.resolve(
            packet,
            trn_state={"anomalies": ["test_anomaly"]},
        )
        assert result.hint_count > 0


# ---------------------------------------------------------------------------
# Test 9: conflict preservation
# ---------------------------------------------------------------------------

class TestConflictPreservation:
    """Conflicts and unresolved items should not be silently dropped."""

    def test_conflicts_preserved_in_trim(self, mediator: MemoryMediator) -> None:
        hints = [
            StateHint(hint_type="conflict", text="critical conflict", confidence=0.9, source_doc_id="d1", source_span_id="s1"),
            StateHint(hint_type="trend", text="mild trend", confidence=0.3, source_doc_id="d2", source_span_id="s2"),
            StateHint(hint_type="unresolved", text="unresolved item", confidence=0.5, source_doc_id="d3", source_span_id="s3"),
        ]
        packet = CompactMemoryPacket(state_hints=hints)
        result = mediator.resolve(packet, max_state_hints=2)

        # Conflict and unresolved should be kept over trend
        kept_types = {h.hint_type for h in result.state_hints}
        assert "conflict" in kept_types, "Conflict should survive trimming"
        assert "unresolved" in kept_types, "Unresolved should survive trimming"


# ---------------------------------------------------------------------------
# Test 10: working memory focus_keys boost
# ---------------------------------------------------------------------------

class TestFocusKeysBoost:
    def test_focus_key_boosts_priority(self, mediator: MemoryMediator) -> None:
        facts = [
            _make_fact("capacity", "450 kN", priority_score=0.5),
            _make_fact("depth", "5.0 m", priority_score=0.5),
        ]
        packet = _make_packet(facts)
        working_memory = {"focus_keys": ["depth"]}

        result = mediator.resolve(packet, working_memory=working_memory)

        # depth should get a boost
        depth_fact = next((f for f in result.exact_facts if f.key == "depth"), None)
        cap_fact = next((f for f in result.exact_facts if f.key == "capacity"), None)
        assert depth_fact is not None
        assert cap_fact is not None
        assert depth_fact.priority_score > cap_fact.priority_score
