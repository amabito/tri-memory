"""Tests for CompactMemoryPacket and related dataclasses."""
from __future__ import annotations

import json

import pytest

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
def sample_fact() -> MemoryFact:
    return MemoryFact(
        key="bearing_capacity",
        value="450 kN",
        confidence=0.9,
        source_doc_id="spec_rev3",
        source_span_id="s001",
        source_title="Foundation Spec Rev.3",
        status="current",
        provenance="spec",
        priority_score=1.5,
    )


@pytest.fixture
def sample_hint() -> StateHint:
    return StateHint(
        hint_type="conflict",
        text="bearing_capacity differs: 450kN vs 380kN",
        confidence=0.8,
        source_doc_id="spec_rev3",
        source_span_id="s001",
    )


@pytest.fixture
def sample_packet(sample_fact: MemoryFact, sample_hint: StateHint) -> CompactMemoryPacket:
    return CompactMemoryPacket(
        exact_facts=[sample_fact],
        state_hints=[sample_hint],
        anomaly_flags=["version_mismatch"],
        provenance_summary=["[spec] Foundation Spec Rev.3 (status=current)"],
        source_refs=[
            SourceRef(
                doc_id="spec_rev3",
                span_id="s001",
                title="Foundation Spec Rev.3",
                status="current",
                provenance="spec",
            ),
        ],
        packet_summary="Primary: bearing_capacity=450 kN (current)",
    )


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_from_dict_round_trip(self, sample_packet: CompactMemoryPacket) -> None:
        d = sample_packet.to_dict()
        restored = CompactMemoryPacket.from_dict(d)
        assert restored.exact_facts[0].key == "bearing_capacity"
        assert restored.exact_facts[0].value == "450 kN"
        assert restored.exact_facts[0].status == "current"
        assert restored.state_hints[0].hint_type == "conflict"
        assert restored.source_refs[0].doc_id == "spec_rev3"
        assert restored.packet_summary == sample_packet.packet_summary

    def test_to_json_from_json_round_trip(self, sample_packet: CompactMemoryPacket) -> None:
        json_str = sample_packet.to_json()
        restored = CompactMemoryPacket.from_json(json_str)
        assert restored.fact_count == 1
        assert restored.hint_count == 1
        assert restored.exact_facts[0].priority_score == 1.5

    def test_json_is_valid_json(self, sample_packet: CompactMemoryPacket) -> None:
        json_str = sample_packet.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "exact_facts" in parsed

    def test_empty_packet_round_trip(self) -> None:
        empty = CompactMemoryPacket()
        d = empty.to_dict()
        restored = CompactMemoryPacket.from_dict(d)
        assert restored.fact_count == 0
        assert restored.hint_count == 0
        assert restored.packet_summary == ""

    def test_from_dict_missing_fields(self) -> None:
        """Missing fields default to empty."""
        restored = CompactMemoryPacket.from_dict({})
        assert restored.fact_count == 0
        assert restored.packet_summary == ""


# ---------------------------------------------------------------------------
# Properties and methods
# ---------------------------------------------------------------------------

class TestProperties:
    def test_fact_count(self, sample_packet: CompactMemoryPacket) -> None:
        assert sample_packet.fact_count == 1

    def test_hint_count(self, sample_packet: CompactMemoryPacket) -> None:
        assert sample_packet.hint_count == 1

    def test_has_conflicts_true(self, sample_packet: CompactMemoryPacket) -> None:
        assert sample_packet.has_conflicts() is True

    def test_has_conflicts_false(self) -> None:
        packet = CompactMemoryPacket(
            state_hints=[
                StateHint(
                    hint_type="trend",
                    text="values increasing",
                    confidence=0.6,
                    source_doc_id="d1",
                    source_span_id="s1",
                ),
            ],
        )
        assert packet.has_conflicts() is False

    def test_has_conflicts_empty(self) -> None:
        assert CompactMemoryPacket().has_conflicts() is False


# ---------------------------------------------------------------------------
# MemoryFact defaults
# ---------------------------------------------------------------------------

class TestMemoryFact:
    def test_optional_fields_default_none(self) -> None:
        fact = MemoryFact(
            key="k", value="v", confidence=0.5,
            source_doc_id="d", source_span_id="s", source_title="t",
        )
        assert fact.timestamp is None
        assert fact.status is None
        assert fact.provenance is None
        assert fact.priority_score == 0.0
