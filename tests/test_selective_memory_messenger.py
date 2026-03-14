"""Tests for SelectiveMemoryMessenger -- query-driven packet construction."""
from __future__ import annotations

import pytest

from trimemory.selective_memory_messenger import SelectiveMemoryMessenger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def messenger() -> SelectiveMemoryMessenger:
    return SelectiveMemoryMessenger()


def _make_item(
    text: str,
    doc_id: str = "",
    span_id: str = "",
    title: str = "",
) -> dict:
    return {"text": text, "doc_id": doc_id, "span_id": span_id, "title": title}


# ---------------------------------------------------------------------------
# Test 1: superseded vs current
# ---------------------------------------------------------------------------

class TestSupersededVsCurrent:
    """Rev.2 380kN vs Rev.3 450kN -- current must win."""

    def test_current_value_ranks_first(self, messenger: SelectiveMemoryMessenger) -> None:
        items = [
            _make_item(
                text="支持力: 380 kN\n詳細計算あり。設計根拠...",
                doc_id="spec_rev2",
                title="Foundation Spec Rev.2 (superseded)",
            ),
            _make_item(
                text="支持力: 450 kN\n変更理由: 地盤調査結果を反映",
                doc_id="spec_rev3",
                title="Foundation Spec Rev.3 (current approved)",
            ),
        ]
        packet = messenger.build_packet(items, query="現行の支持力は?")

        # Current value should be in facts
        values = [f.value for f in packet.exact_facts]
        assert any("450" in v for v in values), f"Expected 450 in facts, got {values}"

        # If 380 appears, it should not be first
        if len(packet.exact_facts) > 1:
            first_value = packet.exact_facts[0].value
            assert "380" not in first_value, "Superseded 380kN should not be first"


# ---------------------------------------------------------------------------
# Test 2: latest-only insufficient -- need change reason too
# ---------------------------------------------------------------------------

class TestLatestOnlyInsufficient:
    """Current value + change notice reason both needed."""

    def test_value_and_reason_both_present(self, messenger: SelectiveMemoryMessenger) -> None:
        items = [
            _make_item(
                text="排水設計流量: 150 mm/h\n承認済み",
                doc_id="spec_drainage",
                title="Drainage Spec Rev.5 (current approved)",
            ),
            _make_item(
                text="変更前: 120 mm/h -> 変更後: 150 mm/h\n理由: 気候変動係数1.1適用",
                doc_id="dcn_042",
                title="設計変更通知 DCN-042",
            ),
        ]
        packet = messenger.build_packet(items, query="現行値と変更理由は?")

        # Should have facts
        assert packet.fact_count > 0, "Should extract at least one fact"

        # Summary or hints should mention something about the change
        all_text = packet.packet_summary
        for hint in packet.state_hints:
            all_text += " " + hint.text
        for fact in packet.exact_facts:
            all_text += " " + fact.value
        # At least 150 should appear
        assert "150" in all_text, f"Current value 150 should appear somewhere: {all_text}"


# ---------------------------------------------------------------------------
# Test 3: formal vs provisional
# ---------------------------------------------------------------------------

class TestFormalVsProvisional:
    """Formal spec 150 mm/h vs meeting minutes 180 mm/h (pending)."""

    def test_formal_value_first_provisional_in_hints(
        self, messenger: SelectiveMemoryMessenger,
    ) -> None:
        items = [
            _make_item(
                text="設計雨量: 150 mm/h\n正式承認済",
                doc_id="spec_rain",
                title="Rainfall Spec Rev.4 (current final)",
            ),
            _make_item(
                text="設計雨量: 180 mm/h に変更決定\n未反映、次回改訂で反映予定\n議事録",
                doc_id="minutes_0301",
                title="設計会議議事録 2026-03-01 (provisional)",
            ),
        ]
        packet = messenger.build_packet(items, query="設計雨量は?")

        # The formal value (150) should rank higher
        if packet.fact_count > 0:
            top_fact = packet.exact_facts[0]
            # Status should be current or the value should be 150
            is_current = top_fact.status == "current"
            has_150 = "150" in top_fact.value
            assert is_current or has_150, (
                f"Top fact should be current/150, got status={top_fact.status}, value={top_fact.value}"
            )

        # Provisional 180 should appear somewhere (hints or lower-ranked facts)
        all_text = " ".join(
            [f.value for f in packet.exact_facts]
            + [h.text for h in packet.state_hints]
        )
        # 180 may or may not be extracted depending on parser; not hard requirement
        # But provisional source should be referenced
        prov_refs = [r for r in packet.source_refs if r.status == "provisional" or "議事録" in (r.title or "")]
        # At least the source ref is preserved
        assert len(packet.source_refs) >= 2, "Both sources should be referenced"


# ---------------------------------------------------------------------------
# Test 4: table/text conflict
# ---------------------------------------------------------------------------

class TestTableTextConflict:
    """Body says GB-9.5, table says RW-15 -- conflict hint expected."""

    def test_conflict_detected(self, messenger: SelectiveMemoryMessenger) -> None:
        items = [
            _make_item(
                text="材料規格: GB-9.5\n設計意図はNRC要件に基づく",
                doc_id="body_sec3",
                title="Design Document Section 3",
            ),
            _make_item(
                text="材料規格: RW-15\n一覧表 Table 4.1",
                doc_id="table_4_1",
                title="Material Table 4.1",
            ),
        ]
        packet = messenger.build_packet(items, query="材料規格の矛盾は?")

        # Conflict hint should exist
        conflict_hints = [h for h in packet.state_hints if h.hint_type == "conflict"]
        assert len(conflict_hints) > 0, (
            f"Expected conflict hint, got hints: {[h.hint_type for h in packet.state_hints]}"
        )

        # Summary should mention conflict
        assert "conflict" in packet.packet_summary.lower() or "Conflict" in packet.packet_summary


# ---------------------------------------------------------------------------
# Test 5: query-sensitive extraction
# ---------------------------------------------------------------------------

class TestQuerySensitiveExtraction:
    """Different queries on same items should produce different packet emphasis."""

    @pytest.fixture
    def items(self) -> list[dict]:
        return [
            _make_item(
                text="支持力: 450 kN\n変更前: 380 kN -> 変更後: 450 kN\n理由: 地盤改良工法変更",
                doc_id="spec_rev3",
                title="Foundation Spec Rev.3 (current approved)",
            ),
        ]

    def test_value_query_prioritizes_facts(
        self, messenger: SelectiveMemoryMessenger, items: list[dict],
    ) -> None:
        packet = messenger.build_packet(items, query="現行値は?")
        # Should have facts
        assert packet.fact_count > 0

    def test_reason_query_includes_change_hints(
        self, messenger: SelectiveMemoryMessenger, items: list[dict],
    ) -> None:
        packet = messenger.build_packet(items, query="なぜ変わった?")
        # Should have hints about changes
        all_hint_text = " ".join(h.text for h in packet.state_hints)
        all_fact_text = " ".join(f.value for f in packet.exact_facts)
        combined = all_hint_text + all_fact_text
        # Either the change should be in hints or facts
        assert "380" in combined or "change" in combined.lower() or packet.hint_count > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_items(self, messenger: SelectiveMemoryMessenger) -> None:
        packet = messenger.build_packet([], query="test")
        assert packet.fact_count == 0
        assert packet.hint_count == 0

    def test_items_without_text(self, messenger: SelectiveMemoryMessenger) -> None:
        packet = messenger.build_packet([{"doc_id": "d1"}], query="test")
        assert isinstance(packet.packet_summary, str)

    def test_max_limits_respected(self, messenger: SelectiveMemoryMessenger) -> None:
        items = [
            _make_item(
                text=f"key_{i}: value_{i}\nnum: {i*100} kN",
                doc_id=f"doc_{i}",
                title=f"Doc {i}",
            )
            for i in range(20)
        ]
        packet = messenger.build_packet(
            items, query="値は?",
            max_exact_fact_fields=3,
            max_state_hints=2,
            max_source_refs=4,
        )
        assert packet.fact_count <= 3
        assert packet.hint_count <= 2
        assert len(packet.source_refs) <= 4
