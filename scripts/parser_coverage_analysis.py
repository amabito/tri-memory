"""Phase 2.8: Parser coverage analysis -- before vs after enhanced extraction."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

from trimemory.disentangled_archive import MetadataParser

# Import the enhanced extraction helpers from eval script
# We re-define them here to avoid __file__ issues with exec()

_MD_BOLD_KV_PATTERN = re.compile(
    r"\*\*([^*]{1,60})\*\*\s*[:=]\s*(.{1,500}?)(?:\n|$)",
    re.MULTILINE,
)
_TABLE_ROW_PATTERN = re.compile(
    r"^\s*\|\s*(.{1,80}?)\s*\|\s*(.{1,200}?)\s*\|",
    re.MULTILINE,
)
_NUMBERED_BOLD_KV = re.compile(
    r"^\s*\d+\.\s*\*\*([^*]{1,60})\*\*\s*[:：]\s*(.{1,300}?)$",
    re.MULTILINE,
)
_BULLET_BOLD_KV = re.compile(
    r"^\s*[-*]\s*\*\*([^*]{1,60})\*\*\s*[:：]\s*(.{1,300}?)$",
    re.MULTILINE,
)
_MD_CHANGE_PATTERN = re.compile(
    r"\*\*(?:改定|変更)(?:前|後)\*\*\s*[:：]\s*(.{1,300}?)$",
    re.MULTILINE,
)
_INLINE_BOLD_VALUE = re.compile(
    r"(\S{2,20})\s*\*\*(\d+(?:\.\d+)?[^*]{0,10}?)\*\*"
)
_DEPRECATED_PATTERN = re.compile(
    r"(?:非推奨|deprecated|廃止予定|obsolete)\s*(?:[(（]([^)）]+)[)）])?",
    re.IGNORECASE,
)

_CANONICAL_SLOTS: dict[str, dict] = {
    "sla_target_percentage": {
        "aliases_re": re.compile(
            r"(?:SLA|稼働率|availability|uptime|サービスレベル).*?(\d+(?:\.\d+)?)\s*%",
            re.IGNORECASE,
        ),
        "context_re": re.compile(r"(?:稼働率保証|目標稼働率|SLA\s*target|service\s*level)", re.IGNORECASE),
    },
    "password_minimum_length": {
        "aliases_re": re.compile(r"(?:最低文字数|パスワード.*?文字|minimum.*?length).*?(\d+)", re.IGNORECASE),
        "context_re": re.compile(r"(?:パスワード|password|文字数)", re.IGNORECASE),
    },
    "monthly_price_per_user": {
        "aliases_re": re.compile(r"(?:月額|ユーザー単価|unit\s*price).*?[¥$]\s*(\d[\d,]*)", re.IGNORECASE),
        "context_re": re.compile(r"(?:月額|単価|ユーザー|price|料金)", re.IGNORECASE),
    },
    "payment_terms_days": {
        "aliases_re": re.compile(r"(?:NET\s*(\d+)|支払期限.*?(\d+)\s*日|payment.*?(\d+)\s*days)", re.IGNORECASE),
        "context_re": re.compile(r"(?:支払|payment|NET|請求)", re.IGNORECASE),
    },
    "annual_leave_days": {
        "aliases_re": re.compile(r"(?:年次有給|有給休暇|annual\s*leave).*?(\d+)\s*日", re.IGNORECASE),
        "context_re": re.compile(r"(?:有給|休暇|leave)", re.IGNORECASE),
    },
    "vulnerability_count": {
        "aliases_re": re.compile(r"(?:未是正|Critical|脆弱性).*?(\d+)\s*件", re.IGNORECASE),
        "context_re": re.compile(r"(?:脆弱性|vulnerability|Critical|未是正)", re.IGNORECASE),
    },
    "connection_limit": {
        "aliases_re": re.compile(r"(?:同時接続|接続上限|concurrent|connection\s*limit).*?(\d+)", re.IGNORECASE),
        "context_re": re.compile(r"(?:接続|connection|concurrent)", re.IGNORECASE),
    },
    "admin_user_limit": {
        "aliases_re": re.compile(r"(?:管理者|admin).*?(?:上限|limit|最大).*?(\d+)", re.IGNORECASE),
        "context_re": re.compile(r"(?:管理者|admin|ユーザー数|上限|フリープラン)", re.IGNORECASE),
    },
}


def extract_enhanced_facts(text: str) -> list[str]:
    facts: list[str] = []
    seen_keys: set[str] = set()

    for m in _MD_BOLD_KV_PATTERN.finditer(text):
        key, val = m.group(1).strip(), m.group(2).strip()
        if key and val and key.lower() not in seen_keys:
            seen_keys.add(key.lower())
            facts.append(f"{key}: {val}")

    for m in _NUMBERED_BOLD_KV.finditer(text):
        key, val = m.group(1).strip(), m.group(2).strip()
        if key and val and key.lower() not in seen_keys:
            seen_keys.add(key.lower())
            facts.append(f"{key}: {val}")

    for m in _BULLET_BOLD_KV.finditer(text):
        key, val = m.group(1).strip(), m.group(2).strip()
        if key and val and key.lower() not in seen_keys:
            seen_keys.add(key.lower())
            facts.append(f"{key}: {val}")

    header_keywords = {"項目", "header", "---", "field", "column"}
    for m in _TABLE_ROW_PATTERN.finditer(text):
        key = m.group(1).strip().strip("*").strip()
        val = m.group(2).strip().strip("*").strip()
        if (key and val and key.lower() not in header_keywords
                and val.lower() not in header_keywords
                and "---" not in val and key.lower() not in seen_keys):
            seen_keys.add(key.lower())
            facts.append(f"{key}: {val}")

    for m in _MD_CHANGE_PATTERN.finditer(text):
        val = m.group(1).strip()
        if val:
            facts.append(f"[change] {val}")

    # Inline bold values: "年間 **25日** の..."
    for m in _INLINE_BOLD_VALUE.finditer(text):
        context_word, bold_val = m.group(1).strip(), m.group(2).strip()
        if bold_val and context_word.lower() not in seen_keys:
            facts.append(f"{context_word}: {bold_val}")

    # Deprecated/obsolete markers
    for m in _DEPRECATED_PATTERN.finditer(text):
        detail = m.group(1)
        if detail:
            facts.append(f"[deprecated] {detail.strip()}")
        else:
            facts.append("[deprecated]")

    return facts[:30]


def extract_canonical_slots(text: str, doc_status: str = "unknown") -> list[dict]:
    results = []
    lines = text.split("\n")
    for slot_name, slot_def in _CANONICAL_SLOTS.items():
        for i, line in enumerate(lines):
            m = slot_def["aliases_re"].search(line)
            if not m:
                continue
            value = None
            for g in m.groups():
                if g is not None:
                    value = g
                    break
            if value is None:
                continue
            context_window = "\n".join(lines[max(0, i - 2):min(len(lines), i + 3)])
            has_context = bool(slot_def["context_re"].search(context_window))
            results.append({
                "slot": slot_name, "value": value.replace(",", ""),
                "status": doc_status, "confidence": "high" if has_context else "medium",
            })
    return results


def merge_facts(base: list[str], enhanced: list[str]) -> list[str]:
    base_keys: set[str] = set()
    for f in base:
        if ":" in f:
            base_keys.add(f.split(":")[0].strip().lower())
    merged = list(base)
    for ef in enhanced:
        if ":" in ef:
            key = ef.split(":")[0].strip().lower()
            if key not in base_keys:
                merged.append(ef)
                base_keys.add(key)
        elif ef.startswith("[change]") or ef.startswith("[deprecated]"):
            merged.append(ef)
    return merged[:30]


# Gold fact expectations: sample_id -> {value_substring, doc_id}
GOLD_FACTS = {
    "PRACT-001": {"value": "12", "doc": "doc_02", "slot": "password_minimum_length"},
    "PRACT-002": {"value": "12", "doc": "doc_04", "slot": "monthly_price_per_user"},
    "PRACT-003": {"value": "25", "doc": "doc_02", "slot": "annual_leave_days"},
    "PRACT-004": {"value": "300", "doc": "doc_01", "slot": "approval_threshold"},
    "PRACT-005": {"value": "99.9", "doc": "doc_01", "slot": "sla_target_percentage"},
    "PRACT-006": {"value": "3", "doc": "doc_02", "slot": "admin_user_limit"},
    "PRACT-007": {"value": "deprecated", "doc": "doc_02", "slot": "api_status"},
    "PRACT-008": {"value": "480", "doc": "doc_02", "slot": "connection_limit"},
    "PRACT-016": {"value": "2", "doc": "doc_02", "slot": "vulnerability_count"},
    "PRACT-017": {"value": "45", "doc": "doc_02", "slot": "payment_terms_days"},
}


def main():
    parser = MetadataParser()
    data_path = Path("data/docbench/final/pract_v1.jsonl")

    print("=== Phase 2.8: Parser Coverage Analysis (BEFORE vs AFTER) ===")
    print()
    print(f"{'Sample':<12} {'Gold':<10} {'Base#':<7} {'Enh#':<7} "
          f"{'Base':<8} {'Enh':<8} {'Slot':<8} {'New Facts'}")
    print("-" * 90)

    total = {"base": 0, "enh": 0, "slot": 0}

    with open(data_path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s.get("split") != "dev":
                continue
            sid = s["sample_id"]
            if sid not in GOLD_FACTS:
                continue
            gold = GOLD_FACTS[sid]

            for doc in s["documents"]:
                if doc["doc_id"] != gold["doc"]:
                    continue

                content = doc["content"]
                doc_status = doc.get("metadata", {}).get("status", "unknown")

                # BEFORE
                meta = parser.parse(content, doc_id=doc["doc_id"], title=doc.get("title", ""))
                base_facts = meta.exact_fact_candidates
                base_ents = [f"{e['value']}{e['unit']}" for e in meta.entity_value_pairs]
                base_all = base_facts + base_ents
                base_hit = any(gold["value"] in str(f) for f in base_all)

                # AFTER
                enhanced = extract_enhanced_facts(content)
                merged = merge_facts(base_facts, enhanced)
                merged_all = merged + base_ents
                enh_hit = any(gold["value"] in str(f) for f in merged_all)

                # Canonical slots
                slots = extract_canonical_slots(content, doc_status)
                slot_hit = any(gold["value"] in str(sv.get("value", "")) for sv in slots)

                if base_hit:
                    total["base"] += 1
                if enh_hit:
                    total["enh"] += 1
                if slot_hit:
                    total["slot"] += 1

                # Show new facts added
                new_facts = [f for f in enhanced if f not in base_facts][:3]
                new_str = " | ".join(nf[:40] for nf in new_facts) if new_facts else "-"

                b_tag = "HIT" if base_hit else "MISS"
                e_tag = "HIT" if enh_hit else "MISS"
                s_tag = "HIT" if slot_hit else "MISS"

                print(f"{sid:<12} {gold['value']:<10} {len(base_facts):<7} "
                      f"{len(merged):<7} {b_tag:<8} {e_tag:<8} {s_tag:<8} {new_str}")

    print("-" * 90)
    print(f"{'TOTAL':<12} {'':10} {'':7} {'':7} "
          f"{total['base']}/10   {total['enh']}/10   {total['slot']}/10")
    print()

    # Detailed: show what enhanced extraction found for each sample
    print("=== New Facts Detail (Enhanced - Base) ===")
    print()
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s.get("split") != "dev":
                continue
            sid = s["sample_id"]
            if sid not in GOLD_FACTS:
                continue
            gold = GOLD_FACTS[sid]

            for doc in s["documents"]:
                if doc["doc_id"] != gold["doc"]:
                    continue

                content = doc["content"]
                doc_status = doc.get("metadata", {}).get("status", "unknown")

                meta = parser.parse(content, doc_id=doc["doc_id"], title=doc.get("title", ""))
                enhanced = extract_enhanced_facts(content)
                new_only = [f for f in enhanced if f not in meta.exact_fact_candidates]

                slots = extract_canonical_slots(content, doc_status)

                if new_only or slots:
                    print(f"{sid} ({doc['doc_id']}):")
                    for nf in new_only[:8]:
                        marker = " <<GOLD>>" if gold["value"] in nf else ""
                        print(f"  + fact: {nf[:80]}{marker}")
                    for sv in slots[:4]:
                        marker = " <<GOLD>>" if gold["value"] in str(sv.get("value", "")) else ""
                        print(f"  + slot: {sv['slot']}={sv['value']} [{sv['status']}]{marker}")
                    print()


if __name__ == "__main__":
    main()
