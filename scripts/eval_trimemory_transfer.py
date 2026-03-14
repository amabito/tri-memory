"""TriMemory Phase 3.0 Transfer Benchmark -- IT Security Policy Compliance QA.

Adapts the TriMemory DocBench pipeline (eval_docbench_ollama.py) to a new domain:
IT Information Security Policy Compliance QA (PolicyBench).

Conditions:
    A) plain      -- all policy docs concatenated, no structure
    B) latest     -- only the highest-priority doc (status + authority rank)
    C) rag        -- top-k chunks by lexical overlap (character trigrams)
    D) trimemory  -- full TriMemory pipeline with policy-domain canonical slots

Renderer policies (frozen):
    3B model  -> short_refs_semantic_en_ja
    7B model  -> short_refs_en_ja_no_relabel

Usage:
    python scripts/eval_trimemory_transfer.py --model llama3.2:3b --condition all
    python scripts/eval_trimemory_transfer.py --model qwen2.5:7b --condition trimemory
    python scripts/eval_trimemory_transfer.py \\
        --model llama3.2:3b qwen2.5:7b --condition all --verbose
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import requests
except ImportError:
    import urllib.request  # type: ignore[no-redef]
    requests = None  # type: ignore[assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

OLLAMA_MAX_RETRIES = 2
OLLAMA_TIMEOUT_SEC = 120
OLLAMA_RETRY_SLEEP_SEC = 2

# ---------------------------------------------------------------------------
# TriMemory imports (fail-safe -- same pattern as eval_docbench_ollama.py)
# ---------------------------------------------------------------------------

_TRIMEMORY_AVAILABLE = False
try:
    from trimemory.disentangled_archive import MetadataParser
    from trimemory.selective_memory_messenger import SelectiveMemoryMessenger
    from trimemory.memory_mediator import MemoryMediator
    _TRIMEMORY_AVAILABLE = True
    logger.info("TriMemory imports OK -- packet path available")
except ImportError as exc:
    logger.debug("TriMemory import detail: %s", exc)
    logger.warning("TriMemory imports failed -- fallback only")

# ---------------------------------------------------------------------------
# Import shared helpers from eval_docbench_ollama.py
# ---------------------------------------------------------------------------
# We import specific functions that are domain-agnostic rather than copy-pasting.
# If the import fails (e.g., file not on sys.path), we re-implement the minimum.

_PARENT_SCRIPT = Path(__file__).parent / "eval_docbench_ollama.py"
_HELPERS_IMPORTED = False

try:
    # Dynamic import of sibling script.
    # Must register in sys.modules before exec_module so that any internal
    # relative imports in the script can resolve correctly.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "eval_docbench_ollama", str(_PARENT_SCRIPT)
    )
    _docbench_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
    sys.modules["eval_docbench_ollama"] = _docbench_mod
    _spec.loader.exec_module(_docbench_mod)  # type: ignore[union-attr]

    # Import shared domain-agnostic helpers
    run_ollama = _docbench_mod.run_ollama
    parse_response = _docbench_mod.parse_response
    ParsedResponse = _docbench_mod.ParsedResponse
    _fuzzy_value_match = _docbench_mod._fuzzy_value_match
    _simple_chunk = _docbench_mod._simple_chunk
    _lexical_score = _docbench_mod._lexical_score
    render_packet_semantic = _docbench_mod.render_packet_semantic
    _format_short_source_refs = _docbench_mod._format_short_source_refs
    _compile_memory_packet = _docbench_mod._compile_memory_packet
    _extract_enhanced_facts = _docbench_mod._extract_enhanced_facts
    _merge_enhanced_facts = _docbench_mod._merge_enhanced_facts
    _normalize_fact_value = _docbench_mod._normalize_fact_value
    _HELPERS_IMPORTED = True
    logger.info("Imported shared helpers from eval_docbench_ollama.py")
except Exception as exc:
    logger.debug("Import detail: %s", exc)
    logger.warning("Could not import from eval_docbench_ollama.py -- using local fallbacks")

# ---------------------------------------------------------------------------
# Local fallbacks when parent import fails
# ---------------------------------------------------------------------------

if not _HELPERS_IMPORTED:
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    def run_ollama(  # type: ignore[no-redef]
        prompt: str,
        model: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Call Ollama API and return the response text."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        url = f"{OLLAMA_URL}/api/generate"
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(OLLAMA_MAX_RETRIES + 1):
            try:
                if requests is not None:
                    resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SEC)
                    resp.raise_for_status()
                    return resp.json().get("response", "")
                else:
                    req = urllib.request.Request(
                        url, data=body,
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SEC) as r:
                        data = json.loads(r.read().decode("utf-8"))
                        return data.get("response", "")
            except Exception as e:
                if attempt < OLLAMA_MAX_RETRIES:
                    logger.debug("Ollama retry %d detail: %s", attempt + 1, e)
                    logger.warning("Ollama retry %d", attempt + 1)
                    time.sleep(OLLAMA_RETRY_SLEEP_SEC)
                else:
                    logger.debug("Ollama failure detail: %s", e)
                    raise RuntimeError(
                        f"Ollama request failed after {OLLAMA_MAX_RETRIES + 1} attempts"
                    ) from e
        raise RuntimeError("Ollama request loop exited without result")

    @dataclass
    class ParsedResponse:  # type: ignore[no-redef]
        answer: str = ""
        current_value: str = ""
        previous_value: str = ""
        reason: str = ""
        evidence_docs: list[str] = field(default_factory=list)
        evidence_spans: list[str] = field(default_factory=list)
        status_judgment: str = ""
        needs_escalation: bool = False
        confidence: float = 0.0
        raw_text: str = ""
        parse_success: bool = False

    def parse_response(raw: str) -> ParsedResponse:  # type: ignore[no-redef]
        result = ParsedResponse(raw_text=raw)
        text = raw.strip()
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)
        try:
            data = json.loads(text)
            result.answer = str(data.get("answer", ""))
            result.current_value = str(data.get("current_value", ""))
            result.previous_value = str(data.get("previous_value", ""))
            result.reason = str(data.get("reason", ""))
            result.evidence_docs = [str(x) for x in data.get("evidence_docs", [])]
            result.evidence_spans = [str(x) for x in data.get("evidence_spans", [])]
            result.status_judgment = str(data.get("status_judgment", ""))
            result.needs_escalation = bool(data.get("needs_escalation", False))
            try:
                _conf = float(data.get("confidence", 0.0))
                result.confidence = _conf if math.isfinite(_conf) else 0.0
            except (TypeError, ValueError):
                result.confidence = 0.0
            result.parse_success = True
        except json.JSONDecodeError:
            result.answer = raw[:500]
            doc_refs = re.findall(r"doc_\d{2,}", raw)
            result.evidence_docs = list(dict.fromkeys(doc_refs))
            if any(kw in raw for kw in ["undetermined", "uncertain"]):
                result.status_judgment = "undetermined"
            elif any(kw in raw for kw in ["contradictory", "conflict"]):
                result.status_judgment = "contradictory"
        return result

    def _fuzzy_value_match(gold: str, response: str) -> float:  # type: ignore[no-redef]
        if not gold:
            return 1.0
        g = gold.lower().strip()
        r = response.lower().strip()
        if not r:
            return 0.0
        if g == r:
            return 1.0
        if g in r or r in g:
            return 0.7
        gold_nums = re.findall(r"[\d.]+", g)
        resp_nums = re.findall(r"[\d.]+", r)
        if gold_nums and resp_nums and set(gold_nums) & set(resp_nums):
            return 0.5
        g_tok = set(re.findall(r"\w{2,}", g))
        r_tok = set(re.findall(r"\w{2,}", r))
        if g_tok and r_tok and len(g_tok & r_tok) / len(g_tok) >= 0.5:
            return 0.3
        return 0.0

    def _simple_chunk(doc: dict, max_chunk_chars: int = 300) -> list[dict]:  # type: ignore[no-redef]
        content = doc.get("content", "")
        doc_id = doc.get("doc_id", "")
        title = doc.get("title", "")
        meta = doc.get("metadata", {})
        role = doc.get("role", "")
        paragraphs = re.split(r"\n{2,}", content)
        chunks = []
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
            if len(para) > max_chunk_chars:
                for j in range(0, len(para), max_chunk_chars):
                    chunks.append({
                        "text": para[j:j + max_chunk_chars],
                        "doc_id": doc_id, "span_id": f"para_{i}_chunk_{j // max_chunk_chars}",
                        "title": title, "metadata": meta, "role": role,
                    })
            else:
                chunks.append({
                    "text": para, "doc_id": doc_id, "span_id": f"para_{i}",
                    "title": title, "metadata": meta, "role": role,
                })
        return chunks

    def _lexical_score(query: str, text: str) -> float:  # type: ignore[no-redef]
        def trigrams(s: str) -> set[str]:
            s = s.lower()
            return {s[i:i+3] for i in range(max(0, len(s) - 2))}
        q_tri = trigrams(query)
        t_tri = trigrams(text)
        if not q_tri or not t_tri:
            return 0.0
        return len(q_tri & t_tri) / len(q_tri | t_tri)

    def _extract_enhanced_facts(text: str) -> list[str]:  # type: ignore[no-redef]
        facts: list[str] = []
        seen_keys: set[str] = set()
        bold_kv = re.compile(r"\*\*([^*]{1,60})\*\*\s*[:=]\s*(.{1,500}?)(?:\n|$)", re.MULTILINE)
        for m in bold_kv.finditer(text):
            k, v = m.group(1).strip(), m.group(2).strip()
            if k and v and k.lower() not in seen_keys:
                seen_keys.add(k.lower())
                facts.append(f"{k}: {v}")
        return facts[:20]

    def _normalize_fact_value(value: str) -> str:  # type: ignore[no-redef]
        v = value.strip().replace("**", "")
        v = re.sub(r"[¥]\s*", "¥", v)
        return v

    def _merge_enhanced_facts(base: list[str], enhanced: list[str]) -> list[str]:  # type: ignore[no-redef]
        base_keys: set[str] = set()
        for f in base:
            if ":" in f:
                base_keys.add(f.split(":")[0].strip().lower())
        merged = list(base)
        for ef in enhanced:
            if ":" in ef:
                k = ef.split(":")[0].strip().lower()
                if k not in base_keys:
                    merged.append(_normalize_fact_value(ef))
                    base_keys.add(k)
        return merged[:30]


# ---------------------------------------------------------------------------
# Domain constants -- IT Security Policy
# ---------------------------------------------------------------------------

# Canonical slot definitions for PolicyBench domain
POLICY_CANONICAL_SLOTS: dict[str, re.Pattern[str]] = {
    "password_minimum_length": re.compile(
        r"(?:パスワード|password).*?(?:最小|最低|minimum).*?(\d+)\s*(?:文字|chars?|characters?)",
        re.IGNORECASE,
    ),
    "encryption_standard": re.compile(
        r"(?:暗号化|encryption).*?(AES-\d+|RSA-\d+|ChaCha20)",
        re.IGNORECASE,
    ),
    "session_timeout": re.compile(
        r"(?:セッション|session).*?(?:タイムアウト|timeout).*?(\d+)\s*(?:分|min)",
        re.IGNORECASE,
    ),
    "retention_period": re.compile(
        r"(?:保存|保管|retention).*?(\d+)\s*(?:年|years?)",
        re.IGNORECASE,
    ),
    "backup_frequency": re.compile(
        r"(?:バックアップ|backup).*?(?:頻度|frequency|周期).*?"
        r"(日次|週次|月次|daily|weekly|monthly)",
        re.IGNORECASE,
    ),
    "assessment_threshold": re.compile(
        r"(?:評価|assessment|審査).*?(\d+)\s*(?:万円|万|円)",
        re.IGNORECASE,
    ),
    "incident_sla": re.compile(
        r"(?:インシデント|incident).*?(?:SLA|対応時間|response).*?"
        r"(\d+)\s*(?:時間|分|hour|min)",
        re.IGNORECASE,
    ),
    "review_frequency": re.compile(
        r"(?:レビュー|review|棚卸).*?(?:頻度|周期|frequency).*?"
        r"(四半期|半期|月次|年次|quarterly|monthly|semi-annual|annual)",
        re.IGNORECASE,
    ),
}


# ---------------------------------------------------------------------------
# Phase 3.0.1 -- Thin Schema V2: broadened canonical slot patterns
# ---------------------------------------------------------------------------
# V2 patterns relax per-line keyword co-occurrence requirements to handle
# policy documents where the keyword (e.g., "バックアップ") is in a heading
# and the value (e.g., "日次") is in a table data row or a separate clause.

POLICY_CANONICAL_SLOTS_V2: dict[str, re.Pattern[str]] = {
    # Broadened: match "最小文字数: N文字" without requiring "パスワード" on same line
    "password_minimum_length": re.compile(
        r"(?:パスワード|password|最小文字数|文字数要件)"
        r".*?(\d+)\s*(?:文字|chars?|characters?)",
        re.IGNORECASE,
    ),
    # Broadened: match "AES-N" or "RSA-N" near encryption-related context
    "encryption_standard": re.compile(
        r"(?:暗号|encryption|対称暗号|cipher|アルゴリズム)"
        r".*?(AES-\d+|RSA-\d+|ChaCha20|ECDSA-P-\d+)",
        re.IGNORECASE,
    ),
    # Broadened: match "セッション.*(\d+)分" without requiring "タイムアウト"
    "session_timeout": re.compile(
        r"(?:セッション|session|タイムアウト|timeout|無操作).*?(\d+)\s*(?:分|min)",
        re.IGNORECASE,
    ),
    # Broadened: match "N年間" in data context without requiring 保管/保存 on same line
    "retention_period": re.compile(
        r"(?:保存|保管|retention|個人データ|データ.*カテゴリ|保管期間)"
        r".*?(\d+)\s*(?:年間|年|years?)",
        re.IGNORECASE,
    ),
    # Broadened: match "日次/週次/月次" near "業務データ" or "バックアップ" context
    "backup_frequency": re.compile(
        r"(?:バックアップ|backup|業務データ|業務システム|分類.*頻度)"
        r".*?(日次|週次|月次|daily|weekly|monthly)",
        re.IGNORECASE,
    ),
    # Broadened: match "N万円" near "契約金額/閾値/アセスメント" context
    "assessment_threshold": re.compile(
        r"(?:評価|assessment|審査|契約金額|閾値|threshold|アセスメント)"
        r".*?(\d+)\s*(?:万円|万|円)",
        re.IGNORECASE,
    ),
    # Broadened: match "P1.*N時間" pattern from SLA tables
    "incident_sla": re.compile(
        r"(?:インシデント|incident|P1|優先度.*最高|初動)"
        r".*?(\d+)\s*(?:時間|分|hour|min)(?:以内)?",
        re.IGNORECASE,
    ),
    # Broadened: match "四半期ごと" without requiring "頻度/周期"
    "review_frequency": re.compile(
        r"(?:レビュー|review|棚卸|権限.*定期|アクセス権)"
        r".*?(四半期|半期|月次|年次|毎月|quarterly|monthly|semi-annual|annual)",
        re.IGNORECASE,
    ),
}


def _extract_policy_slots_v2(
    text: str,
    doc_status: str = "unknown",
) -> list[dict[str, str]]:
    """V2 slot extraction with broadened patterns for PolicyBench.

    Uses POLICY_CANONICAL_SLOTS_V2 (relaxed keyword co-occurrence).
    Otherwise identical to _extract_policy_slots.
    """
    results: list[dict[str, str]] = []
    lines = text.split("\n")
    seen_slot_values: set[str] = set()

    for slot_name, pattern in POLICY_CANONICAL_SLOTS_V2.items():
        for i, line in enumerate(lines):
            m = pattern.search(line)
            if not m:
                continue
            value: str | None = None
            for g in m.groups():
                if g is not None:
                    value = g
                    break
            if value is None:
                continue

            dedup_key = f"{slot_name}:{value}"
            if dedup_key in seen_slot_values:
                continue
            seen_slot_values.add(dedup_key)

            context_window = "\n".join(
                lines[max(0, i - 2):min(len(lines), i + 3)]
            )
            slot_keywords = _SLOT_CONTEXT_KEYWORDS.get(slot_name, [])
            has_keyword = any(kw.lower() in context_window.lower() for kw in slot_keywords)

            results.append({
                "slot": slot_name,
                "value": value.strip(),
                "status": doc_status,
                "context": line.strip()[:100],
                "confidence": "high" if has_keyword else "medium",
            })

    return results


def _filter_canonical_slots_by_status(
    slots: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Status-aware canonical slot filtering (Phase 3.0.1 thin schema).

    For each slot_name, keep only slots from the highest-priority status.
    If approved + temporary co-exist, keep both (amendment/temporary modifies base).
    Never inject draft/superseded slots when approved/temporary slots exist.
    """
    if not slots:
        return []

    _status_rank = {
        "current": 4, "approved": 3, "temporary": 2,
        "draft": 1, "superseded": 0, "archived": -1,
    }

    # Group by slot_name
    by_slot: dict[str, list[dict[str, str]]] = {}
    for s in slots:
        by_slot.setdefault(s["slot"], []).append(s)

    filtered: list[dict[str, str]] = []
    for slot_name, group in by_slot.items():
        # Find max status rank in this group
        max_rank = max(
            _status_rank.get(s["status"], -1) for s in group
        )
        # Keep approved (rank 3+) and temporary (rank 2) if either exists
        # Otherwise fall through to whatever is available
        if max_rank >= 2:
            # Keep approved + temporary, drop draft + superseded
            for s in group:
                rank = _status_rank.get(s["status"], -1)
                if rank >= 2:
                    filtered.append(s)
        else:
            # No authoritative source -- keep all (rare edge case)
            filtered.extend(group)

    return filtered


# Status priority for policy documents (higher = more authoritative/current)
STATUS_PRIORITY: dict[str, int] = {
    "approved": 3,
    "temporary": 2,
    "draft": 1,
    "superseded": 0,
    "archived": -1,
}

# Authority rank for policy documents (higher = more authoritative)
AUTHORITY_RANK: dict[str, int] = {
    "CISO": 4,
    "IT Director": 3,
    "Department Head": 2,
    "Working Group": 1,
}

# Status -> TriMemory trust score
_POLICY_STATUS_TRUST: dict[str, float] = {
    "approved": 0.9,
    "temporary": 0.6,
    "draft": 0.35,
    "superseded": 0.2,
    "archived": 0.1,
    "unknown": 0.4,
}

# Frozen render mode assignments per model size
_RENDER_MODE_BY_MODEL: dict[str, str] = {
    "3b": "short_refs_semantic_en_ja",
    "7b": "short_refs_en_ja_no_relabel",
}
_RENDER_MODE_DEFAULT = "short_refs_semantic_en_ja"

# Factor decomposition for render modes (moved from build_prompt_trimemory)
_RENDER_FACTOR_MAP: dict[str, dict[str, Any]] = {
    "short_refs_semantic_en_ja":   {"alias_mode": "en_ja", "relabel": True,  "answer": False, "refs": True},
    "short_refs_en_ja_no_relabel": {"alias_mode": "en_ja", "relabel": False, "answer": False, "refs": True},
    "packet_only":                 {"alias_mode": "en_ja", "relabel": True,  "answer": False, "refs": False},
    "packet_plus_short_refs":      {"alias_mode": "en_ja", "relabel": True,  "answer": False, "refs": True},
    "semantic_en_ja":              {"alias_mode": "en_ja", "relabel": True,  "answer": False, "refs": False},
}


def _get_render_mode(model: str) -> str:
    """Determine the frozen render mode for a given model name."""
    model_lower = model.lower()
    if "7b" in model_lower:
        return _RENDER_MODE_BY_MODEL["7b"]
    if "3b" not in model_lower:
        logger.warning(
            "Unrecognized model size in '%s' -- defaulting to 3B render mode", model,
        )
    return _RENDER_MODE_BY_MODEL["3b"]


# ---------------------------------------------------------------------------
# System prompts (policy domain, bilingual)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_POLICY_PLAIN = (
    "あなたはIT情報セキュリティポリシーの専門家です。"
    "提供されたポリシー文書のみに基づいて質問に回答してください。"
    "矛盾や廃止されたポリシーが含まれる場合は明示してください。"
    '確実に判断できない場合は "undetermined" と回答してください。\n\n'
    "You are an IT information security policy expert. "
    "Answer questions based ONLY on the provided policy documents. "
    "If documents contain contradictions or superseded policies, say so explicitly. "
    'If you cannot determine the answer with certainty, say "undetermined".\n\n'
    "You MUST respond in this exact JSON format:\n"
    "{\n"
    '  "answer": "your answer text (in Japanese)",\n'
    '  "current_value": "the current/latest value if applicable",\n'
    '  "previous_value": "the previous value if applicable, or empty",\n'
    '  "reason": "why the value changed or why this is the answer",\n'
    '  "evidence_docs": ["policy_doc_01", "policy_doc_03"],\n'
    '  "evidence_spans": ["policy_doc_01:sec_3_2"],\n'
    '  "status_judgment": "confirmed|undetermined|contradictory",\n'
    '  "authority": "which authority/role issued the current policy",\n'
    '  "needs_escalation": false,\n'
    '  "confidence": 0.8\n'
    "}\n"
    "Return ONLY valid JSON. No other text."
)

SYSTEM_PROMPT_POLICY_TRIMEMORY = (
    "あなたはIT情報セキュリティポリシーの専門家です。"
    "以下のMEMORY SUMMARYを使ってポリシーの質問に回答してください。\n\n"
    "MEMORY SUMMARY使用ルール:\n"
    "1. 'Current formal facts' を真の現行ポリシーとして使用してください。\n"
    "2. 'Older or superseded facts' は過去のポリシー比較にのみ使用してください。\n"
    "3. 'Pending or provisional updates' は提案・検討中の変更に関する質問の場合のみ使用してください。\n"
    "4. 承認済み (approved) の現行ポリシーが見つからない場合は 'uncertain' と回答し、"
    "needs_escalation=true に設定してください。\n"
    "5. draftやtemporaryのポリシーをapprovedとして回答しないでください。\n\n"
    "You are an IT information security policy expert. "
    "Use the MEMORY SUMMARY below to answer policy compliance questions.\n\n"
    "Rules for using the memory summary:\n"
    '1. Use "Current formal facts" as the default source of truth.\n'
    '2. Use "Older or superseded facts" only for historical comparison.\n'
    '3. Use "Pending or provisional updates" only if the question asks about proposals.\n'
    "4. If no approved current policy is available, "
    'answer "uncertain" and set needs_escalation=true.\n'
    "5. Do not promote draft or temporary policies to current facts.\n\n"
    "You MUST respond in this exact JSON format:\n"
    "{\n"
    '  "answer": "your answer text (in Japanese)",\n'
    '  "current_value": "the current/latest value from Current formal facts",\n'
    '  "previous_value": "the previous value from Older facts, or empty",\n'
    '  "reason": "why the value changed or why this is the answer",\n'
    '  "evidence_docs": ["policy_doc_01", "policy_doc_03"],\n'
    '  "evidence_spans": ["policy_doc_01:sec_3_2"],\n'
    '  "status_judgment": "confirmed|undetermined|contradictory",\n'
    '  "authority": "which authority/role issued the current policy",\n'
    '  "needs_escalation": false,\n'
    '  "confidence": 0.8\n'
    "}\n"
    "Return ONLY valid JSON. No other text."
)

# ---------------------------------------------------------------------------
# Data model -- PolicyBench sample
# ---------------------------------------------------------------------------

@dataclass
class PolicySample:
    """One PolicyBench evaluation sample."""
    sample_id: str
    query: str
    gold_answer: str
    gold_status: str           # "approved" | "confirmed" | "undetermined" | "contradictory"
    gold_authority: str        # issuing authority of the correct policy
    evaluation_type: str       # "single_doc" | "multi_doc" | "conflict" | "superseded"
    difficulty: str            # "easy" | "medium" | "hard"
    failure_class: str         # expected failure class (e.g. "STALE_FACT", "DRAFT_CONFUSION")
    documents: list[dict[str, Any]]
    split: str = "dev"


def load_policy_samples(
    path: Path,
    split_filter: str | None = "dev",
) -> list[PolicySample]:
    """Load PolicyBench samples from JSONL file.

    Each line has:
      sample_id, query, gold_answer, gold_status, gold_authority,
      evaluation_type, difficulty, failure_class, documents[]

    Each document has:
      doc_id, title, status, effective_date, authority, supersedes, content
    """
    samples: list[PolicySample] = []
    if not path.exists():
        logger.error("PolicyBench data file not found: %s", path)
        return samples

    with open(path, encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.debug("Line %d parse detail: %s", line_no, exc)
                logger.warning("Line %d: JSON parse error", line_no)
                continue

            s = PolicySample(
                sample_id=d.get("sample_id", f"sample_{line_no:04d}"),
                query=d.get("query", d.get("question", "")),
                gold_answer=d.get("gold_answer", ""),
                gold_status=d.get("gold_status", "confirmed"),
                gold_authority=d.get("gold_authority", ""),
                evaluation_type=d.get("evaluation_type", d.get("type", "")),
                difficulty=d.get("difficulty", "medium"),
                failure_class=d.get("failure_class", ""),
                documents=d.get("documents", []),
                split=d.get("split", "dev"),
            )
            if split_filter is None or s.split == split_filter:
                samples.append(s)

    logger.info("Loaded %d samples from %s (split=%s)", len(samples), path, split_filter)
    return samples


# ---------------------------------------------------------------------------
# Policy document adapter -- normalize to TriMemory format
# ---------------------------------------------------------------------------

def _policy_doc_to_trimemory_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    """Map PolicyBench document fields to TriMemory-compatible metadata.

    PolicyBench doc fields:
        doc_id, title, status, effective_date, authority, supersedes, content

    TriMemory metadata fields (from MetadataParser):
        status, provenance, source_trust, date, version, entity_value_pairs,
        exact_fact_candidates, ...

    Status injection:
        approved   -> currentness="current", trust=0.9
        temporary  -> currentness="draft",   trust=0.6
        draft      -> currentness="draft",   trust=0.35
        superseded -> currentness="superseded", trust=0.2
        archived   -> currentness="superseded", trust=0.1
    """
    status = doc.get("status", "unknown").lower()
    authority = doc.get("authority", "")
    effective_date = doc.get("effective_date", "")
    supersedes = doc.get("supersedes", "")

    # Map policy status to TriMemory currentness indicator
    if status == "approved":
        currentness = "current"
    elif status in ("temporary", "draft"):
        currentness = "draft"
    elif status in ("superseded", "archived"):
        currentness = "superseded"
    else:
        currentness = "unknown"

    trust = _POLICY_STATUS_TRUST.get(status, 0.4)
    provenance = "spec"

    meta: dict[str, Any] = {
        "status": currentness,
        "provenance": provenance,
        "source_trust": trust,
        "date": effective_date,
        "authority": authority,
        "policy_status_raw": status,
        "supersedes": supersedes,
    }
    if authority:
        meta["entity_value_pairs"] = [{"key": "authority", "value": authority}]

    return meta


def _doc_priority_score(doc: dict[str, Any]) -> int:
    """Compute a scalar priority score for a policy document.

    Used to select the 'best' single document in the 'latest' condition.
    Higher is better (more authoritative and current).
    """
    status = doc.get("status", "unknown").lower()
    authority = doc.get("authority", "")
    status_score = STATUS_PRIORITY.get(status, 0) * 100
    authority_score = AUTHORITY_RANK.get(authority, 0)
    return status_score + authority_score


# ---------------------------------------------------------------------------
# Policy-domain canonical slot extractor
# ---------------------------------------------------------------------------

# Context keywords per slot for confidence boosting
_SLOT_CONTEXT_KEYWORDS: dict[str, list[str]] = {
    "password_minimum_length": ["パスワードポリシー", "password policy", "認証", "authentication"],
    "encryption_standard": ["暗号化ポリシー", "encryption policy", "データ保護", "data protection"],
    "session_timeout": ["セッション管理", "session management", "アクセス制御", "access control"],
    "retention_period": ["データ保持", "data retention", "保管期間", "記録管理"],
    "backup_frequency": ["バックアップポリシー", "backup policy", "災害復旧", "disaster recovery"],
    "assessment_threshold": ["リスク評価", "risk assessment", "調達", "procurement"],
    "incident_sla": ["インシデント管理", "incident management", "対応手順", "response procedure"],
    "review_frequency": ["ポリシーレビュー", "policy review", "コンプライアンス", "compliance"],
}


def _extract_policy_slots(
    text: str,
    doc_status: str = "unknown",
) -> list[dict[str, str]]:
    """Extract policy-domain canonical slot values from text.

    Returns list of dicts with: slot, value, status, context, confidence.
    """
    results: list[dict[str, str]] = []
    lines = text.split("\n")
    seen_slot_values: set[str] = set()

    for slot_name, pattern in POLICY_CANONICAL_SLOTS.items():
        for i, line in enumerate(lines):
            m = pattern.search(line)
            if not m:
                continue
            # First non-None capture group
            value: str | None = None
            for g in m.groups():
                if g is not None:
                    value = g
                    break
            if value is None:
                continue

            dedup_key = f"{slot_name}:{value}"
            if dedup_key in seen_slot_values:
                continue
            seen_slot_values.add(dedup_key)

            # Context window for confidence scoring
            context_window = "\n".join(
                lines[max(0, i - 2):min(len(lines), i + 3)]
            )
            # Confidence: keyword reinforcement in context
            slot_keywords = _SLOT_CONTEXT_KEYWORDS.get(slot_name, [])
            has_keyword = any(kw.lower() in context_window.lower() for kw in slot_keywords)

            results.append({
                "slot": slot_name,
                "value": value.strip(),
                "status": doc_status,
                "context": line.strip()[:100],
                "confidence": "high" if has_keyword else "medium",
            })

    return results


# ---------------------------------------------------------------------------
# Prompt builders -- 4 conditions adapted for policy domain
# ---------------------------------------------------------------------------

def _format_policy_doc(doc: dict[str, Any]) -> str:
    """Format a policy document for prompt inclusion."""
    doc_id = doc.get("doc_id", "?")
    title = doc.get("title", "Untitled")
    status = doc.get("status", "?")
    effective_date = doc.get("effective_date", "?")
    authority = doc.get("authority", "?")
    supersedes = doc.get("supersedes", "")
    content = doc.get("content", "")

    header = (
        f"[{doc_id}] {title} "
        f"(status={status}, date={effective_date}, authority={authority}"
    )
    if supersedes:
        header += f", supersedes={supersedes}"
    header += ")"
    return f"{header}\n{content}"


def build_prompt_plain(sample: PolicySample) -> str:
    """Condition A: All policy documents concatenated."""
    docs_text = "\n\n---\n\n".join(
        _format_policy_doc(doc) for doc in sample.documents
    )
    return f"""Policy Documents:

{docs_text}

Question: {sample.query}

Respond in the JSON format specified."""


def build_prompt_latest(sample: PolicySample) -> str:
    """Condition B: Only the highest-priority policy document.

    Selects the document with the best (status priority + authority rank) score.
    """
    if not sample.documents:
        return f"No documents available.\n\nQuestion: {sample.query}\n\nRespond in the JSON format specified."

    best_doc = max(sample.documents, key=_doc_priority_score)
    return f"""Policy Document (highest authority/status):

{_format_policy_doc(best_doc)}

Question: {sample.query}

Respond in the JSON format specified."""


def build_prompt_rag(sample: PolicySample, top_k: int = 4) -> str:
    """Condition C: Top-k chunks by lexical overlap with the query."""
    all_chunks: list[dict[str, Any]] = []
    for doc in sample.documents:
        # Adapt policybench doc to the format _simple_chunk expects
        adapted_doc = {
            "doc_id": doc.get("doc_id", ""),
            "title": doc.get("title", ""),
            "content": doc.get("content", ""),
            "metadata": {
                "status": doc.get("status", "unknown"),
                "date": doc.get("effective_date", ""),
                "authority": doc.get("authority", ""),
            },
            "role": doc.get("status", ""),  # use status as role proxy
        }
        all_chunks.extend(_simple_chunk(adapted_doc))

    scored = [
        (c, _lexical_score(sample.query, c["text"]))
        for c in all_chunks
    ]
    scored.sort(key=lambda x: -x[1])
    top_chunks = scored[:top_k]

    chunks_text = "\n\n---\n\n".join(
        f"[{c['doc_id']}:{c['span_id']}] ({c.get('title', '')})\n{c['text']}"
        for c, _ in top_chunks
    )

    return f"""Retrieved policy document chunks (ranked by relevance):

{chunks_text}

Question: {sample.query}

Respond in the JSON format specified."""


# ---------------------------------------------------------------------------
# TriMemory chunk builder -- policy domain adapter
# ---------------------------------------------------------------------------

def _build_policy_trimemory_chunks(
    sample: PolicySample,
    use_thin_schema: bool = False,
) -> list[dict[str, Any]]:
    """Build TriMemory-compatible chunk dicts from PolicyBench documents.

    Uses MetadataParser for text-based fact extraction, then overrides
    status/trust/provenance with policy document metadata.
    Also runs policy-domain canonical slot extraction.

    When use_thin_schema=True, uses broadened V2 patterns.
    """
    if not _TRIMEMORY_AVAILABLE:
        return []

    parser = MetadataParser()
    all_items: list[dict[str, Any]] = []

    for doc in sample.documents:
        doc_id = doc.get("doc_id", "")
        title = doc.get("title", "")
        status_raw = doc.get("status", "unknown").lower()
        content = doc.get("content", "")

        # Map policy status to TriMemory currentness
        policy_meta = _policy_doc_to_trimemory_metadata(doc)
        trimemory_status = policy_meta["status"]
        trust = policy_meta["source_trust"]
        provenance = policy_meta["provenance"]

        # Adapt to _simple_chunk's expected format
        adapted_doc = {
            "doc_id": doc_id,
            "title": title,
            "content": content,
            "metadata": {"status": trimemory_status, "date": doc.get("effective_date", "")},
            "role": status_raw,
        }
        chunks = _simple_chunk(adapted_doc)

        for chunk in chunks:
            # MetadataParser: extract facts from text
            parsed_meta = parser.parse(
                text=chunk["text"],
                doc_id=doc_id,
                span_id=chunk["span_id"],
                title=title,
            )

            chunk_meta = parsed_meta.to_dict()

            # Override with authoritative policy metadata
            chunk_meta["status"] = trimemory_status
            chunk_meta["provenance"] = provenance
            chunk_meta["source_trust"] = trust
            chunk_meta["authority"] = doc.get("authority", "")
            chunk_meta["policy_status_raw"] = status_raw

            # Enhanced fact extraction (bold KV, tables)
            enhanced = _extract_enhanced_facts(chunk["text"])
            chunk_meta["exact_fact_candidates"] = _merge_enhanced_facts(
                chunk_meta.get("exact_fact_candidates", []),
                enhanced,
            )

            # Policy-domain canonical slot extraction
            _slot_extractor = _extract_policy_slots_v2 if use_thin_schema else _extract_policy_slots
            slot_values = _slot_extractor(chunk["text"], trimemory_status)
            if slot_values:
                chunk_meta.setdefault("canonical_slots", []).extend(slot_values)

            all_items.append({
                "text": chunk["text"],
                "doc_id": doc_id,
                "span_id": chunk["span_id"],
                "title": title,
                "metadata": chunk_meta,
            })

    return all_items


def build_prompt_trimemory(
    sample: PolicySample,
    top_k: int = 4,
    render_mode: str = "short_refs_semantic_en_ja",
    packet_log: list[dict[str, Any]] | None = None,
    use_thin_schema: bool = False,
) -> str:
    """Condition D/E: TriMemory compact packet via real pipeline.

    Pipeline:
      1. MetadataParser extracts facts from policy document text
      2. Policy-domain canonical slots injected
      3. SelectiveMemoryMessenger builds packet
      4. MemoryMediator resolves conflicts
      5. Memory Compiler v6 (_compile_memory_packet) filters and ranks
      6. Render with frozen render_mode

    When use_thin_schema=True (Phase 3.0.1 trimemory_schema condition):
      - Uses broadened V2 slot patterns for better coverage
      - Applies status-aware slot filtering (suppress draft/superseded slots)

    Falls back to a simple summary if TriMemory is unavailable.
    """
    if not _TRIMEMORY_AVAILABLE:
        logger.warning(
            "[%s] TriMemory not available, using fallback", sample.sample_id,
        )
        return _build_prompt_trimemory_fallback(sample)

    messenger = SelectiveMemoryMessenger()
    mediator = MemoryMediator()

    all_items = _build_policy_trimemory_chunks(sample, use_thin_schema=use_thin_schema)
    if not all_items:
        return _build_prompt_trimemory_fallback(sample)

    n_facts = sum(
        len(item["metadata"].get("exact_fact_candidates", []))
        for item in all_items
    )
    n_slots = sum(
        len(item["metadata"].get("canonical_slots", []))
        for item in all_items
    )
    logger.info(
        "  [PACKET] %d chunks, %d text facts, %d policy slots",
        len(all_items), n_facts, n_slots,
    )

    # Build packet
    packet = messenger.build_packet(
        all_items,
        query=sample.query,
        max_exact_fact_fields=12,
        max_state_hints=6,
        max_source_refs=8,
    )

    # Mediate (prefer current/approved)
    packet = mediator.resolve(packet, prefer_current=True)

    logger.info(
        "  [PACKET] After mediation: %d facts, %d hints, %d anomalies, "
        "has_conflicts=%s",
        packet.fact_count, packet.hint_count,
        len(packet.anomaly_flags), packet.has_conflicts(),
    )

    # Collect canonical slots for Memory Compiler
    all_canonical_slots: list[dict[str, str]] = []
    for item in all_items:
        all_canonical_slots.extend(item["metadata"].get("canonical_slots", []))

    # Phase 3.0.1: status-aware slot filtering for thin schema mode
    if use_thin_schema:
        pre_filter_count = len(all_canonical_slots)
        all_canonical_slots = _filter_canonical_slots_by_status(all_canonical_slots)
        logger.info(
            "  [SCHEMA] V2 slot filter: %d -> %d (dropped %d draft/superseded)",
            pre_filter_count, len(all_canonical_slots),
            pre_filter_count - len(all_canonical_slots),
        )

    # Memory Compiler v6 (frozen -- reuse from parent script)
    compiler_diag = _compile_memory_packet(
        packet,
        query=sample.query,
        canonical_slots=all_canonical_slots,
        max_facts=12,
        max_per_family=2,
    )

    logger.info(
        "  [COMPILER] %d -> %d facts (deduped=%d, mismatch_pen=%d, unit=%s)",
        compiler_diag["original_facts"],
        compiler_diag["final_facts"],
        compiler_diag["n_deduped_rows"],
        compiler_diag["n_mismatch_penalties"],
        compiler_diag["query_unit_hint"],
    )

    if packet_log is not None:
        log_entry: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "schema_version": "v2_thin" if use_thin_schema else "v1_core",
            "n_input_chunks": len(all_items),
            "n_text_facts": n_facts,
            "n_policy_slots": n_slots,
            "n_canonical_slots_injected": len(all_canonical_slots),
            "canonical_slots": all_canonical_slots,
            "render_mode": render_mode,
            "packet": packet.to_dict(),
            "compiler_diagnostics": compiler_diag,
        }
        packet_log.append(log_entry)

    # Adapt documents to the format _format_short_source_refs expects
    adapted_docs = [
        {
            "doc_id": doc.get("doc_id", ""),
            "title": doc.get("title", ""),
            "content": doc.get("content", ""),
            "metadata": {
                "status": doc.get("status", "unknown"),
                "date": doc.get("effective_date", ""),
                "authority": doc.get("authority", ""),
            },
            "role": doc.get("status", ""),
        }
        for doc in sample.documents
    ]

    # Render -- using the factor-decomposition render modes from parent script
    factors = _RENDER_FACTOR_MAP.get(render_mode, _RENDER_FACTOR_MAP["short_refs_semantic_en_ja"])

    semantic_packet = render_packet_semantic(
        packet,
        sample.query,
        include_answer_slot=factors["answer"],
        alias_mode=factors["alias_mode"],
        relabel_enabled=factors["relabel"],
    )

    parts = [semantic_packet]

    if factors["refs"]:
        short_refs = _format_short_source_refs(adapted_docs, packet)
        parts.append(f"\nShort source excerpts:\n{short_refs}")

    parts.append(f"\nQuestion: {sample.query}")
    parts.append(
        "\nUse the MEMORY SUMMARY above to answer. "
        "Respond in the JSON format specified."
    )

    return "\n".join(parts)


def _build_prompt_trimemory_fallback(sample: PolicySample) -> str:
    """Fallback when TriMemory imports are unavailable."""
    current_docs: list[str] = []
    other_docs: list[str] = []
    for doc in sample.documents:
        status = doc.get("status", "unknown")
        doc_id = doc.get("doc_id", "?")
        title = doc.get("title", "")
        authority = doc.get("authority", "")
        preview = doc.get("content", "")[:200]
        entry = (
            f"[{doc_id}] {title} (status={status}, authority={authority}): {preview}"
        )
        if status == "approved":
            current_docs.append(entry)
        else:
            other_docs.append(entry)

    sections = ["## Memory Packet (fallback mode)"]
    if current_docs:
        sections.append("### Approved Policies")
        sections.extend(current_docs)
    if other_docs:
        sections.append("### Other Policies")
        sections.extend(other_docs)

    return f"""{chr(10).join(sections)}

Question: {sample.query}

Respond in the JSON format specified."""


# ---------------------------------------------------------------------------
# Scoring -- policy domain extensions
# ---------------------------------------------------------------------------

@dataclass
class PolicyScore:
    """Per-sample scoring breakdown for PolicyBench."""
    current_value_match: float = 0.0
    previous_value_match: float = 0.0
    reason_match: float = 0.0
    status_score: float = 0.0         # gold_status vs parsed status_judgment
    authority_score: float = 0.0      # gold_authority vs parsed authority field
    evidence_score: float = 0.0       # evidence_doc_recall
    evidence_precision: float = 0.0
    uncertainty_handling: float = 0.0
    stale_fact_error: bool = False
    draft_confusion_error: bool = False   # draft policy promoted to current
    unsupported_definitive_error: bool = False
    needs_escalation_match: float = 0.0
    composite_score: float = 0.0


def _parse_authority_from_response(parsed: ParsedResponse) -> str:
    """Extract authority field from parsed response.

    The JSON schema includes an 'authority' field; fall back to scanning
    the answer text for authority keywords.
    """
    # Check raw_text for authority field in JSON
    raw = parsed.raw_text
    try:
        data = json.loads(re.search(
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL
        ).group(0))  # type: ignore[union-attr]
        authority = data.get("authority", "")
        if authority:
            return str(authority)
    except (json.JSONDecodeError, AttributeError):
        pass

    # Scan answer text for known authority names
    answer_lower = (parsed.answer + " " + parsed.reason).lower()
    for auth in sorted(AUTHORITY_RANK.keys(), key=lambda k: -AUTHORITY_RANK[k]):
        if auth.lower() in answer_lower:
            return auth
    return ""


def score_policy_sample(
    sample: PolicySample,
    parsed: ParsedResponse,
) -> PolicyScore:
    """Score a PolicyBench sample response.

    Extends the PRACT field scoring with:
    - status_score: does the answer correctly identify policy status?
    - authority_score: does the answer correctly resolve authority hierarchy?
    - draft_confusion_error: was a draft/temporary policy treated as approved?
    """
    ps = PolicyScore()

    # -- current_value_match --
    gold_curr = sample.gold_answer
    resp_curr = parsed.current_value or parsed.answer
    ps.current_value_match = _fuzzy_value_match(gold_curr, resp_curr)

    # -- previous_value_match --
    # PolicyBench does not have a dedicated previous_value field.
    # We set it to 1.0 (not penalized) unless there's a clear superseded case.
    ps.previous_value_match = 1.0

    # -- reason_match --
    # Not directly in PolicyBench gold; use partial credit from answer overlap.
    ps.reason_match = min(1.0, ps.current_value_match * 0.8 + 0.2)

    # -- status_score --
    gold_status = sample.gold_status.lower()
    resp_status = parsed.status_judgment.lower()
    _confirmed_set = {"confirmed", "current", "approved"}
    _undetermined_set = {"undetermined", "uncertain", "provisional"}
    _contradictory_set = {"contradictory", "conflict", "conflicting"}
    status_equiv: dict[str, set[str]] = {
        "confirmed": _confirmed_set,
        "approved": _confirmed_set,
        "current": _confirmed_set,
        "undetermined": _undetermined_set,
        "contradictory": _contradictory_set,
    }
    gold_set = status_equiv.get(gold_status, {gold_status})
    if resp_status in gold_set:
        ps.status_score = 1.0
    elif any(g in resp_status for g in gold_set):
        ps.status_score = 0.5
    else:
        ps.status_score = 0.0

    # -- authority_score --
    gold_auth = sample.gold_authority.strip()
    resp_auth = _parse_authority_from_response(parsed).strip()
    if not gold_auth:
        ps.authority_score = 1.0  # not applicable
    elif not resp_auth:
        ps.authority_score = 0.0
    elif gold_auth.lower() == resp_auth.lower():
        ps.authority_score = 1.0
    elif gold_auth.lower() in resp_auth.lower() or resp_auth.lower() in gold_auth.lower():
        ps.authority_score = 0.7
    else:
        # Partial: check authority hierarchy rank
        gold_rank = AUTHORITY_RANK.get(gold_auth, 0)
        resp_rank = AUTHORITY_RANK.get(resp_auth, -1)
        if resp_rank == gold_rank:
            ps.authority_score = 0.5
        else:
            ps.authority_score = 0.0

    # -- evidence_score (recall) --
    # gold_authority serves as a proxy for the authoritative document.
    # We check if the response mentions the correct authority doc.
    answer_lower = (parsed.answer + " " + parsed.reason).lower()
    if gold_auth and gold_auth.lower() in answer_lower:
        ps.evidence_score = 1.0
    elif parsed.evidence_docs:
        # Give partial credit for citing any document
        ps.evidence_score = 0.5
    else:
        ps.evidence_score = 0.0
    ps.evidence_precision = 0.5 if parsed.evidence_docs else 0.0

    # -- uncertainty_handling --
    if gold_status in ("undetermined", "contradictory"):
        if resp_status in ("undetermined", "uncertain", "contradictory", "conflict"):
            ps.uncertainty_handling = 1.0
        elif parsed.needs_escalation:
            ps.uncertainty_handling = 0.5
        else:
            ps.uncertainty_handling = 0.0
    else:
        if resp_status in ("undetermined", "uncertain"):
            ps.uncertainty_handling = 0.0
        else:
            ps.uncertainty_handling = 1.0

    # -- needs_escalation_match --
    gold_esc = (gold_status in ("undetermined", "contradictory"))
    ps.needs_escalation_match = 1.0 if (parsed.needs_escalation == gold_esc) else 0.0

    # -- stale_fact_error --
    # Scoped to failure classes where superseded/stale values are the primary
    # confusion mode. Other classes (amendment_override, authority_hierarchy,
    # exception_handling, etc.) have different failure patterns -- expanding
    # this check to all classes would create false positives.
    if sample.failure_class in (
        "current_vs_draft", "superseded_value", "version_conflict",
    ):
        if ps.current_value_match < 0.3 and ps.status_score < 0.5:
            ps.stale_fact_error = True

    # -- draft_confusion_error --
    # Scoped to classes where draft/temporary promotion is the expected
    # failure mode. Classes like conflicting_directives or amendment_override
    # fail differently (authority confusion, not draft promotion).
    if sample.failure_class in ("current_vs_draft", "status_evolution"):
        if ps.current_value_match < 0.3 and resp_status in (
            "confirmed", "current", "approved",
        ):
            ps.draft_confusion_error = True

    # -- unsupported_definitive_error --
    if gold_status in ("undetermined", "contradictory") and not parsed.needs_escalation:
        if resp_status not in ("undetermined", "contradictory", "uncertain", "conflict", ""):
            ps.unsupported_definitive_error = True

    # -- composite_score --
    # Weights (sum = 1.00):
    #   current_value 0.25, status 0.15, authority 0.10,
    #   evidence 0.10, uncertainty 0.10, error_absence_bonus 3x0.10 = 0.30
    # Error flags work as bonus subtraction: each detected error removes 0.10
    # from the maximum possible 1.00 score.
    answer_score = (
        ps.current_value_match * 0.25
        + ps.status_score * 0.15
    )
    evidence_score_component = (
        ps.evidence_score * 0.10
        + ps.authority_score * 0.10
        + ps.uncertainty_handling * 0.10
    )
    error_absence_bonus = (
        (0.0 if ps.stale_fact_error else 0.10)
        + (0.0 if ps.draft_confusion_error else 0.10)
        + (0.0 if ps.unsupported_definitive_error else 0.10)
    )
    ps.composite_score = answer_score + evidence_score_component + error_absence_bonus
    return ps


# ---------------------------------------------------------------------------
# Latency / token measurement
# ---------------------------------------------------------------------------

@dataclass
class LatencyRecord:
    """Per-inference latency and token measurements."""
    sample_id: str
    model: str
    condition: str
    render_mode: str
    prompt_chars: int
    inference_sec: float
    # Ollama provides eval_count (output tokens) and prompt_eval_count (input tokens)
    input_tokens: int = 0
    output_tokens: int = 0


def run_ollama_with_metrics(
    prompt: str,
    model: str,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> tuple[str, LatencyRecord | None]:
    """Call Ollama and capture response plus raw token/latency metrics.

    Returns (response_text, LatencyRecord_partial).
    The LatencyRecord has inference_sec, input_tokens, output_tokens.
    sample_id / model / condition fields are filled by the caller.
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if system:
        payload["system"] = system

    url = f"{os.environ.get('OLLAMA_URL', 'http://localhost:11434')}/api/generate"

    for attempt in range(OLLAMA_MAX_RETRIES + 1):
        t0 = time.time()
        try:
            if requests is not None:
                resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SEC)
                resp.raise_for_status()
                data = resp.json()
            else:
                body = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url, data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SEC) as r:
                    data = json.loads(r.read().decode("utf-8"))

            elapsed = time.time() - t0
            response_text = data.get("response", "")
            partial = LatencyRecord(
                sample_id="",
                model=model,
                condition="",
                render_mode="",
                prompt_chars=len(prompt),
                inference_sec=round(elapsed, 3),
                input_tokens=int(data.get("prompt_eval_count", 0)),
                output_tokens=int(data.get("eval_count", 0)),
            )
            return response_text, partial

        except Exception as exc:
            if attempt < OLLAMA_MAX_RETRIES:
                logger.debug("Ollama retry %d detail: %s", attempt + 1, exc)
                logger.warning("Ollama retry %d", attempt + 1)
                time.sleep(OLLAMA_RETRY_SLEEP_SEC)
            else:
                logger.debug("Ollama failure detail: %s", exc)
                raise RuntimeError(
                    f"Ollama request failed after {OLLAMA_MAX_RETRIES + 1} attempts"
                ) from exc

    raise RuntimeError("Ollama request loop exited without result")


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

CONDITIONS = ["plain", "latest", "rag", "trimemory", "trimemory_schema"]


def run_evaluation(
    samples: list[PolicySample],
    model: str,
    out_dir: Path,
    top_k: int = 4,
    conditions: list[str] | None = None,
    temperature: float = 0.0,
    verbose: bool = False,
    packet_log: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[LatencyRecord]]:
    """Run Phase 3.0 evaluation across all conditions and samples.

    Returns (results_list, latency_list).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if conditions is None:
        conditions = list(CONDITIONS)

    # Determine render mode (frozen per model size)
    render_mode = _get_render_mode(model)
    logger.info(
        "Model=%s -- render_mode=%s (frozen)", model, render_mode,
    )

    all_results: list[dict[str, Any]] = []
    all_latency: list[LatencyRecord] = []
    results_path = out_dir / f"results_{model.replace(':', '_')}.jsonl"

    total = len(samples) * len(conditions)
    done = 0

    with open(results_path, "w", encoding="utf-8") as rf:
        for sample in samples:
            for cond in conditions:
                done += 1
                logger.info(
                    "[%d/%d] %s / %s / %s",
                    done, total, model, sample.sample_id, cond,
                )

                # Build prompt
                if cond == "plain":
                    prompt = build_prompt_plain(sample)
                elif cond == "latest":
                    prompt = build_prompt_latest(sample)
                elif cond == "rag":
                    prompt = build_prompt_rag(sample, top_k=top_k)
                elif cond == "trimemory":
                    prompt = build_prompt_trimemory(
                        sample,
                        top_k=top_k,
                        render_mode=render_mode,
                        packet_log=packet_log,
                    )
                elif cond == "trimemory_schema":
                    prompt = build_prompt_trimemory(
                        sample,
                        top_k=top_k,
                        render_mode=render_mode,
                        packet_log=packet_log,
                        use_thin_schema=True,
                    )
                else:
                    continue

                # Select system prompt -- trimemory variants share the same system prompt
                sys_prompt = (
                    SYSTEM_PROMPT_POLICY_TRIMEMORY
                    if cond in ("trimemory", "trimemory_schema")
                    else SYSTEM_PROMPT_POLICY_PLAIN
                )

                # Inference with metrics
                raw_response = ""
                error_msg = ""
                lat_rec: LatencyRecord | None = None
                try:
                    raw_response, lat_rec = run_ollama_with_metrics(
                        prompt=prompt,
                        model=model,
                        system=sys_prompt,
                        temperature=temperature,
                    )
                except Exception as exc:
                    error_msg = "inference failed"
                    logger.debug("%s/%s/%s detail: %s", model, sample.sample_id, cond, exc)
                    logger.error(
                        "%s/%s/%s: inference failed", model, sample.sample_id, cond,
                    )
                    _is_tri = cond in ("trimemory", "trimemory_schema")
                    lat_rec = LatencyRecord(
                        sample_id=sample.sample_id,
                        model=model,
                        condition=cond,
                        render_mode=render_mode if _is_tri else "",
                        prompt_chars=len(prompt),
                        inference_sec=0.0,
                    )

                _is_tri = cond in ("trimemory", "trimemory_schema")
                if lat_rec is not None:
                    lat_rec.sample_id = sample.sample_id
                    lat_rec.condition = cond
                    lat_rec.render_mode = render_mode if _is_tri else ""
                    all_latency.append(lat_rec)

                # Parse and score
                parsed = parse_response(raw_response)
                score = score_policy_sample(sample, parsed)

                # Build result record
                record: dict[str, Any] = {
                    "model": model,
                    "sample_id": sample.sample_id,
                    "condition": cond,
                    "render_mode": render_mode if cond in ("trimemory", "trimemory_schema") else "",
                    "evaluation_type": sample.evaluation_type,
                    "difficulty": sample.difficulty,
                    "failure_class": sample.failure_class,
                    "prompt_length": len(prompt),
                    "raw_response": raw_response,
                    "parse_success": parsed.parse_success,
                    "parsed_answer": parsed.answer[:300],
                    "parsed_current_value": parsed.current_value[:200],
                    "parsed_previous_value": parsed.previous_value[:100],
                    "parsed_reason": parsed.reason[:300],
                    "parsed_evidence_docs": parsed.evidence_docs,
                    "parsed_status_judgment": parsed.status_judgment,
                    "parsed_needs_escalation": parsed.needs_escalation,
                    "parsed_confidence": parsed.confidence,
                    "gold_answer": sample.gold_answer,
                    "gold_status": sample.gold_status,
                    "gold_authority": sample.gold_authority,
                    "score": asdict(score),
                    "latency_sec": lat_rec.inference_sec if lat_rec else 0.0,
                    "input_tokens": lat_rec.input_tokens if lat_rec else 0,
                    "output_tokens": lat_rec.output_tokens if lat_rec else 0,
                    "error": error_msg,
                }
                all_results.append(record)
                rf.write(json.dumps(record, ensure_ascii=False) + "\n")
                rf.flush()

                if verbose:
                    logger.info(
                        "  composite=%.2f val=%.2f status=%.2f auth=%.2f "
                        "stale=%s draft_conf=%s",
                        score.composite_score,
                        score.current_value_match,
                        score.status_score,
                        score.authority_score,
                        score.stale_fact_error,
                        score.draft_confusion_error,
                    )

    return all_results, all_latency


# ---------------------------------------------------------------------------
# Aggregation and report generation
# ---------------------------------------------------------------------------

def _avg(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def export_results_csv(
    results: list[dict[str, Any]],
    out_dir: Path,
) -> Path:
    """Export per-sample results to phase30_results.csv."""
    csv_path = out_dir / "phase30_results.csv"
    fieldnames = [
        "model", "sample_id", "condition", "render_mode",
        "evaluation_type", "difficulty", "failure_class",
        "composite_score", "current_value_match", "status_score",
        "authority_score", "evidence_score", "uncertainty_handling",
        "stale_fact_error", "draft_confusion_error",
        "unsupported_definitive_error", "parse_success",
        "latency_sec", "input_tokens", "output_tokens",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            sc = r["score"]
            writer.writerow({
                "model": r["model"],
                "sample_id": r["sample_id"],
                "condition": r["condition"],
                "render_mode": r.get("render_mode", ""),
                "evaluation_type": r.get("evaluation_type", ""),
                "difficulty": r.get("difficulty", ""),
                "failure_class": r.get("failure_class", ""),
                "composite_score": f"{sc['composite_score']:.3f}",
                "current_value_match": f"{sc['current_value_match']:.3f}",
                "status_score": f"{sc['status_score']:.3f}",
                "authority_score": f"{sc['authority_score']:.3f}",
                "evidence_score": f"{sc['evidence_score']:.3f}",
                "uncertainty_handling": f"{sc['uncertainty_handling']:.3f}",
                "stale_fact_error": sc["stale_fact_error"],
                "draft_confusion_error": sc["draft_confusion_error"],
                "unsupported_definitive_error": sc["unsupported_definitive_error"],
                "parse_success": r["parse_success"],
                "latency_sec": r.get("latency_sec", 0.0),
                "input_tokens": r.get("input_tokens", 0),
                "output_tokens": r.get("output_tokens", 0),
            })
    logger.info("Saved results CSV: %s", csv_path)
    return csv_path


def export_latency_csv(
    latency_records: list[LatencyRecord],
    out_dir: Path,
) -> Path:
    """Export per-inference latency to phase30_latency.csv."""
    csv_path = out_dir / "phase30_latency.csv"
    fieldnames = [
        "model", "sample_id", "condition", "render_mode",
        "prompt_chars", "inference_sec", "input_tokens", "output_tokens",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in latency_records:
            writer.writerow({
                "model": rec.model,
                "sample_id": rec.sample_id,
                "condition": rec.condition,
                "render_mode": rec.render_mode,
                "prompt_chars": rec.prompt_chars,
                "inference_sec": f"{rec.inference_sec:.3f}",
                "input_tokens": rec.input_tokens,
                "output_tokens": rec.output_tokens,
            })
    logger.info("Saved latency CSV: %s", csv_path)
    return csv_path


def build_report(
    results: list[dict[str, Any]],
    latency_records: list[LatencyRecord],
    out_dir: Path,
) -> Path:
    """Build phase30_report.md with aggregate tables and analysis."""
    lines: list[str] = []
    lines.append("# Phase 3.0 Transfer Benchmark -- IT Security Policy Compliance QA\n")

    models = sorted(set(r["model"] for r in results))
    n_samples = len(set(r["sample_id"] for r in results))
    n_total = len(results)

    lines.append(f"Models: {', '.join(models)}")
    lines.append(f"Samples: {n_samples}")
    lines.append(f"Total runs: {n_total}")
    lines.append(
        f"TriMemory packet path: {'YES' if _TRIMEMORY_AVAILABLE else 'FALLBACK (no trn)'}"
    )
    render_modes = set(r.get("render_mode", "") for r in results if r.get("render_mode"))
    lines.append(f"Render modes: {', '.join(sorted(render_modes)) or 'N/A'}\n")

    # -- Model x Condition comparison table --
    lines.append("## Model x Condition Comparison\n")
    lines.append(
        "| Model | Condition | Composite | CurrVal | Status | Authority | "
        "Evidence | Uncertainty | StaleErr | DraftConf | Parse% |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    for model in models:
        for cond in CONDITIONS:
            subset = [
                r for r in results
                if r["model"] == model and r["condition"] == cond
            ]
            if not subset:
                continue
            n = len(subset)
            sc_list = [r["score"] for r in subset]
            parse_n = sum(1 for r in subset if r["parse_success"])
            stale_n = sum(1 for s in sc_list if s["stale_fact_error"])
            draft_n = sum(1 for s in sc_list if s["draft_confusion_error"])
            lines.append(
                f"| {model} | {cond} | "
                f"{_avg([s['composite_score'] for s in sc_list]):.3f} | "
                f"{_avg([s['current_value_match'] for s in sc_list]):.3f} | "
                f"{_avg([s['status_score'] for s in sc_list]):.3f} | "
                f"{_avg([s['authority_score'] for s in sc_list]):.3f} | "
                f"{_avg([s['evidence_score'] for s in sc_list]):.3f} | "
                f"{_avg([s['uncertainty_handling'] for s in sc_list]):.3f} | "
                f"{stale_n}/{n} | {draft_n}/{n} | "
                f"{round(parse_n * 100 / n)}% |"
            )

    # -- Difficulty breakdown --
    lines.append("\n## Score by Difficulty\n")
    lines.append("| Model | Condition | Difficulty | N | Composite | CurrVal | Status |")
    lines.append("|---|---|---|---|---|---|---|")
    difficulties = sorted(set(r.get("difficulty", "?") for r in results))
    for model in models:
        for cond in CONDITIONS:
            for diff in difficulties:
                subset = [
                    r for r in results
                    if r["model"] == model and r["condition"] == cond
                    and r.get("difficulty") == diff
                ]
                if not subset:
                    continue
                sc_list = [r["score"] for r in subset]
                lines.append(
                    f"| {model} | {cond} | {diff} | {len(subset)} | "
                    f"{_avg([s['composite_score'] for s in sc_list]):.3f} | "
                    f"{_avg([s['current_value_match'] for s in sc_list]):.3f} | "
                    f"{_avg([s['status_score'] for s in sc_list]):.3f} |"
                )

    # -- Failure class breakdown --
    lines.append("\n## Failure Class Analysis\n")
    failure_classes = sorted(set(
        r.get("failure_class", "") for r in results if r.get("failure_class")
    ))
    if failure_classes:
        lines.append("| Failure Class | Model | Condition | N | Composite |")
        lines.append("|---|---|---|---|---|")
        for fc in failure_classes:
            for model in models:
                for cond in CONDITIONS:
                    subset = [
                        r for r in results
                        if r["model"] == model and r["condition"] == cond
                        and r.get("failure_class") == fc
                    ]
                    if not subset:
                        continue
                    sc_list = [r["score"] for r in subset]
                    lines.append(
                        f"| {fc} | {model} | {cond} | {len(subset)} | "
                        f"{_avg([s['composite_score'] for s in sc_list]):.3f} |"
                    )

    # -- TriMemory wins/losses per model --
    for model in models:
        lines.append(f"\n## TriMemory vs Others ({model})\n")
        model_results = [r for r in results if r["model"] == model]
        wins, losses, ties = [], [], []

        for sid in sorted(set(r["sample_id"] for r in model_results)):
            sid_by_cond = {
                r["condition"]: r for r in model_results if r["sample_id"] == sid
            }
            if "trimemory" not in sid_by_cond:
                continue
            tri_sc = sid_by_cond["trimemory"]["score"]["composite_score"]
            others = [
                sid_by_cond[c]["score"]["composite_score"]
                for c in ["plain", "latest", "rag"]
                if c in sid_by_cond
            ]
            best_other = max(others) if others else 0.0
            if tri_sc > best_other + 0.01:
                wins.append((sid, tri_sc, best_other))
            elif tri_sc < best_other - 0.01:
                losses.append((sid, tri_sc, best_other))
            else:
                ties.append((sid, tri_sc, best_other))

        lines.append(
            f"Wins: {len(wins)}, Losses: {len(losses)}, Ties: {len(ties)}"
        )
        if wins:
            lines.append("\nWin examples:")
            for sid, ts, bo in sorted(wins, key=lambda x: -(x[1] - x[2]))[:3]:
                lines.append(
                    f"  {sid}: tri={ts:.3f} vs best_other={bo:.3f} (+{ts - bo:.3f})"
                )
        if losses:
            lines.append("\nLoss examples:")
            for sid, ts, bo in sorted(losses, key=lambda x: x[1] - x[2])[:3]:
                lines.append(
                    f"  {sid}: tri={ts:.3f} vs best_other={bo:.3f} ({ts - bo:.3f})"
                )

    # -- Latency summary --
    if latency_records:
        lines.append("\n## Latency Summary\n")
        lines.append("| Model | Condition | N | Avg Latency (s) | Avg Input Tok | Avg Output Tok |")
        lines.append("|---|---|---|---|---|---|")
        for model in models:
            for cond in CONDITIONS:
                subset = [
                    rec for rec in latency_records
                    if rec.model == model and rec.condition == cond
                ]
                if not subset:
                    continue
                avg_lat = _avg([rec.inference_sec for rec in subset])
                avg_in = _avg([float(rec.input_tokens) for rec in subset])
                avg_out = _avg([float(rec.output_tokens) for rec in subset])
                lines.append(
                    f"| {model} | {cond} | {len(subset)} | "
                    f"{avg_lat:.2f} | {avg_in:.0f} | {avg_out:.0f} |"
                )

    report_path = out_dir / "phase30_report.md"
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    logger.info("Saved report: %s", report_path)

    # Print to stdout (ASCII-safe for Windows cp932 terminals)
    safe_text = report_text.encode("ascii", errors="replace").decode("ascii")
    print("\n" + "=" * 64)
    print(safe_text)
    print("=" * 64)

    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TriMemory Phase 3.0 Transfer Benchmark -- "
            "IT Security Policy Compliance QA"
        ),
    )
    parser.add_argument(
        "--data", type=str,
        default="data/policybench/policy_v1.jsonl",
        help="Path to PolicyBench JSONL data (default: data/policybench/policy_v1.jsonl)",
    )
    parser.add_argument(
        "--model", type=str, nargs="+", default=["llama3.2:3b"],
        help="Ollama model name(s) (e.g. llama3.2:3b qwen2.5:7b)",
    )
    parser.add_argument(
        "--out", type=str, default="artifacts/phase30_transfer",
        help="Output directory for CSV and report files",
    )
    parser.add_argument(
        "--condition", type=str, default="all",
        help="Comma-separated conditions or 'all' (plain,latest,rag,trimemory)",
    )
    parser.add_argument(
        "--split", type=str, default="dev",
        help="Data split to evaluate ('dev', 'test', or 'all')",
    )
    parser.add_argument(
        "--top-k", type=int, default=4,
        help="Number of chunks for RAG retrieval",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of samples (0 = all)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-sample scores during evaluation",
    )

    args = parser.parse_args()

    if not math.isfinite(args.temperature):
        logger.error("--temperature must be finite, got %s", args.temperature)
        sys.exit(1)

    # Resolve paths relative to project root when not absolute
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    # Load samples
    split_filter: str | None = args.split if args.split != "all" else None
    samples = load_policy_samples(data_path, split_filter=split_filter)
    if not samples:
        logger.error("No samples loaded from %s (split=%s)", data_path, args.split)
        sys.exit(1)
    if args.limit > 0:
        samples = samples[:args.limit]
    logger.info("Evaluating %d samples (split=%s)", len(samples), args.split)

    # Determine conditions
    if args.condition == "all":
        conditions = list(CONDITIONS)
    else:
        conditions = [c.strip() for c in args.condition.split(",")]

    # Validate conditions
    invalid = [c for c in conditions if c not in CONDITIONS]
    if invalid:
        logger.error("Unknown conditions: %s (valid: %s)", invalid, CONDITIONS)
        sys.exit(1)

    # Check TriMemory availability when trimemory condition is requested
    _tri_conds = {"trimemory", "trimemory_schema"}
    if _tri_conds.intersection(conditions) and not _TRIMEMORY_AVAILABLE:
        logger.warning(
            "TriMemory imports failed -- trimemory condition will use fallback mode. "
            "Ensure the 'trn' package is importable from %s",
            PROJECT_ROOT / "src",
        )

    # Verify Ollama for each model
    for model_name in args.model:
        try:
            run_ollama("Hello", model=model_name, max_tokens=8)
            logger.info("Ollama OK (model=%s)", model_name)
        except Exception as exc:
            logger.debug("Ollama health check detail for %s: %s", model_name, exc)
            logger.error("Ollama health check failed for %s", model_name)
            sys.exit(1)

    # Run evaluation
    out_dir.mkdir(parents=True, exist_ok=True)
    packet_log: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    all_latency: list[LatencyRecord] = []

    for model_name in args.model:
        logger.info("=== Running model: %s ===", model_name)
        results, latency = run_evaluation(
            samples=samples,
            model=model_name,
            out_dir=out_dir,
            top_k=args.top_k,
            conditions=conditions,
            temperature=args.temperature,
            verbose=args.verbose,
            packet_log=packet_log,
        )
        all_results.extend(results)
        all_latency.extend(latency)

    # Save packet debug log
    if packet_log:
        packets_path = out_dir / "phase30_packets.jsonl"
        with open(packets_path, "w", encoding="utf-8") as fh:
            for p in packet_log:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        logger.info("Saved %d packets to %s", len(packet_log), packets_path)

    # Export artifacts
    export_results_csv(all_results, out_dir)
    export_latency_csv(all_latency, out_dir)
    build_report(all_results, all_latency, out_dir)

    logger.info("Phase 3.0 evaluation complete. Artifacts in: %s", out_dir)


if __name__ == "__main__":
    main()
