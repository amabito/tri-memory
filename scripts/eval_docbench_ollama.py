"""TriMem-DocBench evaluation pipeline -- Phase 2.7: factor decomposition.

Usage:
    # Phase 2.5 (natural language packet + field scoring + multi-model):
    python scripts/eval_docbench_ollama.py \
        --data data/docbench/final/pract_v1.jsonl \
        --model llama3.2:3b qwen2.5:7b \
        --out artifacts/docbench_ollama \
        --top-k 4 --split dev --phase2 --verbose

    # Ablation: compare render modes
    python scripts/eval_docbench_ollama.py \
        --data data/docbench/final/pract_v1.jsonl \
        --model llama3.2:3b --out artifacts/docbench_ollama \
        --top-k 4 --split dev --phase2 --verbose \
        --trimemory-render-mode packet_plus_short_refs

Conditions:
    A) plain      -- all docs concatenated
    B) latest     -- current_main / latest doc only
    C) rag        -- top-k chunks by lexical overlap
    D) trimemory  -- CompactMemoryPacket via real pipeline, NL rendered
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path for trn imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import requests
except ImportError:
    import urllib.request
    import urllib.error
    requests = None  # type: ignore[assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TriMemory imports (fail-safe)
# ---------------------------------------------------------------------------

_TRIMEMORY_AVAILABLE = False
try:
    from trimemory.disentangled_archive import ChunkMetadata, MetadataParser
    from trimemory.selective_memory_messenger import SelectiveMemoryMessenger
    from trimemory.memory_mediator import MemoryMediator
    from trimemory.memory_packet import CompactMemoryPacket
    _TRIMEMORY_AVAILABLE = True
    logger.info("TriMemory imports OK -- packet path available")
except ImportError as exc:
    logger.warning("TriMemory imports failed: %s -- fallback only", exc)


# ---------------------------------------------------------------------------
# Task 1: Data Loader
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    sample_id: str
    type: str
    difficulty: str
    documents: list[dict[str, Any]]
    question: str
    gold_answer_text: str
    gold_answer_structured: dict[str, Any]
    gold_evidence: list[dict[str, Any]]
    expected_failure_modes: list[str]
    split: str = "dev"
    required_capabilities: list[str] = field(default_factory=list)
    distractors: list[dict[str, Any]] = field(default_factory=list)
    evaluator_notes: str = ""


def load_samples(path: Path, split_filter: str | None = "dev") -> list[EvalSample]:
    """Load samples from JSON or JSONL file."""
    raw: list[dict] = []
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            raw = data if isinstance(data, list) else [data]

    samples = []
    for d in raw:
        s = EvalSample(
            sample_id=d["sample_id"],
            type=d.get("type", ""),
            difficulty=d.get("difficulty", ""),
            documents=d.get("documents", []),
            question=d["question"],
            gold_answer_text=d.get("gold_answer_text", ""),
            gold_answer_structured=d.get("gold_answer_structured", {}),
            gold_evidence=d.get("gold_evidence", []),
            expected_failure_modes=d.get("expected_failure_modes", []),
            split=d.get("split", "dev"),
            required_capabilities=d.get("required_capabilities", []),
            distractors=d.get("distractors", []),
            evaluator_notes=d.get("evaluator_notes", ""),
        )
        if split_filter is None or s.split == split_filter:
            samples.append(s)
    return samples


# ---------------------------------------------------------------------------
# Task 2: Ollama Wrapper
# ---------------------------------------------------------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MAX_RETRIES = 2
TIMEOUT_SEC = 120


def run_ollama(
    prompt: str,
    model: str,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Call Ollama API and return the response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    url = f"{OLLAMA_URL}/api/generate"
    body = json.dumps(payload).encode("utf-8")

    for attempt in range(MAX_RETRIES + 1):
        try:
            if requests is not None:
                resp = requests.post(url, json=payload, timeout=TIMEOUT_SEC)
                resp.raise_for_status()
                return resp.json().get("response", "")
            else:
                req = urllib.request.Request(
                    url, data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("response", "")
        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.warning("Ollama retry %d: %s", attempt + 1, e)
                time.sleep(2)
            else:
                raise RuntimeError(
                    f"Ollama failed after {MAX_RETRIES + 1} attempts: {e}"
                ) from e
    return ""


# ---------------------------------------------------------------------------
# Task 3: Prompt Builders (4 conditions)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_PLAIN = (
    "You are a document analysis assistant. "
    "Answer questions based ONLY on the provided documents. "
    "Always respond in Japanese. "
    "If the documents contain contradictions or unresolved changes, say so explicitly. "
    'If you cannot determine the answer with certainty, say "undetermined". '
    "\n\nYou MUST respond in this exact JSON format:\n"
    "{\n"
    '  "answer": "your answer text (in Japanese)",\n'
    '  "current_value": "the current/latest value if applicable",\n'
    '  "previous_value": "the previous value if applicable, or empty",\n'
    '  "reason": "why the value changed or why this is the answer",\n'
    '  "evidence_docs": ["doc_01", "doc_03"],\n'
    '  "evidence_spans": ["doc_01:sec_3_2"],\n'
    '  "status_judgment": "confirmed|undetermined|contradictory",\n'
    '  "needs_escalation": false,\n'
    '  "confidence": 0.8\n'
    "}\n"
    "Return ONLY valid JSON. No other text."
)

SYSTEM_PROMPT_TRIMEMORY = (
    "You are a document analysis assistant. "
    "You are given a MEMORY SUMMARY built from multiple documents.\n\n"
    "Rules for using the memory summary:\n"
    '1. Use "Current formal facts" as the default source of truth.\n'
    '2. Use "Older or superseded facts" only for historical comparison.\n'
    '3. Use "Pending or provisional updates" only if the question asks '
    "about proposals, pending changes, or uncertainty.\n"
    "4. If conflicts are detected and no formally approved current value "
    'is available, answer "uncertain" and set needs_escalation=true.\n'
    "5. Do not promote draft or provisional values to current facts "
    "unless the summary explicitly says they are approved.\n"
    '6. Copy the exact value from "Current formal facts" into '
    '"current_value" if it answers the question.\n\n'
    "Always respond in Japanese.\n\n"
    "You MUST respond in this exact JSON format:\n"
    "{\n"
    '  "answer": "your answer text (in Japanese)",\n'
    '  "current_value": "the current/latest value from Current formal facts",\n'
    '  "previous_value": "the previous value from Older facts, or empty",\n'
    '  "reason": "why the value changed or why this is the answer",\n'
    '  "evidence_docs": ["doc_01", "doc_03"],\n'
    '  "evidence_spans": ["doc_01:sec_3_2"],\n'
    '  "status_judgment": "confirmed|undetermined|contradictory",\n'
    '  "needs_escalation": false,\n'
    '  "confidence": 0.8\n'
    "}\n"
    "Return ONLY valid JSON. No other text."
)


def _format_doc(doc: dict) -> str:
    """Format a single document for prompt inclusion."""
    meta = doc.get("metadata", {})
    header = f"[{doc.get('doc_id', '?')}] {doc.get('title', 'Untitled')}"
    header += (
        f" (role={doc.get('role', '?')}, date={meta.get('date', '?')}, "
        f"version={meta.get('version', '?')}, status={meta.get('status', '?')})"
    )
    content = doc.get("content", "")
    return f"{header}\n{content}"


def build_prompt_plain(sample: EvalSample) -> str:
    """Condition A: All documents concatenated."""
    docs_text = "\n\n---\n\n".join(
        _format_doc(doc) for doc in sample.documents
    )
    return f"""Documents:

{docs_text}

Question: {sample.question}

Respond in the JSON format specified."""


def build_prompt_latest(sample: EvalSample) -> str:
    """Condition B: Latest/current document only."""
    best = sample.documents[0]
    for doc in sample.documents:
        role = doc.get("role", "")
        status = doc.get("metadata", {}).get("status", "")
        if role == "current_main":
            best = doc
            break
        if status in ("current", "final"):
            best = doc

    return f"""Document:

{_format_doc(best)}

Question: {sample.question}

Respond in the JSON format specified."""


def _simple_chunk(doc: dict, max_chunk_chars: int = 300) -> list[dict]:
    """Split a document into paragraph-level chunks."""
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
                    "doc_id": doc_id,
                    "span_id": f"para_{i}_chunk_{j // max_chunk_chars}",
                    "title": title,
                    "metadata": meta,
                    "role": role,
                })
        else:
            chunks.append({
                "text": para,
                "doc_id": doc_id,
                "span_id": f"para_{i}",
                "title": title,
                "metadata": meta,
                "role": role,
            })
    return chunks


def _lexical_score(query: str, text: str) -> float:
    """Simple lexical overlap score (Jaccard on character trigrams)."""
    def trigrams(s: str) -> set[str]:
        s = s.lower()
        return {s[i:i+3] for i in range(max(0, len(s) - 2))}
    q_tri = trigrams(query)
    t_tri = trigrams(text)
    if not q_tri or not t_tri:
        return 0.0
    return len(q_tri & t_tri) / len(q_tri | t_tri)


def build_prompt_rag(sample: EvalSample, top_k: int = 4) -> str:
    """Condition C: RAG with lexical retrieval."""
    all_chunks = []
    for doc in sample.documents:
        all_chunks.extend(_simple_chunk(doc))

    scored = [
        (c, _lexical_score(sample.question, c["text"]))
        for c in all_chunks
    ]
    scored.sort(key=lambda x: -x[1])
    top_chunks = scored[:top_k]

    chunks_text = "\n\n---\n\n".join(
        f"[{c['doc_id']}:{c['span_id']}] ({c.get('title', '')})\n{c['text']}"
        for c, _ in top_chunks
    )

    return f"""Retrieved document chunks (ranked by relevance):

{chunks_text}

Question: {sample.question}

Respond in the JSON format specified."""


# ---------------------------------------------------------------------------
# TriMemory Packet Builder (real pipeline)
# ---------------------------------------------------------------------------

# Role -> provenance mapping for PRACT documents
_ROLE_TO_PROVENANCE: dict[str, str] = {
    "current_main": "spec",
    "previous_version": "spec",
    "change_notice": "note",
    "minutes": "meeting",
    "table": "table",
    "note": "note",
    "faq": "faq",
    "calc": "calc",
    "draft": "spec",
    "final": "spec",
    "supplementary": "note",
}

# Status -> trust score mapping
_STATUS_TRUST: dict[str, float] = {
    "current": 0.9,
    "final": 0.85,
    "draft": 0.4,
    "provisional": 0.3,
    "superseded": 0.2,
    "unknown": 0.5,
}


# ---------------------------------------------------------------------------
# Phase 2.8: Enhanced Fact Extraction + Status Detection
# ---------------------------------------------------------------------------

# Markdown bold KV: **key**: value or **key** = value (in lists, tables, etc.)
_MD_BOLD_KV_PATTERN = re.compile(
    r"\*\*([^*]{1,60})\*\*\s*[:=]\s*(.{1,500}?)(?:\n|$)",
    re.MULTILINE,
)

# Table row pattern: | cell1 | cell2 | (extract key-value from 2-column tables)
_TABLE_ROW_PATTERN = re.compile(
    r"^\s*\|\s*(.{1,80}?)\s*\|\s*(.{1,200}?)\s*\|",
    re.MULTILINE,
)

# Numbered list with bold key: 1. **key**: value
_NUMBERED_BOLD_KV = re.compile(
    r"^\s*\d+\.\s*\*\*([^*]{1,60})\*\*\s*[:：]\s*(.{1,300}?)$",
    re.MULTILINE,
)

# Bullet list with bold key: - **key**: value
_BULLET_BOLD_KV = re.compile(
    r"^\s*[-*]\s*\*\*([^*]{1,60})\*\*\s*[:：]\s*(.{1,300}?)$",
    re.MULTILINE,
)

# Price pattern: ¥N or ¥N,NNN or $N.NN
_PRICE_PATTERN = re.compile(
    r"[¥$]\s*(\d[\d,]*(?:\.\d+)?)\s*(?:/\s*.+?)?(?:\s*[(（]|$|\n|。)",
)

# Japanese counter/unit pattern (broader than _NUMERIC_UNIT_PATTERN)
_JA_NUMERIC_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(文字|日|件|名|分|時間|ユーザー|回|円|年|月|接続|台)",
)

# Change marker: 改定前/改定後, 変更前/変更後 with bold
_MD_CHANGE_PATTERN = re.compile(
    r"\*\*(?:改定|変更)(?:前|後)\*\*\s*[:：]\s*(.{1,300}?)$",
    re.MULTILINE,
)

# Inline bold value near keyword: "勤続10年以上の従業員には年間 **25日** の..."
_INLINE_BOLD_VALUE = re.compile(
    r"(\S{2,20})\s*\*\*(\d+(?:\.\d+)?[^*]{0,10}?)\*\*",
)

# Deprecated/non-recommended endpoint detection
_DEPRECATED_PATTERN = re.compile(
    r"(?:非推奨|deprecated|廃止予定|obsolete)\s*(?:[(（]([^)）]+)[)）])?",
    re.IGNORECASE,
)

# Status-from-text indicators (stronger than base parser)
_STATUS_TEXT_INDICATORS: list[tuple[str, re.Pattern[str]]] = [
    ("current", re.compile(r"(?:本版は.*廃止し|即日有効|現行|currently\s+effective)", re.IGNORECASE)),
    ("current", re.compile(r"(?:v\d+(?:\.\d+)*)\s*\(現行\)", re.IGNORECASE)),
    ("superseded", re.compile(r"(?:を廃止する|を.*?に置き換え|旧版|旧価格)", re.IGNORECASE)),
    ("draft", re.compile(r"(?:DRAFT|草稿|案\s*[:)]|下書き|未署名)", re.IGNORECASE)),
]

# Canonical slot definitions for DocBench PRACT data
_CANONICAL_SLOTS: dict[str, dict] = {
    "sla_target_percentage": {
        "aliases_re": re.compile(
            r"(?:SLA|稼働率|availability|uptime|サービスレベル).*?(\d+(?:\.\d+)?)\s*%",
            re.IGNORECASE,
        ),
        "context_re": re.compile(
            r"(?:稼働率保証|目標稼働率|SLA\s*target|service\s*level)",
            re.IGNORECASE,
        ),
    },
    "password_minimum_length": {
        "aliases_re": re.compile(
            r"(?:最低文字数|パスワード.*?文字|minimum.*?length|password.*?char).*?(\d+)",
            re.IGNORECASE,
        ),
        "context_re": re.compile(
            r"(?:パスワード|password|文字数)",
            re.IGNORECASE,
        ),
    },
    "monthly_price_per_user": {
        "aliases_re": re.compile(
            r"(?:月額|ユーザー単価|unit\s*price).*?[¥$]\s*(\d[\d,]*)",
            re.IGNORECASE,
        ),
        "context_re": re.compile(
            r"(?:月額|単価|ユーザー|price|料金)",
            re.IGNORECASE,
        ),
    },
    "payment_terms_days": {
        "aliases_re": re.compile(
            r"(?:NET\s*(\d+)|支払期限.*?(\d+)\s*日|payment.*?(\d+)\s*days)",
            re.IGNORECASE,
        ),
        "context_re": re.compile(
            r"(?:支払|payment|NET|請求|invoice)",
            re.IGNORECASE,
        ),
    },
    "annual_leave_days": {
        "aliases_re": re.compile(
            r"(?:年次有給|有給休暇|年間|annual\s*leave|paid\s*leave).*?(\d+)\s*日",
            re.IGNORECASE,
        ),
        "context_re": re.compile(
            r"(?:有給|休暇|leave|vacation)",
            re.IGNORECASE,
        ),
    },
    "vulnerability_count": {
        "aliases_re": re.compile(
            r"(?:未是正|Critical|脆弱性).*?(\d+)\s*件",
            re.IGNORECASE,
        ),
        "context_re": re.compile(
            r"(?:脆弱性|vulnerability|Critical|未是正|監査)",
            re.IGNORECASE,
        ),
    },
    "connection_limit": {
        "aliases_re": re.compile(
            r"(?:同時接続|接続上限|concurrent|connection\s*limit).*?(\d+)",
            re.IGNORECASE,
        ),
        "context_re": re.compile(
            r"(?:接続|connection|concurrent|capacity)",
            re.IGNORECASE,
        ),
    },
    "admin_user_limit": {
        "aliases_re": re.compile(
            r"(?:管理者|admin).*?(?:上限|limit|最大).*?(\d+)",
            re.IGNORECASE,
        ),
        "context_re": re.compile(
            r"(?:管理者|admin|ユーザー数|上限|フリープラン)",
            re.IGNORECASE,
        ),
    },
}


def _extract_enhanced_facts(text: str) -> list[str]:
    """Phase 2.8: Extract facts that MetadataParser misses.

    Targets:
    - Markdown bold key-value pairs (**key**: value)
    - Table rows (| key | value |)
    - Numbered/bulleted lists with bold keys
    - Price values
    - Japanese counter/unit values
    """
    facts: list[str] = []
    seen_keys: set[str] = set()

    # 1. Markdown bold KV
    for m in _MD_BOLD_KV_PATTERN.finditer(text):
        key = m.group(1).strip()
        val = m.group(2).strip()
        if key and val and key.lower() not in seen_keys:
            seen_keys.add(key.lower())
            facts.append(f"{key}: {val}")

    # 2. Numbered list bold KV
    for m in _NUMBERED_BOLD_KV.finditer(text):
        key = m.group(1).strip()
        val = m.group(2).strip()
        if key and val and key.lower() not in seen_keys:
            seen_keys.add(key.lower())
            facts.append(f"{key}: {val}")

    # 3. Bullet list bold KV
    for m in _BULLET_BOLD_KV.finditer(text):
        key = m.group(1).strip()
        val = m.group(2).strip()
        if key and val and key.lower() not in seen_keys:
            seen_keys.add(key.lower())
            facts.append(f"{key}: {val}")

    # 4. Table rows (skip header-like rows)
    header_keywords = {"項目", "header", "---", "field", "column"}
    for m in _TABLE_ROW_PATTERN.finditer(text):
        key = m.group(1).strip().strip("*").strip()
        val = m.group(2).strip().strip("*").strip()
        if (
            key and val
            and key.lower() not in header_keywords
            and val.lower() not in header_keywords
            and "---" not in val
            and key.lower() not in seen_keys
        ):
            seen_keys.add(key.lower())
            facts.append(f"{key}: {val}")

    # 5. Change markers (bold)
    for m in _MD_CHANGE_PATTERN.finditer(text):
        val = m.group(1).strip()
        if val:
            facts.append(f"[change] {val}")

    # 6. Inline bold values near context keywords
    for m in _INLINE_BOLD_VALUE.finditer(text):
        context_word = m.group(1).strip()
        bold_val = m.group(2).strip()
        if bold_val and context_word.lower() not in seen_keys:
            key = f"{context_word} {bold_val}"
            if key.lower() not in seen_keys:
                seen_keys.add(key.lower())
                facts.append(f"{context_word}: {bold_val}")

    # 7. Deprecated/non-recommended markers
    for m in _DEPRECATED_PATTERN.finditer(text):
        detail = m.group(1) or ""
        # Get surrounding context (100 chars before match)
        start = max(0, m.start() - 100)
        context = text[start:m.end()]
        facts.append(f"[deprecated] {context.strip()[:120]}")

    return facts[:30]


def _extract_canonical_slot_values(
    text: str,
    doc_status: str = "unknown",
) -> list[dict[str, str]]:
    """Phase 2.8: Extract values for canonical semantic slots.

    Returns list of dicts with keys: slot, value, status, context.
    """
    results: list[dict[str, str]] = []
    lines = text.split("\n")

    for slot_name, slot_def in _CANONICAL_SLOTS.items():
        aliases_re = slot_def["aliases_re"]
        context_re = slot_def["context_re"]

        # Search line-by-line for context-aware extraction
        for i, line in enumerate(lines):
            m = aliases_re.search(line)
            if not m:
                continue

            # Extract the captured value (first non-None group)
            value = None
            for g in m.groups():
                if g is not None:
                    value = g
                    break
            if value is None:
                continue

            # Check context window (2 lines before, 2 lines after)
            context_window = "\n".join(
                lines[max(0, i - 2):min(len(lines), i + 3)]
            )

            # Confidence: higher if context keyword is in nearby text
            has_context = bool(context_re.search(context_window))

            results.append({
                "slot": slot_name,
                "value": value.replace(",", ""),
                "status": doc_status,
                "context": line.strip()[:100],
                "confidence": "high" if has_context else "medium",
            })

    return results


def _detect_text_status(text: str, doc_status: str = "unknown") -> str:
    """Phase 2.8: Detect status from text indicators when doc-level is ambiguous.

    Returns refined status or the original doc_status if no strong signal.
    """
    if doc_status in ("current", "superseded", "draft"):
        return doc_status  # trust doc-level metadata

    for status, pattern in _STATUS_TEXT_INDICATORS:
        if pattern.search(text):
            return status

    return doc_status


def _normalize_fact_value(value: str) -> str:
    """Phase 2.8: Normalize extracted values to canonical form."""
    v = value.strip()
    # Yen normalization
    v = re.sub(r"[¥]\s*", "¥", v)
    # Percentage normalization
    v = re.sub(r"(\d+(?:\.\d+)?)\s*percent", r"\1%", v, flags=re.IGNORECASE)
    # Remove markdown bold
    v = v.replace("**", "")
    return v


def _merge_enhanced_facts(
    base_facts: list[str],
    enhanced_facts: list[str],
) -> list[str]:
    """Phase 2.8: Merge enhanced facts into base, deduplicating by key.

    Base facts (from MetadataParser) take precedence for same keys.
    Enhanced facts fill gaps.
    """
    # Build set of existing base fact keys (rough: first part before ':')
    base_keys: set[str] = set()
    for f in base_facts:
        if ":" in f:
            key = f.split(":")[0].strip().lower()
            base_keys.add(key)

    merged = list(base_facts)
    for ef in enhanced_facts:
        if ":" in ef:
            key = ef.split(":")[0].strip().lower()
            if key not in base_keys:
                merged.append(_normalize_fact_value(ef))
                base_keys.add(key)
        elif ef.startswith("[change]") or ef.startswith("[deprecated]"):
            # Change/deprecated markers always add
            merged.append(ef)

    return merged[:30]


def _build_trimemory_chunks(sample: EvalSample) -> list[dict[str, Any]]:
    """Build chunk dicts with full MetadataParser-extracted metadata.

    Phase 2.8: Enhanced with markdown bold KV extraction, table row
    extraction, canonical slot extraction, and improved status detection.
    """
    parser = MetadataParser()
    all_items: list[dict[str, Any]] = []

    for doc in sample.documents:
        doc_id = doc.get("doc_id", "")
        title = doc.get("title", "")
        role = doc.get("role", "")
        meta = doc.get("metadata", {})
        content = doc.get("content", "")

        # Determine status and provenance from PRACT metadata
        doc_status = meta.get("status", "unknown")
        doc_provenance = _ROLE_TO_PROVENANCE.get(role, "unknown")
        doc_trust = _STATUS_TRUST.get(doc_status, 0.5)

        # Split into chunks
        chunks = _simple_chunk(doc)

        for chunk in chunks:
            # Use MetadataParser to extract facts from actual text
            parsed_meta = parser.parse(
                text=chunk["text"],
                doc_id=doc_id,
                span_id=chunk["span_id"],
                title=title,
            )

            # Override status/provenance/trust with PRACT document-level values
            # (MetadataParser detects from text heuristics, but PRACT metadata
            # is authoritative)
            chunk_meta = parsed_meta.to_dict()
            chunk_meta["status"] = doc_status
            chunk_meta["provenance"] = doc_provenance
            chunk_meta["source_trust"] = doc_trust

            # Phase 2.8: Enhanced extraction
            enhanced_facts = _extract_enhanced_facts(chunk["text"])
            chunk_meta["exact_fact_candidates"] = _merge_enhanced_facts(
                chunk_meta.get("exact_fact_candidates", []),
                enhanced_facts,
            )

            # Phase 2.8: Canonical slot extraction
            slot_values = _extract_canonical_slot_values(
                chunk["text"], doc_status,
            )
            if slot_values:
                chunk_meta.setdefault("canonical_slots", []).extend(slot_values)

            # Phase 2.8: Refine status from text when doc-level is ambiguous
            if doc_status in ("unknown", "final"):
                refined = _detect_text_status(chunk["text"], doc_status)
                if refined != doc_status:
                    chunk_meta["status"] = refined

            all_items.append({
                "text": chunk["text"],
                "doc_id": doc_id,
                "span_id": chunk["span_id"],
                "title": title,
                "metadata": chunk_meta,
            })

    return all_items


_TRIMEMORY_RENDER_MODES = (
    "packet_only",
    "packet_plus_short_refs",
    "packet_plus_raw",
    "packet_plus_short_refs_semantic",
    "packet_plus_short_refs_semantic_answer",
    # Phase 2.7 factor decomposition variants
    "semantic_en",                    # Factor A=en, B=relabel, C=no_answer, D=no_refs
    "semantic_ja",                    # Factor A=ja, B=relabel, C=no_answer, D=no_refs
    "semantic_en_ja",                 # Factor A=en_ja, B=relabel, C=no_answer, D=no_refs
    "short_refs_semantic_en_ja",      # Factor A=en_ja, B=relabel, C=no_answer, D=short_refs
    "short_refs_en_ja_no_relabel",    # Factor A=en_ja, B=no_relabel, C=no_answer, D=short_refs
    "semantic_answer_en_ja",          # Factor A=en_ja, B=relabel, C=answer, D=no_refs
)


def build_prompt_trimemory(
    sample: EvalSample,
    top_k: int = 4,
    packet_log: list[dict] | None = None,
    render_mode: str = "packet_only",
) -> str:
    """Condition D: TriMemory compact packet via real pipeline.

    Pipeline: MetadataParser -> SelectiveMemoryMessenger -> MemoryMediator
    render_mode: packet_only | packet_plus_short_refs | packet_plus_raw
                 | packet_plus_short_refs_semantic | packet_plus_short_refs_semantic_answer
    """
    if not _TRIMEMORY_AVAILABLE:
        logger.warning(
            "[%s] TriMemory not available, using fallback", sample.sample_id,
        )
        return _build_prompt_trimemory_fallback(sample)

    messenger = SelectiveMemoryMessenger()
    mediator = MemoryMediator()

    # Build chunks with MetadataParser-extracted metadata
    all_items = _build_trimemory_chunks(sample)
    n_facts_extracted = sum(
        len(item["metadata"].get("exact_fact_candidates", []))
        for item in all_items
    )
    n_entities_extracted = sum(
        len(item["metadata"].get("entity_value_pairs", []))
        for item in all_items
    )
    logger.info(
        "  [PACKET] %d chunks, %d facts, %d entities extracted",
        len(all_items), n_facts_extracted, n_entities_extracted,
    )

    # Build packet via SelectiveMemoryMessenger
    packet = messenger.build_packet(
        all_items,
        query=sample.question,
        max_exact_fact_fields=12,
        max_state_hints=6,
        max_source_refs=8,
    )

    # Resolve via MemoryMediator
    packet = mediator.resolve(packet, prefer_current=True)

    logger.info(
        "  [PACKET] After mediation: %d facts, %d hints, %d anomalies, "
        "has_conflicts=%s, summary=%s",
        packet.fact_count, packet.hint_count,
        len(packet.anomaly_flags), packet.has_conflicts(),
        packet.packet_summary[:80] if packet.packet_summary else "empty",
    )

    # Phase 2.10: Memory Compiler -- context-aware packet assembly
    # Collect canonical slots from all chunks for compatibility scoring
    all_canonical_slots: list[dict[str, str]] = []
    for item in all_items:
        all_canonical_slots.extend(
            item["metadata"].get("canonical_slots", [])
        )

    compiler_diag = _compile_memory_packet(
        packet,
        query=sample.question,
        canonical_slots=all_canonical_slots,
        max_facts=12,
        max_per_family=2,
    )

    logger.info(
        "  [COMPILER] %d -> %d facts (deduped %d, mismatch penalties %d, "
        "query_unit=%s)",
        compiler_diag["original_facts"],
        compiler_diag["final_facts"],
        compiler_diag["n_deduped_rows"],
        compiler_diag["n_mismatch_penalties"],
        compiler_diag["query_unit_hint"],
    )

    # Save packet for debug (always machine-readable)
    if packet_log is not None:
        packet_log.append({
            "sample_id": sample.sample_id,
            "n_input_chunks": len(all_items),
            "n_facts_extracted": n_facts_extracted,
            "n_entities_extracted": n_entities_extracted,
            "render_mode": render_mode,
            "packet": packet.to_dict(),
            "compiler_diagnostics": compiler_diag,
        })

    # Render packet as natural language
    nl_packet = render_packet_for_llm(packet)

    # Build prompt based on render mode
    if render_mode == "packet_only":
        return f"""{nl_packet}

Question: {sample.question}

Use the MEMORY SUMMARY above to answer. Respond in the JSON format specified."""

    elif render_mode == "packet_plus_short_refs":
        # Add 1-2 line short source snippets per referenced doc
        short_refs = _format_short_source_refs(sample.documents, packet)
        return f"""{nl_packet}

Short source excerpts:
{short_refs}

Question: {sample.question}

Use the MEMORY SUMMARY above to answer. Respond in the JSON format specified."""

    elif render_mode == "packet_plus_short_refs_semantic":
        # Semantic relabeling + query-grounded fact selection + short refs
        semantic_packet = render_packet_semantic(packet, sample.question, include_answer_slot=False)
        short_refs = _format_short_source_refs(sample.documents, packet)
        return f"""{semantic_packet}

Short source excerpts:
{short_refs}

Question: {sample.question}

Use the MEMORY SUMMARY above to answer. Respond in the JSON format specified."""

    elif render_mode == "packet_plus_short_refs_semantic_answer":
        # Semantic relabeling + query-grounded fact selection + answer slot + short refs
        semantic_packet = render_packet_semantic(packet, sample.question, include_answer_slot=True)
        short_refs = _format_short_source_refs(sample.documents, packet)
        return f"""{semantic_packet}

Short source excerpts:
{short_refs}

Question: {sample.question}

Use the MEMORY SUMMARY above to answer. Copy the suggested answer into current_value if you agree with it. Respond in the JSON format specified."""

    elif render_mode.startswith("semantic_") or render_mode.startswith("short_refs_"):
        # Phase 2.7 factor decomposition variants
        # Parse factor settings from render mode name
        _FACTOR_MAP: dict[str, dict] = {
            "semantic_en":                 {"alias_mode": "en",    "relabel": True,  "answer": False, "refs": False},
            "semantic_ja":                 {"alias_mode": "ja",    "relabel": True,  "answer": False, "refs": False},
            "semantic_en_ja":              {"alias_mode": "en_ja", "relabel": True,  "answer": False, "refs": False},
            "short_refs_semantic_en_ja":   {"alias_mode": "en_ja", "relabel": True,  "answer": False, "refs": True},
            "short_refs_en_ja_no_relabel": {"alias_mode": "en_ja", "relabel": False, "answer": False, "refs": True},
            "semantic_answer_en_ja":       {"alias_mode": "en_ja", "relabel": True,  "answer": True,  "refs": False},
        }
        factors = _FACTOR_MAP.get(render_mode, {
            "alias_mode": "en_ja", "relabel": True, "answer": False, "refs": False,
        })

        semantic_packet = render_packet_semantic(
            packet, sample.question,
            include_answer_slot=factors["answer"],
            alias_mode=factors["alias_mode"],
            relabel_enabled=factors["relabel"],
        )

        parts = [semantic_packet]

        if factors["refs"]:
            short_refs = _format_short_source_refs(sample.documents, packet)
            parts.append(f"\nShort source excerpts:\n{short_refs}")

        parts.append(f"\nQuestion: {sample.question}")

        if factors["answer"]:
            parts.append(
                "\nUse the MEMORY SUMMARY above to answer. "
                "Copy the suggested answer into current_value if you agree with it. "
                "Respond in the JSON format specified."
            )
        else:
            parts.append(
                "\nUse the MEMORY SUMMARY above to answer. "
                "Respond in the JSON format specified."
            )

        return "\n".join(parts)

    else:  # packet_plus_raw
        doc_context = _format_doc_context(sample.documents)
        raw_excerpts = "\n\n---\n\n".join(
            _format_doc(doc) for doc in sample.documents
        )
        return f"""{nl_packet}

Document context:
{doc_context}

Full document excerpts:
{raw_excerpts}

Question: {sample.question}

Use the MEMORY SUMMARY above as the primary source. Respond in the JSON format specified."""


# ---------------------------------------------------------------------------
# Phase 2.6: Semantic Relabeling + Query-Grounded Fact Selection
# ---------------------------------------------------------------------------

# Domain alias table -- split into EN-only and JA-only for factor decomposition.
# Phase 2.7: test alias language as independent factor.

_DOMAIN_ALIASES_EN: dict[str, list[str]] = {
    "sla target percentage": [
        "numeric_%", "availability", "service level", "uptime",
    ],
    "maximum allowed downtime": [
        "allowed_downtime", "monthly downtime", "downtime", "outage",
    ],
    "measurement method": [
        "measurement", "measurement definition", "uptime definition",
    ],
    "monthly price per user": [
        "numeric_price", "unit price", "monthly fee", "price",
    ],
    "penalty rate": [
        "penalty", "service credit", "credit rate", "late fee",
    ],
    "payment terms": [
        "payment", "net days", "payment period", "invoice",
    ],
    "discount rate": ["discount", "early payment"],
    "approval authority": [
        "approval", "approver", "authorization", "sign-off",
    ],
    "approval threshold": [
        "threshold", "amount limit", "purchase limit",
    ],
    "password minimum length": [
        "password", "minimum length", "character requirement",
    ],
    "annual leave days": [
        "leave", "vacation", "annual leave", "paid leave", "days off",
    ],
    "employee tenure": ["tenure", "years of service", "seniority"],
    "api endpoint status": [
        "api", "version", "endpoint", "api version", "deprecated",
    ],
    "admin limit": ["admin", "administrator", "manager limit"],
    "plan limit": ["plan", "tier", "subscription"],
    "effective date": ["effective", "start date", "enforcement date"],
    "vulnerability count": ["vulnerability", "critical", "finding"],
    "connection limit": ["connection", "concurrent", "capacity"],
    "count": ["numeric_count", "number of", "total"],
    "percentage": ["numeric_%", "rate", "ratio"],
}

_DOMAIN_ALIASES_JA: dict[str, list[str]] = {
    "sla target percentage": [
        "稼働率", "SLA", "サービスレベル", "稼働率保証", "可用性",
    ],
    "maximum allowed downtime": [
        "許容ダウンタイム", "ダウンタイム", "停止時間",
    ],
    "measurement method": ["計測方法", "測定方法", "算定方法"],
    "monthly price per user": [
        "月額", "単価", "ユーザー単価", "料金", "価格",
    ],
    "penalty rate": [
        "ペナルティ", "サービスクレジット", "クレジット", "遅延損害金", "違約金",
    ],
    "payment terms": ["支払条件", "支払期限", "請求", "NET"],
    "discount rate": ["割引", "早期支払"],
    "approval authority": ["承認", "承認者", "承認権限", "決裁"],
    "approval threshold": ["購買金額", "閾値", "上限金額"],
    "password minimum length": [
        "パスワード", "最低文字数", "文字数", "パスワード要件",
    ],
    "annual leave days": [
        "有給休暇", "年次有給", "休暇", "付与日数", "日数",
    ],
    "employee tenure": ["勤続", "勤続年数", "在籍年数"],
    "api endpoint status": [
        "エンドポイント", "API", "非推奨", "廃止", "利用", "バージョン",
    ],
    "admin limit": ["管理者", "ユーザー数", "上限"],
    "plan limit": ["プラン", "フリープラン", "制限"],
    "effective date": ["発効日", "適用開始", "施行日", "変更日"],
    "vulnerability count": ["脆弱性", "未是正", "Critical", "監査"],
    "connection limit": ["同時接続", "接続上限", "接続数"],
    "count": ["件数", "件"],
    "percentage": ["パーセント"],
}

# Valid alias modes for factor decomposition
ALIAS_MODES = ("none", "en", "ja", "en_ja")


def _build_alias_reverse(alias_mode: str = "en_ja") -> dict[str, list[str]]:
    """Build flattened reverse index based on alias language mode.

    alias_mode:
      "none"  -- no aliases (empty reverse index)
      "en"    -- English aliases only
      "ja"    -- Japanese aliases only
      "en_ja" -- both English and Japanese aliases (default, Phase 2.6.1 behavior)
    """
    if alias_mode == "none":
        return {}

    combined: dict[str, list[str]] = {}
    sources = []
    if alias_mode in ("en", "en_ja"):
        sources.append(_DOMAIN_ALIASES_EN)
    if alias_mode in ("ja", "en_ja"):
        sources.append(_DOMAIN_ALIASES_JA)

    for src_table in sources:
        for label, aliases in src_table.items():
            for alias in aliases:
                combined.setdefault(alias.lower(), []).append(label)
            combined.setdefault(label.lower(), []).append(label)

    return combined


# Default reverse index (en_ja) for backward compatibility
_ALIAS_REVERSE: dict[str, list[str]] = _build_alias_reverse("en_ja")


def _tokenize_query(query: str) -> set[str]:
    """Extract lowercased tokens from query, handling both English and Japanese.

    English: split on whitespace/punctuation, keep 2+ char tokens.
    Japanese: extract CJK runs then generate 2-4 char n-grams for matching.
    """
    query_lower = query.lower()

    # English / ASCII tokens (2+ chars)
    tokens = set(re.findall(r"[a-z0-9]{2,}", query_lower))

    # Katakana runs (2+ chars, full-width + half-width)
    tokens.update(re.findall(r"[\u30A0-\u30FF\uFF65-\uFF9F]{2,}", query))

    # CJK / Hiragana runs -- split into overlapping n-grams (2-4 chars)
    cjk_runs = re.findall(r"[\u3040-\u309F\u4E00-\u9FFF\uF900-\uFAFF]{2,}", query)
    for run in cjk_runs:
        for n in (2, 3, 4):
            for i in range(len(run) - n + 1):
                tokens.add(run[i:i + n])
        # Also add the full run itself
        if len(run) >= 2:
            tokens.add(run)

    return tokens


def _infer_query_domain(
    query: str,
    alias_reverse: dict[str, list[str]] | None = None,
) -> list[str]:
    """Infer likely domain labels from query text via alias matching.

    Returns a ranked list of domain labels (best match first).
    Handles both English (token overlap) and Japanese (substring match).
    """
    if alias_reverse is None:
        alias_reverse = _ALIAS_REVERSE
    if not alias_reverse:
        return []

    query_tokens = _tokenize_query(query)
    query_lower = query.lower()

    label_scores: dict[str, float] = {}
    for alias_key, labels in alias_reverse.items():
        score = 0.0

        # Substring match (works for both languages)
        if alias_key in query_lower:
            score += 1.5

        # Token overlap (primarily for English)
        alias_tokens = set(alias_key.split())
        token_overlap = len(alias_tokens & query_tokens)
        if token_overlap > 0:
            score += token_overlap * 1.0

        # N-gram overlap (for Japanese aliases matched against query n-grams)
        if not score and len(alias_key) >= 2:
            for token in query_tokens:
                if alias_key in token or token in alias_key:
                    score += 0.8
                    break

        if score > 0:
            for label in labels:
                label_scores[label] = max(label_scores.get(label, 0.0), score)

    return sorted(label_scores, key=lambda k: -label_scores[k])


def _semantic_relabel_fact(
    fact_key: str,
    fact_value: str,
    query: str,
    query_domains: list[str],
    alias_reverse: dict[str, list[str]] | None = None,
    relabel_enabled: bool = True,
) -> str:
    """Relabel an opaque fact key to a query-aligned human-readable label.

    Falls back to the original key if no alias match is found.
    If relabel_enabled=False, return cleaned-up original key without alias lookup.
    """
    key_lower = fact_key.lower().strip()

    if not relabel_enabled:
        # Just clean up the key, no semantic relabeling
        cleaned = fact_key.replace("_", " ").strip()
        if cleaned.startswith("numeric "):
            cleaned = cleaned[8:]
        return cleaned if cleaned else fact_key

    if alias_reverse is None:
        alias_reverse = _ALIAS_REVERSE

    # Direct alias lookup: does fact key match any alias?
    candidate_labels: list[str] = []
    for alias_key, labels in alias_reverse.items():
        if alias_key in key_lower or key_lower in alias_key:
            candidate_labels.extend(labels)

    if not candidate_labels:
        # No alias match -- return cleaned-up original key
        cleaned = fact_key.replace("_", " ").strip()
        if cleaned.startswith("numeric "):
            cleaned = cleaned[8:]  # remove "numeric " prefix
        return cleaned if cleaned else fact_key

    # Pick the label that best matches query domains
    for domain in query_domains:
        if domain in candidate_labels:
            return domain

    # Fallback: most common label among candidates
    return candidate_labels[0]


# ---------------------------------------------------------------------------
# Phase 2.10: Memory Compiler -- context-aware packet assembly
# ---------------------------------------------------------------------------

# Semantic type inference patterns
_UNIT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("currency", re.compile(r"[¥$€]\s*[\d,]+|[\d,]+\s*円", re.IGNORECASE)),
    ("percentage", re.compile(r"[\d.]+\s*%")),
    ("count", re.compile(r"\d+\s*(?:件|名|個|台|本|回|人|接続|ユーザー)", re.IGNORECASE)),
    ("duration", re.compile(r"\d+\s*(?:日|ヶ月|年|時間|分|秒|hours?|days?|months?)", re.IGNORECASE)),
    ("size", re.compile(r"\d+\s*(?:文字|bytes?|KB|MB|GB|TB)", re.IGNORECASE)),
    ("version", re.compile(r"v?[\d.]+\s*(?:版|version)", re.IGNORECASE)),
]

_QUERY_UNIT_HINTS: dict[str, list[str]] = {
    "currency": ["料金", "費用", "単価", "価格", "コスト", "cost", "price", "fee", "円", "¥"],
    "count": ["接続", "件数", "人数", "名", "上限", "制限", "limit", "count", "connection"],
    "percentage": ["稼働率", "SLA", "率", "percent", "%"],
    "duration": ["期限", "日数", "有給", "休暇", "leave", "days", "期間", "NET"],
    "size": ["文字数", "パスワード", "length", "password", "文字"],
}


def _infer_fact_semantic_type(key: str, value: str) -> str:
    """Infer the semantic type of a fact value (currency, count, etc.)."""
    combined = f"{key} {value}"
    for unit_type, pattern in _UNIT_PATTERNS:
        if pattern.search(combined):
            return unit_type
    return "unknown"


def _infer_query_unit_hint(query: str) -> str:
    """Infer what unit type the query is asking about."""
    q_lower = query.lower()
    for unit_type, keywords in _QUERY_UNIT_HINTS.items():
        for kw in keywords:
            if kw.lower() in q_lower:
                return unit_type
    return "unknown"


def _score_fact_query_compatibility(
    fact_key: str,
    fact_value: str,
    query: str,
    query_unit: str,
    canonical_slots: list[dict[str, str]] | None = None,
) -> float:
    """Score semantic compatibility between a fact and the query.

    Returns a bonus/penalty in [-0.5, +0.5]:
    - slot_match_bonus: +0.3 if fact matches a canonical slot relevant to query
    - unit_match_bonus: +0.2 if fact unit type matches query unit type
    - unit_mismatch_penalty: -0.4 if fact unit type conflicts with query unit type
    - ambiguous_numeric_penalty: -0.2 if fact is numeric-only with no context
    """
    bonus = 0.0
    fact_type = _infer_fact_semantic_type(fact_key, fact_value)

    # Slot match bonus: canonical slot extraction matched this fact
    if canonical_slots:
        for slot in canonical_slots:
            slot_val = slot.get("value", "")
            if slot_val and slot_val in fact_value:
                # Check if slot type is compatible with query
                slot_name = slot.get("slot", "")
                slot_unit = _slot_to_unit(slot_name)
                if slot_unit == query_unit:
                    bonus += 0.3
                    break

    # Unit match / mismatch
    if fact_type != "unknown" and query_unit != "unknown":
        if fact_type == query_unit:
            bonus += 0.2
        else:
            # Strong penalty for clear type mismatch
            # e.g., currency fact when query asks for count
            bonus -= 0.4

    # Ambiguous numeric-only key penalty (e.g., "numeric_%")
    if fact_key.startswith("numeric_") or fact_key.strip().replace(",", "").isdigit():
        bonus -= 0.2

    return max(-0.5, min(0.5, bonus))


def _slot_to_unit(slot_name: str) -> str:
    """Map canonical slot name to unit type."""
    mapping = {
        "sla_target_percentage": "percentage",
        "password_minimum_length": "size",
        "monthly_price_per_user": "currency",
        "payment_terms_days": "duration",
        "annual_leave_days": "duration",
        "vulnerability_count": "count",
        "connection_limit": "count",
        "admin_user_limit": "count",
    }
    return mapping.get(slot_name, "unknown")


def _detect_table_row_family(facts: list[Any]) -> dict[str, list[int]]:
    """Detect groups of facts that form table rows (same source, sequential keys).

    A "family" is a group of facts from the same source_doc_id + source_span_id
    where keys follow a pattern (e.g., time periods like 6mo, 1.5yr, 2.5yr).

    Excludes canonical_slot synthetic facts (these are not table rows).

    Returns {family_id: [indices into facts list]}.
    """
    families: dict[str, list[int]] = {}
    for i, fact in enumerate(facts):
        doc_id = getattr(fact, "source_doc_id", "") or ""
        # Skip canonical slot synthetic facts -- they are not table rows
        if doc_id == "canonical_slot":
            continue
        span_id = getattr(fact, "source_span_id", "") or ""
        family_key = f"{doc_id}:{span_id}"
        families.setdefault(family_key, []).append(i)

    # Only return families with 5+ members (conservative: avoid deduping
    # small tables where every row may be important)
    return {k: v for k, v in families.items() if len(v) >= 5}


def _dedup_table_facts(
    facts: list[Any],
    query: str,
    max_per_family: int = 2,
) -> list[Any]:
    """Compress table row families, keeping only the most query-relevant rows.

    For families with 3+ members, keep at most max_per_family rows
    plus a summary hint. Non-family facts pass through unchanged.
    """
    families = _detect_table_row_family(facts)
    if not families:
        return list(facts)

    # Collect indices that belong to large families
    family_indices: set[int] = set()
    for indices in families.values():
        family_indices.update(indices)

    # Score each family member by query relevance
    q_lower = query.lower()
    q_tokens = set(re.findall(r"\w+", q_lower))

    result: list[Any] = []
    kept_family_indices: set[int] = set()

    for family_id, indices in families.items():
        # Score each row in the family
        scored: list[tuple[int, float]] = []
        for idx in indices:
            fact = facts[idx]
            key_lower = (getattr(fact, "key", "") or "").lower()
            val_lower = (getattr(fact, "value", "") or "").lower()
            combined = f"{key_lower} {val_lower}"
            c_tokens = set(re.findall(r"\w+", combined))
            overlap = len(q_tokens & c_tokens)
            scored.append((idx, overlap))

        # Sort by relevance, keep top max_per_family
        scored.sort(key=lambda x: -x[1])
        for idx, _ in scored[:max_per_family]:
            kept_family_indices.add(idx)

    # Build result: non-family facts + kept family members
    for i, fact in enumerate(facts):
        if i not in family_indices:
            result.append(fact)
        elif i in kept_family_indices:
            result.append(fact)

    return result


def _inject_canonical_slot_facts(
    packet: Any,
    canonical_slots: list[dict[str, str]],
    query: str,
) -> int:
    """Inject canonical slot values as synthetic MemoryFact objects.

    Only injects slots whose value is NOT already present in existing facts.
    Returns the number of injected facts.
    """
    from trimemory.memory_packet import MemoryFact

    existing_values: set[str] = set()
    for fact in packet.exact_facts:
        val = (getattr(fact, "value", "") or "").replace(",", "")
        existing_values.add(val)
        # Also add numeric portions
        for num in re.findall(r"\d+(?:\.\d+)?", val):
            existing_values.add(num)

    injected = 0
    seen_slot_values: set[str] = set()  # avoid duplicate slot injections

    for slot in canonical_slots:
        slot_name = slot.get("slot", "")
        slot_value = slot.get("value", "")
        slot_status = slot.get("status", "unknown")
        slot_context = slot.get("context", "")
        slot_confidence = slot.get("confidence", "medium")

        if not slot_value:
            continue

        # Deduplicate: same slot+value already injected
        dedup_key = f"{slot_name}:{slot_value}"
        if dedup_key in seen_slot_values:
            continue
        seen_slot_values.add(dedup_key)

        # Check if this value is already in existing facts
        # (skip if exact value already present in a fact with matching context)
        value_present = slot_value in existing_values
        if value_present:
            continue

        # Create synthetic fact with high priority for query-relevant slots
        query_unit = _infer_query_unit_hint(query)
        slot_unit = _slot_to_unit(slot_name)
        priority = 4.0 if slot_unit == query_unit else 2.5

        human_key = slot_name.replace("_", " ")
        synthetic = MemoryFact(
            key=human_key,
            value=f"{slot_value} ({slot_context[:60]})" if slot_context else slot_value,
            confidence=0.9 if slot_confidence == "high" else 0.7,
            source_doc_id="canonical_slot",
            source_span_id=slot_name,
            source_title=f"Canonical: {slot_name}",
            status=slot_status,
            provenance="extraction",
            priority_score=priority,
        )

        packet.exact_facts.append(synthetic)
        injected += 1

    return injected


def _compile_memory_packet(
    packet: Any,
    query: str,
    canonical_slots: list[dict[str, str]] | None = None,
    max_facts: int = 12,
    max_per_family: int = 2,
) -> dict[str, Any]:
    """Phase 2.10 Memory Compiler: context-aware packet assembly.

    Pipeline:
    0. Canonical slot injection (synthetic facts from slot extraction)
    1. Table row deduplication (family-aware compression)
    2. Context-aware scoring (unit/type compatibility)
    3. Capacity-aware packing (diversity-preserving selection)

    Returns diagnostics dict and modifies packet.exact_facts in place.
    """
    original_count = len(packet.exact_facts)
    query_unit = _infer_query_unit_hint(query)

    # Step 0: Inject canonical slot values as synthetic facts
    n_injected = 0
    if canonical_slots:
        n_injected = _inject_canonical_slot_facts(packet, canonical_slots, query)

    # Step 1: Table row deduplication
    deduped = _dedup_table_facts(packet.exact_facts, query, max_per_family)
    n_deduped = (original_count + n_injected) - len(deduped)

    # Step 2: Context-aware scoring
    scored: list[tuple[Any, float]] = []
    n_mismatch_penalties = 0
    for fact in deduped:
        key = getattr(fact, "key", "") or ""
        value = getattr(fact, "value", "") or ""
        base_score = getattr(fact, "priority_score", 0.0) or 0.0

        compat = _score_fact_query_compatibility(
            key, value, query, query_unit, canonical_slots,
        )
        if compat < -0.1:
            n_mismatch_penalties += 1

        final_score = base_score + compat
        scored.append((fact, final_score))

    # Step 3: Conditional re-sorting -- only re-sort when a SINGLE
    # high-relevance canonical slot was injected and query_unit is known.
    # Multiple competing slots = ambiguous, preserve Mediator order.
    # Unknown query_unit = no strong signal, preserve Mediator order.
    should_resort = (
        n_injected == 1
        and query_unit != "unknown"
    )

    if should_resort:
        # Targeted mode: re-sort to promote the single canonical slot
        scored.sort(key=lambda x: -x[1])
    # else: preserve original Mediator order (deduped but not re-ranked)

    # Capacity-aware packing with family diversity cap
    # Only apply family cap when slot injection triggers re-sorting,
    # otherwise pass through all facts unchanged (preserve Mediator order).
    family_count: dict[str, int] = {}
    packed: list[Any] = []
    max_family_in_packet = 3 if should_resort else max_facts  # no cap when not re-sorting

    for fact, score in scored:
        if len(packed) >= max_facts:
            break
        doc_id = getattr(fact, "source_doc_id", "") or ""
        span_id = getattr(fact, "source_span_id", "") or ""
        fam_key = f"{doc_id}:{span_id}"
        current = family_count.get(fam_key, 0)
        # Canonical slot synthetic facts bypass family cap
        if doc_id != "canonical_slot" and current >= max_family_in_packet:
            continue
        packed.append(fact)
        family_count[fam_key] = current + 1

    # Update packet
    packet.exact_facts = packed

    # Build diagnostics
    diagnostics = {
        "original_facts": original_count,
        "n_slot_injected": n_injected,
        "after_dedup": len(deduped),
        "n_deduped_rows": n_deduped,
        "n_mismatch_penalties": n_mismatch_penalties,
        "query_unit_hint": query_unit,
        "final_facts": len(packed),
        "family_occupancy": dict(family_count),
    }

    return diagnostics


def _score_fact_relevance(
    fact: Any,
    query: str,
    query_tokens: set[str],
    query_domains: list[str],
    alias_reverse: dict[str, list[str]] | None = None,
) -> float:
    """Score a single fact's relevance to the query.

    Components:
    - label_query_overlap: overlap between relabeled key and query tokens
    - alias_match: whether fact key maps to a query domain alias
    - current_bonus: +0.3 for current/final status
    - formal_bonus: +0.2 for spec provenance
    - superseded_penalty: -0.3 for superseded status
    - draft_penalty: -0.2 for draft/provisional status
    """
    if alias_reverse is None:
        alias_reverse = _ALIAS_REVERSE

    fact_key_lower = (fact.key or "").lower()
    fact_status = (fact.status or "unknown").lower()
    fact_prov = (fact.provenance or "unknown").lower()

    # Label overlap: tokenize fact key+value same way as query
    combined_fact_text = f"{fact_key_lower} {(fact.value or '').lower()}"
    fact_tokens = _tokenize_query(combined_fact_text)
    label_overlap = len(query_tokens & fact_tokens) / max(len(query_tokens), 1)

    # Alias match: does fact key map to any query domain?
    alias_score = 0.0
    for alias_key, labels in alias_reverse.items():
        if alias_key in fact_key_lower or fact_key_lower in alias_key:
            for label in labels:
                if label in query_domains:
                    alias_score = 1.0
                    break
            if alias_score > 0:
                break

    # Status bonuses/penalties
    current_bonus = 0.3 if fact_status in ("current", "final") else 0.0
    formal_bonus = 0.2 if fact_prov == "spec" else 0.0
    superseded_penalty = 0.3 if fact_status == "superseded" else 0.0
    draft_penalty = 0.2 if fact_status in ("draft", "provisional") else 0.0

    return (
        0.3 * label_overlap
        + 0.3 * alias_score
        + current_bonus
        + formal_bonus
        - superseded_penalty
        - draft_penalty
    )


def _select_primary_facts(
    packet: Any,
    query: str,
    max_primary: int = 4,
    max_secondary: int = 3,
    alias_reverse: dict[str, list[str]] | None = None,
    relabel_enabled: bool = True,
) -> tuple[list[tuple[Any, float, str]], list[tuple[Any, float, str]]]:
    """Select and rank facts by query relevance.

    Returns (primary_facts, secondary_facts) where each element is
    (fact, relevance_score, relabeled_key).
    """
    if alias_reverse is None:
        alias_reverse = _ALIAS_REVERSE

    query_tokens = _tokenize_query(query)
    query_domains = _infer_query_domain(query, alias_reverse)

    scored: list[tuple[Any, float, str]] = []
    for fact in packet.exact_facts:
        relevance = _score_fact_relevance(
            fact, query, query_tokens, query_domains, alias_reverse,
        )
        relabeled = _semantic_relabel_fact(
            fact.key, fact.value, query, query_domains,
            alias_reverse, relabel_enabled,
        )
        scored.append((fact, relevance, relabeled))

    # Sort by relevance descending, then by priority_score descending
    scored.sort(key=lambda x: (-x[1], -x[0].priority_score))

    primary = scored[:max_primary]
    secondary = scored[max_primary:max_primary + max_secondary]
    return primary, secondary


def _build_answer_candidate(
    primary_facts: list[tuple[Any, float, str]],
    packet: Any,
) -> str | None:
    """Build an explicit answer candidate line, or None if uncertain.

    Rules:
    - Show candidate only if a current/formal fact exists with sufficient relevance
    - Return "uncertain" if only draft/provisional or conflicting values
    - Return None if no relevant facts at all
    """
    if not primary_facts:
        return None

    top_fact, top_relevance, top_label = primary_facts[0]
    top_status = (top_fact.status or "unknown").lower()

    # Check for strong conflicts
    has_conflicts = packet.has_conflicts()
    conflict_on_top_key = False
    for h in packet.state_hints:
        if h.hint_type in ("conflict", "unresolved"):
            if top_fact.key.lower() in h.text.lower():
                conflict_on_top_key = True
                break

    # If no current/formal fact in primary, uncertain
    has_current_primary = any(
        (f.status or "").lower() in ("current", "final")
        for f, _, _ in primary_facts
    )

    if not has_current_primary:
        return "Most likely current answer: uncertain (no formally approved current value found)"

    if conflict_on_top_key:
        return (
            f"Most likely current answer: uncertain "
            f"(conflicting values detected for {top_label})"
        )

    if top_status not in ("current", "final"):
        return (
            f"Most likely current answer: uncertain "
            f"({top_label} = {top_fact.value}, but status is {top_status})"
        )

    if top_relevance < 0.2:
        return None  # Not confident enough to suggest

    return f"Most likely current answer: {top_label} = {top_fact.value}"


def render_packet_semantic(
    packet: Any,
    query: str,
    include_answer_slot: bool = False,
    alias_mode: str = "en_ja",
    relabel_enabled: bool = True,
) -> str:
    """Render packet with semantic relabeling and query-grounded fact selection.

    Phase 2.6/2.7 renderer. Produces:
    1. Question focus (if aliases available)
    2. Most likely current answer (conditional)
    3. Current formal facts (relabeled or opaque, relevance-ordered)
    4. Older / superseded / draft facts
    5. Conflicts detected
    6. Reasoning guidance
    7. Source references

    alias_mode: "none" | "en" | "ja" | "en_ja"
    relabel_enabled: if False, keep opaque keys but still score/rank facts
    """
    alias_reverse = _build_alias_reverse(alias_mode)
    primary_facts, secondary_facts = _select_primary_facts(
        packet, query, alias_reverse=alias_reverse,
        relabel_enabled=relabel_enabled,
    )

    sections: list[str] = ["=== MEMORY SUMMARY ==="]

    # 1. Question focus
    query_domains = _infer_query_domain(query, alias_reverse)
    if query_domains:
        sections.append(f"\nQuestion focus: {', '.join(query_domains[:3])}")

    # 2. Answer candidate (conditional)
    if include_answer_slot:
        candidate = _build_answer_candidate(primary_facts, packet)
        if candidate:
            sections.append(f"\n{candidate}")

    # 3. Current formal facts (primary, relabeled)
    current_lines: list[str] = []
    superseded_lines: list[str] = []
    provisional_lines: list[str] = []

    for fact, relevance, relabeled in primary_facts:
        status = (fact.status or "unknown").lower()
        src = f"source: {fact.source_doc_id}"
        if fact.source_title:
            src += f", {fact.source_title[:40]}"

        line = f"- {relabeled} = {fact.value} ({src}, {status})."

        if status in ("current", "final"):
            current_lines.append(line)
        elif status == "superseded":
            superseded_lines.append(line)
        elif status in ("draft", "provisional"):
            provisional_lines.append(line)
        else:
            current_lines.append(line)

    # Add secondary facts as lower-priority
    for fact, relevance, relabeled in secondary_facts:
        status = (fact.status or "unknown").lower()
        src = f"source: {fact.source_doc_id}"
        line = f"- {relabeled} = {fact.value} ({src}, {status})."
        if status in ("current", "final"):
            current_lines.append(line)
        elif status == "superseded":
            superseded_lines.append(line)
        else:
            superseded_lines.append(line)

    if current_lines:
        sections.append("\nCurrent formal facts:")
        sections.extend(current_lines)
    else:
        sections.append(
            "\nCurrent formal facts:\n"
            "- No formally approved current facts found in the documents."
        )

    if superseded_lines:
        sections.append("\nOlder or superseded facts:")
        sections.extend(superseded_lines)

    if provisional_lines:
        sections.append("\nPending or provisional updates:")
        sections.extend(provisional_lines)
        sections.append(
            "Note: Do not treat these as current unless explicitly approved."
        )

    # 5. Conflicts
    conflict_lines: list[str] = []
    change_lines: list[str] = []
    for h in packet.state_hints:
        ht = (h.hint_type or "").lower()
        if ht in ("conflict", "unresolved"):
            conflict_lines.append(f"- {h.text}")
        else:
            change_lines.append(f"- {h.text}")

    if conflict_lines:
        sections.append("\nConflicts detected:")
        sections.extend(conflict_lines)
        sections.append(
            "Guidance: If the conflict affects the answer, "
            "report status as 'contradictory' and set needs_escalation=true."
        )

    # 6. Reasoning guidance
    if change_lines:
        sections.append("\nReason / change context:")
        sections.extend(change_lines)

    if packet.anomaly_flags:
        sections.append(f"\nAnomalies: {', '.join(packet.anomaly_flags)}")

    # 7. Source references
    source_lines: list[str] = []
    for r in packet.source_refs:
        status_tag = f", {r.status}" if r.status else ""
        source_lines.append(f"- {r.doc_id}: {r.title}{status_tag}")
    if source_lines:
        # Deduplicate
        seen: set[str] = set()
        unique_lines: list[str] = []
        for line in source_lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        sections.append("\nSource references:")
        sections.extend(unique_lines)

    return "\n".join(sections)


def _format_short_source_refs(
    documents: list[dict], packet: Any,
) -> str:
    """Generate 1-2 line short excerpts per source doc referenced in the packet."""
    ref_doc_ids = {r.doc_id for r in packet.source_refs}
    # Also include docs referenced in facts
    for f in packet.exact_facts:
        ref_doc_ids.add(f.source_doc_id)

    lines: list[str] = []
    for doc in documents:
        doc_id = doc.get("doc_id", "")
        if doc_id not in ref_doc_ids:
            continue
        meta = doc.get("metadata", {})
        content = doc.get("content", "")
        # Take first 150 chars as short excerpt
        excerpt = content[:150].replace("\n", " ").strip()
        if len(content) > 150:
            excerpt += "..."
        lines.append(
            f"- [{doc_id}] {doc.get('title', '?')[:50]} "
            f"(status={meta.get('status', '?')}): {excerpt}"
        )
    return "\n".join(lines) if lines else "No source excerpts available."


def _build_prompt_trimemory_fallback(sample: EvalSample) -> str:
    """Fallback when TriMemory imports fail."""
    current_facts = []
    other_facts = []
    for doc in sample.documents:
        meta = doc.get("metadata", {})
        status = meta.get("status", "unknown")
        doc_id = doc.get("doc_id", "")
        title = doc.get("title", "")
        content_preview = doc.get("content", "")[:200]
        entry = (
            f"[{doc_id}] {title} (status={status}, "
            f"date={meta.get('date', '?')}): {content_preview}"
        )
        if status in ("current", "final"):
            current_facts.append(entry)
        else:
            other_facts.append(entry)

    sections = ["## Memory Packet (fallback mode)"]
    if current_facts:
        sections.append("### Current/Final Sources")
        sections.extend(current_facts)
    if other_facts:
        sections.append("### Other Sources")
        sections.extend(other_facts)

    return f"""{chr(10).join(sections)}

Question: {sample.question}

Respond in the JSON format specified."""


def _format_packet_for_prompt_json(packet: Any) -> str:
    """Legacy: Format CompactMemoryPacket as structured text (Phase 2 style)."""
    lines = []

    if packet.exact_facts:
        lines.append("## Extracted Facts (priority-ordered)")
        for f in packet.exact_facts:
            lines.append(
                f"- {f.key}: {f.value} "
                f"[source={f.source_doc_id}, status={f.status}, "
                f"prov={f.provenance}, score={f.priority_score:.2f}]"
            )

    if packet.state_hints:
        lines.append("\n## State Hints")
        for h in packet.state_hints:
            lines.append(
                f"- [{h.hint_type}] {h.text} (confidence={h.confidence:.2f})"
            )

    if packet.anomaly_flags:
        lines.append(f"\n## Anomalies: {', '.join(packet.anomaly_flags)}")

    if packet.provenance_summary:
        lines.append("\n## Provenance")
        for p in packet.provenance_summary:
            lines.append(f"- {p}")

    if packet.source_refs:
        lines.append("\n## Sources")
        for r in packet.source_refs:
            lines.append(
                f"- {r.doc_id}: {r.title} (status={r.status}, prov={r.provenance})"
            )

    if packet.packet_summary:
        lines.append(f"\n## Summary: {packet.packet_summary}")

    return "\n".join(lines) if lines else "No structured data extracted."


# ---------------------------------------------------------------------------
# Phase 2.5: Natural Language Packet Renderer
# ---------------------------------------------------------------------------

_STATUS_ORDER = {"current": 0, "final": 1, "draft": 2, "provisional": 3, "superseded": 4, "unknown": 5}


def render_packet_for_llm(packet: Any) -> str:
    """Render CompactMemoryPacket as natural language for LLM consumption.

    Produces an English block with explicit sections:
    - Current formal facts
    - Older or superseded facts
    - Pending / provisional updates
    - Conflicts detected
    - Reason / change history
    - Source references
    """
    current_facts: list[str] = []
    superseded_facts: list[str] = []
    provisional_facts: list[str] = []

    for f in packet.exact_facts:
        status = (f.status or "unknown").lower()
        source_info = f"source: {f.source_doc_id}"
        if f.source_title:
            source_info += f", {f.source_title}"
        prov = f.provenance or "unknown"

        fact_line = f"- {f.key} = {f.value} ({source_info}, {prov}, {status})."

        if status in ("current", "final"):
            current_facts.append(fact_line)
        elif status in ("superseded",):
            superseded_facts.append(fact_line)
        elif status in ("draft", "provisional"):
            provisional_facts.append(fact_line)
        else:
            # Unknown status -- treat as current if high priority
            if f.priority_score >= 2.0:
                current_facts.append(fact_line)
            else:
                superseded_facts.append(fact_line)

    # Build conflict section from state hints
    conflict_lines: list[str] = []
    change_lines: list[str] = []
    for h in packet.state_hints:
        ht = (h.hint_type or "").lower()
        if ht in ("conflict", "unresolved"):
            conflict_lines.append(f"- {h.text}")
        elif ht in ("pending_change", "trend"):
            change_lines.append(f"- {h.text}")
        else:
            change_lines.append(f"- {h.text}")

    # Build source reference section
    source_lines: list[str] = []
    for r in packet.source_refs:
        status_tag = f", {r.status}" if r.status else ""
        source_lines.append(f"- {r.doc_id}: {r.title}{status_tag}")

    # Assemble output
    sections: list[str] = ["=== MEMORY SUMMARY ==="]

    if current_facts:
        sections.append("\nCurrent formal facts:")
        sections.extend(current_facts)
    else:
        sections.append("\nCurrent formal facts:\n- No formally approved current facts found in the documents.")

    if superseded_facts:
        sections.append("\nOlder or superseded facts:")
        sections.extend(superseded_facts)

    if provisional_facts:
        sections.append("\nPending or provisional updates:")
        sections.extend(provisional_facts)
        sections.append("Note: Do not treat these as current unless explicitly approved.")

    if conflict_lines:
        sections.append("\nConflicts detected:")
        sections.extend(conflict_lines)
        sections.append("Guidance: If no formally approved current value exists, answer 'uncertain'.")

    if change_lines:
        sections.append("\nReason / change context:")
        sections.extend(change_lines)

    if packet.anomaly_flags:
        sections.append(f"\nAnomalies: {', '.join(packet.anomaly_flags)}")

    if source_lines:
        sections.append("\nSource references:")
        sections.extend(source_lines)

    return "\n".join(sections)


def _format_doc_context(documents: list[dict]) -> str:
    """Brief doc-level context: role + status + date per document."""
    lines = []
    for doc in documents:
        meta = doc.get("metadata", {})
        lines.append(
            f"- {doc.get('doc_id', '?')}: {doc.get('title', '?')[:60]} "
            f"(role={doc.get('role', '?')}, status={meta.get('status', '?')}, "
            f"date={meta.get('date', '?')})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 4: Output Parsing
# ---------------------------------------------------------------------------

@dataclass
class ParsedResponse:
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


def parse_response(raw: str) -> ParsedResponse:
    """Parse model response into structured format."""
    result = ParsedResponse(raw_text=raw)
    text = raw.strip()

    # Extract JSON block if wrapped in markdown
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    # Try to find JSON object in text
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
            result.confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            result.confidence = 0.0
        result.parse_success = True
        return result
    except json.JSONDecodeError:
        pass

    # Fallback: extract what we can from raw text
    result.answer = raw[:500]
    doc_refs = re.findall(r"doc_\d{2}", raw)
    result.evidence_docs = list(dict.fromkeys(doc_refs))

    if any(kw in raw for kw in ["undetermined", "uncertain"]):
        result.status_judgment = "undetermined"
    elif any(kw in raw for kw in ["contradictory", "conflict"]):
        result.status_judgment = "contradictory"
    elif any(kw in raw for kw in ["confirmed", "current"]):
        result.status_judgment = "confirmed"

    return result


# ---------------------------------------------------------------------------
# Task 5: Field-Based Scoring (Phase 2)
# ---------------------------------------------------------------------------

@dataclass
class FieldScore:
    """Per-field scoring breakdown."""
    current_value_match: float = 0.0
    previous_value_match: float = 0.0
    reason_match: float = 0.0
    status_judgment_match: float = 0.0
    needs_escalation_match: float = 0.0
    evidence_doc_recall: float = 0.0
    evidence_doc_precision: float = 0.0
    stale_fact_error: bool = False
    latest_only_error: bool = False
    unsupported_definitive_error: bool = False
    uncertainty_handling: float = 0.0
    composite_score: float = 0.0


def _fuzzy_value_match(gold: str, response: str) -> float:
    """Fuzzy match between gold and response values.

    Returns 1.0 for exact match, 0.5 for substring containment,
    0.3 for partial keyword overlap, 0.0 otherwise.
    """
    if not gold:
        return 1.0  # no gold value to match
    gold_lower = gold.lower().strip()
    resp_lower = response.lower().strip()

    if not resp_lower:
        return 0.0

    # Exact match
    if gold_lower == resp_lower:
        return 1.0

    # Substring containment (either direction)
    if gold_lower in resp_lower or resp_lower in gold_lower:
        return 0.7

    # Extract numbers for numeric comparison
    gold_nums = re.findall(r"[\d.]+", gold_lower)
    resp_nums = re.findall(r"[\d.]+", resp_lower)
    if gold_nums and resp_nums:
        gold_set = set(gold_nums)
        resp_set = set(resp_nums)
        if gold_set & resp_set:
            return 0.5

    # Keyword overlap (for longer text like reasons)
    gold_tokens = set(re.findall(r"\w{2,}", gold_lower))
    resp_tokens = set(re.findall(r"\w{2,}", resp_lower))
    if gold_tokens and resp_tokens:
        overlap = len(gold_tokens & resp_tokens) / len(gold_tokens)
        if overlap >= 0.5:
            return 0.3

    return 0.0


def score_sample_phase2(
    sample: EvalSample,
    parsed: ParsedResponse,
    condition: str,
) -> FieldScore:
    """Phase 2: Field-based scoring."""
    fs = FieldScore()
    gas = sample.gold_answer_structured

    # -- current_value_match --
    gold_current = str(gas.get("current_value", ""))
    # Check both dedicated field and answer field
    resp_current = parsed.current_value or parsed.answer
    fs.current_value_match = _fuzzy_value_match(gold_current, resp_current)

    # -- previous_value_match --
    gold_previous = str(gas.get("previous_value", ""))
    resp_previous = parsed.previous_value
    if gold_previous:
        fs.previous_value_match = _fuzzy_value_match(gold_previous, resp_previous)
        # Also check in answer text as fallback
        if fs.previous_value_match < 0.3:
            fs.previous_value_match = max(
                fs.previous_value_match,
                _fuzzy_value_match(gold_previous, parsed.answer) * 0.5,
            )
    else:
        fs.previous_value_match = 1.0  # no previous to check

    # -- reason_match --
    gold_reason = str(gas.get("reason", ""))
    resp_reason = parsed.reason
    if gold_reason:
        fs.reason_match = _fuzzy_value_match(gold_reason, resp_reason)
        if fs.reason_match < 0.3:
            fs.reason_match = max(
                fs.reason_match,
                _fuzzy_value_match(gold_reason, parsed.answer) * 0.5,
            )
    else:
        fs.reason_match = 1.0

    # -- status_judgment_match --
    gold_status = str(gas.get("status_judgment", "")).lower()
    resp_status = parsed.status_judgment.lower()
    status_equiv: dict[str, set[str]] = {
        "confirmed": {"confirmed", "current"},
        "undetermined": {"undetermined", "uncertain", "provisional"},
        "contradictory": {"contradictory", "conflict"},
    }
    gold_set = status_equiv.get(gold_status, {gold_status})
    if resp_status in gold_set or gold_status == resp_status:
        fs.status_judgment_match = 1.0
    elif resp_status and any(g in resp_status for g in gold_set):
        fs.status_judgment_match = 0.5
    else:
        fs.status_judgment_match = 0.0

    # -- needs_escalation_match --
    gold_esc = bool(gas.get("needs_escalation", False))
    if parsed.needs_escalation == gold_esc:
        fs.needs_escalation_match = 1.0
    else:
        fs.needs_escalation_match = 0.0

    # -- evidence_doc_recall / precision --
    gold_doc_ids = {ev.get("doc_id", "") for ev in sample.gold_evidence}
    gold_doc_ids.discard("")
    resp_doc_ids = set(parsed.evidence_docs)
    resp_doc_ids.discard("")
    if gold_doc_ids:
        fs.evidence_doc_recall = len(gold_doc_ids & resp_doc_ids) / len(gold_doc_ids)
    else:
        fs.evidence_doc_recall = 1.0
    if resp_doc_ids:
        fs.evidence_doc_precision = len(gold_doc_ids & resp_doc_ids) / len(resp_doc_ids)
    else:
        fs.evidence_doc_precision = 0.0

    # -- stale_fact_error --
    gold_prev_lower = gold_previous.lower()
    gold_curr_lower = gold_current.lower()
    answer_lower = parsed.answer.lower()
    if (
        gold_prev_lower
        and gold_prev_lower in answer_lower
        and gold_curr_lower
        and gold_curr_lower not in answer_lower
    ):
        fs.stale_fact_error = True

    # Check distractor trap values
    for dist in sample.distractors:
        trap_val = str(dist.get("trap_value", "")).lower()
        if (
            trap_val
            and trap_val in answer_lower
            and gold_curr_lower
            and gold_curr_lower not in answer_lower
        ):
            fs.stale_fact_error = True

    # -- unsupported_definitive_error --
    if gold_esc and not parsed.needs_escalation:
        if resp_status not in ("undetermined", "contradictory", "uncertain", "conflict", ""):
            fs.unsupported_definitive_error = True

    # -- latest_only_error --
    if "INCOMPLETE_JUSTIFICATION" in sample.expected_failure_modes:
        if gold_reason and fs.reason_match < 0.1:
            fs.latest_only_error = True

    # -- uncertainty_handling --
    # Good: model says undetermined when gold says undetermined/contradictory
    if gold_status in ("undetermined", "contradictory"):
        if resp_status in ("undetermined", "uncertain", "contradictory", "conflict"):
            fs.uncertainty_handling = 1.0
        elif parsed.needs_escalation:
            fs.uncertainty_handling = 0.5
        else:
            fs.uncertainty_handling = 0.0
    else:
        # Gold is confirmed -- model should not be uncertain
        if resp_status in ("undetermined", "uncertain"):
            fs.uncertainty_handling = 0.0
        else:
            fs.uncertainty_handling = 1.0

    # -- composite_score --
    # Weighted combination: answer fields 0.35, evidence 0.30, errors 0.35
    answer_score = (
        fs.current_value_match * 0.15
        + fs.previous_value_match * 0.05
        + fs.reason_match * 0.05
        + fs.status_judgment_match * 0.05
        + fs.needs_escalation_match * 0.05
    )
    evidence_score = (
        fs.evidence_doc_recall * 0.15
        + fs.evidence_doc_precision * 0.05
        + fs.uncertainty_handling * 0.10
    )
    error_penalty = (
        (0.0 if fs.stale_fact_error else 0.15)
        + (0.0 if fs.latest_only_error else 0.05)
        + (0.0 if fs.unsupported_definitive_error else 0.15)
    )
    fs.composite_score = answer_score + evidence_score + error_penalty

    return fs


# ---------------------------------------------------------------------------
# Task 6: Runner (multi-model support)
# ---------------------------------------------------------------------------

CONDITIONS = ["plain", "latest", "rag", "trimemory"]


def run_evaluation(
    samples: list[EvalSample],
    model: str,
    out_dir: Path,
    top_k: int = 4,
    conditions: list[str] | None = None,
    temperature: float = 0.0,
    verbose: bool = False,
    phase2: bool = False,
    packet_log: list[dict] | None = None,
    trimemory_render_mode: str = "packet_only",
) -> list[dict]:
    """Run full evaluation across all conditions and samples."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if conditions is None:
        conditions = CONDITIONS

    all_results: list[dict] = []
    suffix = "_phase2" if phase2 else ""
    results_path = out_dir / f"results{suffix}_{model.replace(':', '_')}.jsonl"

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
                        sample, top_k=top_k, packet_log=packet_log,
                        render_mode=trimemory_render_mode,
                    )
                else:
                    continue

                # Call Ollama -- select system prompt per condition
                sys_prompt = (
                    SYSTEM_PROMPT_TRIMEMORY if cond == "trimemory"
                    else SYSTEM_PROMPT_PLAIN
                )
                raw_response = ""
                error_msg = ""
                try:
                    raw_response = run_ollama(
                        prompt=prompt,
                        model=model,
                        system=sys_prompt,
                        temperature=temperature,
                    )
                except Exception as e:
                    error_msg = str(e)
                    logger.error("%s/%s/%s: %s", model, sample.sample_id, cond, e)

                # Parse
                parsed = parse_response(raw_response)

                # Score
                if phase2:
                    score = score_sample_phase2(sample, parsed, cond)
                    score_dict = asdict(score)
                else:
                    score = score_sample_phase2(sample, parsed, cond)
                    score_dict = asdict(score)

                # Build result record
                record = {
                    "model": model,
                    "sample_id": sample.sample_id,
                    "condition": cond,
                    "render_mode": trimemory_render_mode if cond == "trimemory" else "",
                    "type": sample.type,
                    "difficulty": sample.difficulty,
                    "prompt_length": len(prompt),
                    "raw_response": raw_response,
                    "parse_success": parsed.parse_success,
                    "parsed_answer": parsed.answer[:300],
                    "parsed_current_value": parsed.current_value[:200],
                    "parsed_previous_value": parsed.previous_value[:200],
                    "parsed_reason": parsed.reason[:300],
                    "parsed_evidence_docs": parsed.evidence_docs,
                    "parsed_status_judgment": parsed.status_judgment,
                    "parsed_needs_escalation": parsed.needs_escalation,
                    "parsed_confidence": parsed.confidence,
                    "gold_current_value": str(
                        sample.gold_answer_structured.get("current_value", "")
                    ),
                    "gold_previous_value": str(
                        sample.gold_answer_structured.get("previous_value", "")
                    ),
                    "gold_reason": str(
                        sample.gold_answer_structured.get("reason", "")
                    ),
                    "gold_status_judgment": str(
                        sample.gold_answer_structured.get("status_judgment", "")
                    ),
                    "gold_needs_escalation": bool(
                        sample.gold_answer_structured.get("needs_escalation", False)
                    ),
                    "gold_evidence_docs": [
                        e.get("doc_id") for e in sample.gold_evidence
                    ],
                    "expected_failure_modes": sample.expected_failure_modes,
                    "score": score_dict,
                    "error": error_msg,
                }
                all_results.append(record)
                rf.write(json.dumps(record, ensure_ascii=False) + "\n")
                rf.flush()

                if verbose:
                    logger.info(
                        "  composite=%.2f val=%.2f status=%.2f evi_r=%.2f "
                        "stale=%s unsup=%s",
                        score.composite_score,
                        score.current_value_match,
                        score.status_judgment_match,
                        score.evidence_doc_recall,
                        score.stale_fact_error,
                        score.unsupported_definitive_error,
                    )

    return all_results


# ---------------------------------------------------------------------------
# Task 7: Aggregation and Report (Phase 2: multi-model)
# ---------------------------------------------------------------------------

def aggregate_results_phase2(
    results: list[dict],
    out_dir: Path,
) -> None:
    """Generate Phase 2 aggregate report with multi-model comparison."""

    # --- Summary CSV ---
    csv_path = out_dir / "summary_phase2.csv"
    fieldnames = [
        "model", "sample_id", "condition", "type", "difficulty",
        "composite_score", "current_value_match", "previous_value_match",
        "reason_match", "status_judgment_match", "needs_escalation_match",
        "evidence_doc_recall", "evidence_doc_precision",
        "stale_fact_error", "unsupported_definitive_error",
        "uncertainty_handling", "parse_success",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            sc = r["score"]
            writer.writerow({
                "model": r.get("model", "?"),
                "sample_id": r["sample_id"],
                "condition": r["condition"],
                "type": r["type"],
                "difficulty": r["difficulty"],
                "composite_score": f"{sc['composite_score']:.3f}",
                "current_value_match": f"{sc['current_value_match']:.3f}",
                "previous_value_match": f"{sc['previous_value_match']:.3f}",
                "reason_match": f"{sc['reason_match']:.3f}",
                "status_judgment_match": f"{sc['status_judgment_match']:.3f}",
                "needs_escalation_match": f"{sc['needs_escalation_match']:.3f}",
                "evidence_doc_recall": f"{sc['evidence_doc_recall']:.3f}",
                "evidence_doc_precision": f"{sc['evidence_doc_precision']:.3f}",
                "stale_fact_error": sc["stale_fact_error"],
                "unsupported_definitive_error": sc["unsupported_definitive_error"],
                "uncertainty_handling": f"{sc['uncertainty_handling']:.3f}",
                "parse_success": r["parse_success"],
            })

    # --- Error cases ---
    error_path = out_dir / "error_cases_phase2.jsonl"
    with open(error_path, "w", encoding="utf-8") as f:
        for r in results:
            sc = r["score"]
            if (
                sc["stale_fact_error"]
                or sc["unsupported_definitive_error"]
                or not r["parse_success"]
            ):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- Aggregate Report ---
    report_path = out_dir / "aggregate_report_phase2.md"
    lines = _build_phase2_report(results)

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Print summary to stdout (safe for cp932 terminals)
    safe_text = report_text.encode("ascii", errors="replace").decode("ascii")
    print("\n" + "=" * 60)
    print(safe_text)
    print("=" * 60)


def _build_phase2_report(results: list[dict]) -> list[str]:
    """Build Phase 2 markdown report."""
    lines: list[str] = []
    lines.append("# DocBench Phase 2 Evaluation Report\n")

    models = sorted(set(r.get("model", "?") for r in results))
    n_samples = len(set(r["sample_id"] for r in results))
    lines.append(f"Models: {', '.join(models)}")
    lines.append(f"Samples: {n_samples}")
    lines.append(f"Total runs: {len(results)}")
    # Detect render mode from results
    render_modes = set()
    for r in results:
        if r["condition"] == "trimemory":
            render_modes.add(r.get("render_mode", "unknown"))
    render_mode_str = ", ".join(sorted(render_modes)) if render_modes else "N/A"
    lines.append(f"TriMemory packet path: {'YES' if _TRIMEMORY_AVAILABLE else 'FALLBACK'}")
    lines.append(f"TriMemory render mode: {render_mode_str}\n")

    def avg(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    # --- Model x Condition comparison table ---
    lines.append("## Model x Condition Comparison\n")
    lines.append(
        "| Model | Condition | Composite | CurrVal | Status | "
        "EviRecall | Stale | Unsupported | Uncertainty | Parse% |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for model in models:
        for cond in CONDITIONS:
            subset = [
                r for r in results
                if r.get("model") == model and r["condition"] == cond
            ]
            if not subset:
                continue
            n = len(subset)
            sc_list = [r["score"] for r in subset]
            stale_n = sum(1 for s in sc_list if s["stale_fact_error"])
            unsup_n = sum(1 for s in sc_list if s["unsupported_definitive_error"])
            parse_n = sum(1 for r in subset if r["parse_success"])
            lines.append(
                f"| {model} | {cond} | "
                f"{avg([s['composite_score'] for s in sc_list]):.3f} | "
                f"{avg([s['current_value_match'] for s in sc_list]):.3f} | "
                f"{avg([s['status_judgment_match'] for s in sc_list]):.3f} | "
                f"{avg([s['evidence_doc_recall'] for s in sc_list]):.3f} | "
                f"{stale_n}/{n} | {unsup_n}/{n} | "
                f"{avg([s['uncertainty_handling'] for s in sc_list]):.3f} | "
                f"{parse_n*100//n}% |"
            )

    # --- Field-wise score table ---
    lines.append("\n## Field-wise Score Breakdown\n")
    lines.append(
        "| Model | Condition | CurrVal | PrevVal | Reason | "
        "StatusJ | Escalation | EviRecall | EviPrec | Uncertainty |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for model in models:
        for cond in CONDITIONS:
            subset = [
                r["score"] for r in results
                if r.get("model") == model and r["condition"] == cond
            ]
            if not subset:
                continue
            lines.append(
                f"| {model} | {cond} | "
                f"{avg([s['current_value_match'] for s in subset]):.3f} | "
                f"{avg([s['previous_value_match'] for s in subset]):.3f} | "
                f"{avg([s['reason_match'] for s in subset]):.3f} | "
                f"{avg([s['status_judgment_match'] for s in subset]):.3f} | "
                f"{avg([s['needs_escalation_match'] for s in subset]):.3f} | "
                f"{avg([s['evidence_doc_recall'] for s in subset]):.3f} | "
                f"{avg([s['evidence_doc_precision'] for s in subset]):.3f} | "
                f"{avg([s['uncertainty_handling'] for s in subset]):.3f} |"
            )

    # --- Failure mode breakdown ---
    lines.append("\n## Failure Mode Analysis\n")
    fm_data: dict[str, dict[str, list[float]]] = {}
    for r in results:
        key = f"{r.get('model', '?')}|{r['condition']}"
        for fm in r.get("expected_failure_modes", []):
            fm_data.setdefault(fm, {}).setdefault(key, []).append(
                r["score"]["composite_score"]
            )

    all_keys = sorted(set(
        f"{m}|{c}" for m in models for c in CONDITIONS
    ))
    header = "| Failure Mode | " + " | ".join(
        f"{k.split('|')[0]}:{k.split('|')[1]}" for k in all_keys
    ) + " |"
    lines.append(header)
    lines.append("|---" * (len(all_keys) + 1) + "|")
    for fm in sorted(fm_data.keys()):
        row = f"| {fm} |"
        for key in all_keys:
            scores = fm_data[fm].get(key, [])
            row += f" {avg(scores):.3f} |" if scores else " - |"
        lines.append(row)

    # --- TriMemory wins/losses per model ---
    for model in models:
        lines.append(f"\n## TriMemory vs Others ({model})\n")
        model_results = [r for r in results if r.get("model") == model]
        tri_wins = []
        tri_losses = []
        tri_ties = []

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
                tri_wins.append((sid, tri_sc, best_other))
            elif tri_sc < best_other - 0.01:
                tri_losses.append((sid, tri_sc, best_other))
            else:
                tri_ties.append((sid, tri_sc, best_other))

        lines.append(f"Wins: {len(tri_wins)}, Losses: {len(tri_losses)}, Ties: {len(tri_ties)}")

        if tri_wins:
            lines.append("\nWin examples:")
            for sid, ts, bo in sorted(tri_wins, key=lambda x: -(x[1]-x[2]))[:3]:
                lines.append(f"  {sid}: tri={ts:.3f} vs best_other={bo:.3f} (+{ts-bo:.3f})")

        if tri_losses:
            lines.append("\nLoss examples:")
            for sid, ts, bo in sorted(tri_losses, key=lambda x: x[1]-x[2])[:3]:
                lines.append(f"  {sid}: tri={ts:.3f} vs best_other={bo:.3f} ({ts-bo:.3f})")

    # --- Top success/failure per model ---
    for model in models:
        lines.append(f"\n## Top Success/Failure ({model}, trimemory)\n")
        tri_results = sorted(
            [r for r in results if r.get("model") == model and r["condition"] == "trimemory"],
            key=lambda r: -r["score"]["composite_score"],
        )
        if tri_results:
            lines.append("Top 3 success:")
            for r in tri_results[:3]:
                sc = r["score"]
                lines.append(
                    f"  {r['sample_id']} (composite={sc['composite_score']:.3f}, "
                    f"val={sc['current_value_match']:.2f}, "
                    f"status={sc['status_judgment_match']:.2f}): "
                    f"{r['parsed_answer'][:80]}"
                )
            lines.append("\nTop 3 failure:")
            for r in tri_results[-3:]:
                sc = r["score"]
                flags = []
                if sc["stale_fact_error"]:
                    flags.append("STALE")
                if sc["unsupported_definitive_error"]:
                    flags.append("UNSUP")
                flag_str = " ".join(flags) or "LOW"
                lines.append(
                    f"  {r['sample_id']} (composite={sc['composite_score']:.3f}, "
                    f"{flag_str}): {r['parsed_answer'][:80]}"
                )

    # --- 3B vs 7B interpretation ---
    if len(models) >= 2:
        lines.append("\n## 3B vs 7B Interpretation\n")
        for cond in CONDITIONS:
            model_avgs = []
            for model in models:
                subset = [
                    r["score"]["composite_score"]
                    for r in results
                    if r.get("model") == model and r["condition"] == cond
                ]
                model_avgs.append((model, avg(subset)))
            if len(model_avgs) >= 2:
                diff = model_avgs[1][1] - model_avgs[0][1]
                direction = "+" if diff > 0 else ""
                lines.append(
                    f"- {cond}: {model_avgs[0][0]}={model_avgs[0][1]:.3f}, "
                    f"{model_avgs[1][0]}={model_avgs[1][1]:.3f} "
                    f"(delta={direction}{diff:.3f})"
                )

        # TriMemory gap analysis
        lines.append("\nTriMemory gap (tri - best_other) by model:")
        for model in models:
            model_results = [r for r in results if r.get("model") == model]
            gaps = []
            for sid in set(r["sample_id"] for r in model_results):
                sid_by_cond = {
                    r["condition"]: r for r in model_results
                    if r["sample_id"] == sid
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
                gaps.append(tri_sc - best_other)
            avg_gap = avg(gaps)
            lines.append(f"  {model}: avg gap = {avg_gap:+.3f}")

    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TriMem-DocBench evaluation via Ollama (Phase 2)",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to JSONL or JSON evaluation data",
    )
    parser.add_argument(
        "--model", type=str, nargs="+", default=["llama3.2:3b"],
        help="Ollama model name(s)",
    )
    parser.add_argument("--out", type=str, default="artifacts/docbench_ollama")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--condition", type=str, default="all",
        help="Comma-separated conditions or 'all'",
    )
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of samples (0=all)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--phase2", action="store_true",
        help="Enable Phase 2 scoring and reporting",
    )
    parser.add_argument(
        "--use-trimemory-packet", action="store_true",
        help="Force trimemory condition to use real packet path (default in phase2)",
    )
    parser.add_argument(
        "--trimemory-render-mode", type=str, default="packet_only",
        choices=list(_TRIMEMORY_RENDER_MODES),
        help="How to render the memory packet for the LLM (default: packet_only)",
    )

    args = parser.parse_args()
    data_path = Path(args.data)
    out_dir = Path(args.out)

    # Phase 2 implies packet path
    if args.phase2:
        args.use_trimemory_packet = True

    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    # Load samples
    split_filter = args.split if args.split != "all" else None
    samples = load_samples(data_path, split_filter=split_filter)
    if args.limit > 0:
        samples = samples[:args.limit]
    logger.info("Loaded %d samples (split=%s)", len(samples), args.split)

    # Determine conditions
    if args.condition == "all":
        conditions = CONDITIONS
    else:
        conditions = [c.strip() for c in args.condition.split(",")]

    # Check TriMemory availability
    if args.use_trimemory_packet and not _TRIMEMORY_AVAILABLE:
        logger.error(
            "TriMemory packet path requested but imports failed. "
            "Check that trn package is importable.",
        )
        sys.exit(1)

    # Verify Ollama for each model
    for model_name in args.model:
        try:
            run_ollama("Hello", model=model_name, max_tokens=8)
            logger.info("Ollama OK (model=%s)", model_name)
        except Exception as e:
            logger.error("Ollama health check failed for %s: %s", model_name, e)
            sys.exit(1)

    # Packet log for debug
    packet_log: list[dict] = []

    # Run evaluation for each model
    all_results: list[dict] = []
    for model_name in args.model:
        logger.info("=== Running model: %s ===", model_name)
        results = run_evaluation(
            samples=samples,
            model=model_name,
            out_dir=out_dir,
            top_k=args.top_k,
            conditions=conditions,
            temperature=args.temperature,
            verbose=args.verbose,
            phase2=args.phase2,
            packet_log=packet_log,
            trimemory_render_mode=args.trimemory_render_mode,
        )
        all_results.extend(results)

    # Save packets
    if packet_log:
        packets_path = out_dir / "packets.jsonl"
        with open(packets_path, "w", encoding="utf-8") as f:
            for p in packet_log:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        logger.info("Saved %d packets to %s", len(packet_log), packets_path)

    # Aggregate
    if args.phase2:
        aggregate_results_phase2(all_results, out_dir)
    else:
        aggregate_results_phase2(all_results, out_dir)

    logger.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
