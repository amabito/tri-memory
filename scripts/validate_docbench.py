"""Validate TriMem-DocBench samples against schema and consistency rules.

Usage:
    python scripts/validate_docbench.py data/docbench/raw/TDB-001_draft.json
    python scripts/validate_docbench.py data/docbench/raw/           # validate all
    python scripts/validate_docbench.py data/docbench/final/docbench_dev.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def load_schema() -> dict:
    schema_path = Path(__file__).parent.parent / "data" / "docbench" / "schema.json"
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


def validate_enums(sample: dict, schema: dict) -> list[str]:
    """Check all enum fields against schema-defined allowed values."""
    errors = []

    # Top-level enums
    for field in ("type", "difficulty", "split"):
        allowed = schema["properties"][field].get("enum", [])
        if allowed and sample.get(field) not in allowed:
            errors.append(f"{field}={sample.get(field)!r} not in {allowed}")

    # Document-level enums
    doc_role_enum = (
        schema["properties"]["documents"]["items"]["properties"]["role"]["enum"]
    )
    doc_status_enum = (
        schema["properties"]["documents"]["items"]["properties"]
        ["metadata"]["properties"]["status"]["enum"]
    )
    for doc in sample.get("documents", []):
        if doc.get("role") not in doc_role_enum:
            errors.append(f"{doc['doc_id']}: role={doc.get('role')!r} invalid")
        status = doc.get("metadata", {}).get("status")
        if status and status not in doc_status_enum:
            errors.append(f"{doc['doc_id']}: status={status!r} invalid")

    # Evidence role enum
    ev_role_enum = (
        schema["properties"]["gold_evidence"]["items"]["properties"]["role"]["enum"]
    )
    for ev in sample.get("gold_evidence", []):
        if ev.get("role") not in ev_role_enum:
            errors.append(f"gold_evidence role={ev.get('role')!r} invalid")

    # Failure mode enum
    fm_mode_enum = (
        schema["properties"]["expected_failure_modes"]["items"]
        ["properties"]["mode"]["enum"]
    )
    fm_trigger_enum = (
        schema["properties"]["expected_failure_modes"]["items"]
        ["properties"]["triggered_by"]["enum"]
    )
    for fm in sample.get("expected_failure_modes", []):
        if fm.get("mode") not in fm_mode_enum:
            errors.append(f"failure_mode={fm.get('mode')!r} invalid")
        trigger = fm.get("triggered_by")
        if trigger and trigger not in fm_trigger_enum:
            errors.append(f"triggered_by={trigger!r} invalid")

    # Capability enum
    cap_enum = schema["properties"]["required_capabilities"]["items"]["enum"]
    for cap in sample.get("required_capabilities", []):
        if cap not in cap_enum:
            errors.append(f"capability={cap!r} invalid")

    return errors


def validate_references(sample: dict) -> list[str]:
    """Check that doc_id references in evidence/distractors point to real docs."""
    errors = []
    doc_ids = {doc["doc_id"] for doc in sample.get("documents", [])}

    for ev in sample.get("gold_evidence", []):
        if ev["doc_id"] not in doc_ids:
            errors.append(
                f"gold_evidence references {ev['doc_id']} "
                f"but available docs are {doc_ids}"
            )

    for dist in sample.get("distractors", []):
        if dist["source_doc_id"] not in doc_ids:
            errors.append(
                f"distractor references {dist['source_doc_id']} "
                f"but available docs are {doc_ids}"
            )

    return errors


def validate_required_fields(sample: dict, schema: dict) -> list[str]:
    """Check that all required top-level fields exist."""
    errors = []
    for field in schema.get("required", []):
        if field not in sample:
            errors.append(f"missing required field: {field}")
    return errors


def validate_sample_id(sample: dict) -> list[str]:
    """Check sample_id format."""
    sid = sample.get("sample_id", "")
    if not re.match(r"^TDB-[0-9]{3}$", sid):
        return [f"sample_id={sid!r} does not match TDB-NNN pattern"]
    return []


def validate_doc_count(sample: dict) -> list[str]:
    """Check document count is 3-6."""
    n = len(sample.get("documents", []))
    if n < 3:
        return [f"too few documents: {n} (min 3)"]
    if n > 6:
        return [f"too many documents: {n} (max 6)"]
    return []


def validate_content_length(sample: dict) -> list[str]:
    """Warn if document content is outside 150-500 token range (approx)."""
    warnings = []
    for doc in sample.get("documents", []):
        content = doc.get("content", "")
        # Rough token estimate: chars / 3 for Japanese, chars / 4 for English
        char_count = len(content)
        if char_count < 100:
            warnings.append(
                f"{doc['doc_id']}: content too short ({char_count} chars)"
            )
        if char_count > 2500:
            warnings.append(
                f"{doc['doc_id']}: content too long ({char_count} chars)"
            )
    return warnings


def validate_one(sample: dict, schema: dict) -> tuple[list[str], list[str]]:
    """Validate one sample. Returns (errors, warnings)."""
    errors = []
    warnings = []

    errors.extend(validate_required_fields(sample, schema))
    errors.extend(validate_sample_id(sample))
    errors.extend(validate_enums(sample, schema))
    errors.extend(validate_references(sample))
    errors.extend(validate_doc_count(sample))
    warnings.extend(validate_content_length(sample))

    # Check gold_evidence has at least 1 entry
    if len(sample.get("gold_evidence", [])) == 0:
        errors.append("gold_evidence is empty")

    # Check expected_failure_modes has at least 1 entry
    if len(sample.get("expected_failure_modes", [])) == 0:
        errors.append("expected_failure_modes is empty")

    return errors, warnings


def load_samples(path: Path) -> list[tuple[str, dict]]:
    """Load samples from a JSON file, JSONL file, or directory."""
    samples = []
    if path.is_dir():
        for f in sorted(path.glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                samples.append((f.name, json.load(fh)))
        for f in sorted(path.glob("*.jsonl")):
            with open(f, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    line = line.strip()
                    if line:
                        samples.append((f"{f.name}:L{i+1}", json.loads(line)))
    elif path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if line:
                    samples.append((f"{path.name}:L{i+1}", json.loads(line)))
    else:
        with open(path, encoding="utf-8") as fh:
            samples.append((path.name, json.load(fh)))
    return samples


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file_or_dir>")
        sys.exit(1)

    path = Path(sys.argv[1])
    schema = load_schema()
    samples = load_samples(path)

    if not samples:
        print(f"No samples found at {path}")
        sys.exit(1)

    total_errors = 0
    total_warnings = 0

    for source, sample in samples:
        errors, warnings = validate_one(sample, schema)
        sid = sample.get("sample_id", "???")

        if errors:
            print(f"\n[FAIL] {source} ({sid})")
            for e in errors:
                print(f"  ERROR: {e}")
            total_errors += len(errors)

        if warnings:
            if not errors:
                print(f"\n[WARN] {source} ({sid})")
            for w in warnings:
                print(f"  WARN:  {w}")
            total_warnings += len(warnings)

        if not errors and not warnings:
            print(f"[OK]   {source} ({sid})")

    print(f"\n--- Summary: {len(samples)} samples, "
          f"{total_errors} errors, {total_warnings} warnings ---")

    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
