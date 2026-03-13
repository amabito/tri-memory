# Prompt 5: JSON Normalization

```
Convert the following TriMem-DocBench sample draft into final JSON
that conforms to the schema.

DRAFT:
{draft_text_or_json}

SCHEMA REQUIREMENTS:
- sample_id: "TDB-{NNN}" (I will tell you the number: {sample_number})
- type: one of the 8 fixed types
- difficulty: easy/medium/hard
- documents[].doc_id: "doc_01" through "doc_06"
- documents[].role: one of the fixed role enums
- documents[].metadata.status: current/superseded/draft/provisional/archived
- documents[].content: plain text with \n for newlines
- gold_evidence[].role: one of the fixed evidence role enums
- expected_failure_modes[].mode: one of the 9 fixed failure mode labels
- expected_failure_modes[].triggered_by: one of the 5 fixed trigger labels
- required_capabilities: list from the 12 fixed capability labels
- split: "dev" or "test" (I will tell you: {split})

NORMALIZATION RULES:
1. All content strings: replace literal newlines with \n
2. All dates: YYYY-MM-DD format
3. Remove any markdown formatting from content (no **, no ##)
4. Ensure all enum values are from the allowed set
5. Ensure doc_id references in gold_evidence and distractors
   match actual doc_ids in the documents array

OUTPUT: Valid JSON only, no commentary.
```
