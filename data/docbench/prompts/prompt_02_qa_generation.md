# Prompt 2: QA / Evidence / Distractor Generation

```
You are annotating evaluation data for TriMem-DocBench.

Given the following documents, generate:
1. A question that REQUIRES cross-document reasoning
2. A gold answer (text + structured)
3. Gold evidence spans
4. Distractors
5. Expected failure modes

DOCUMENTS:
{documents_json}

SAMPLE TYPE: {type}

CONSTRAINTS:
- The question MUST NOT be answerable from doc_01 alone.
- The question must be 1-3 sentences, specific, unambiguous.
- Gold answer text: 2-5 sentences.
- Gold answer structured must include:
  - primary_value: the key fact (number, name, status)
  - source_version: which doc/version it comes from
  - qualifications: list of conditions/caveats
  - superseded_value: the old/wrong value (if applicable)
  - inconsistency_flag: true if documents contradict
  - change_reason: why the value changed (if applicable)
- Gold evidence: 1-3 entries, each with doc_id, span_id, role.
  span_id must reference a real section/item in the document content.
  role is one of: primary_source, change_justification, exception_rule,
  contradiction_source, status_indicator, supporting_context.
- Distractors: 1-3 entries. Each has value, source_doc_id, why_wrong,
  retrieval_attractiveness (low/medium/high).
  retrieval_attractiveness = high means the distractor document
  has more detail or keyword overlap than the correct source.
- Expected failure modes: 1-3 entries.
  mode MUST be from this fixed set:
    STALE_FACT, LATEST_ONLY, MISSING_EXCEPTION,
    INTERMEDIATE_VERSION_CONFUSION, FORMAL_PROVISIONAL_CONFUSION,
    TABLE_TEXT_CONFLICT_MISSED, INCOMPLETE_JUSTIFICATION,
    WRONG_SOURCE_PRIORITY, UNSUPPORTED_DEFINITIVE_ANSWER
  triggered_by: retrieval_only, latest_only, no_cross_reference,
    status_ignorance, shallow_reading.

VERIFICATION STEP (do this before outputting):
1. Try answering the question using ONLY doc_01. If you can fully answer it,
   the question is too easy -- revise to require other documents.
2. Check that gold_answer_structured.primary_value appears verbatim
   in the cited gold_evidence document.
3. Check that each distractor.value appears verbatim in distractor.source_doc_id.
4. Check that at least one failure mode has retrieval_attractiveness = "high".

OUTPUT: JSON object with keys: question, gold_answer_text,
gold_answer_structured, gold_evidence, distractors, expected_failure_modes.
```
