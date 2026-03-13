# Prompt 3: Quality Review

```
You are a quality reviewer for TriMem-DocBench evaluation samples.

Review the following sample and flag issues.

SAMPLE:
{sample_json}

CHECK EACH OF THE FOLLOWING (answer YES/NO + explanation if NO):

1. VERSION CLARITY: Is the relationship between document versions
   unambiguous? Are dates chronologically consistent?

2. TRAP EFFECTIVENESS: Would a retrieval system (cosine similarity
   over chunk embeddings) plausibly prefer the distractor document
   over the correct source? Why or why not?

3. ANSWER UNIQUENESS: Is the gold answer the ONLY correct answer?
   Could a different interpretation also be valid?

4. EVIDENCE COMPLETENESS: Does gold_evidence cover ALL documents
   needed to construct the gold answer? Are any missing?

5. FAILURE MODE VALIDITY: For each expected_failure_mode, is the
   triggered_by cause realistic? Would that system architecture
   actually produce that error?

6. SINGLE-DOC TEST: Can the question be fully answered from doc_01
   alone? (Must be NO for the sample to be valid.)

7. UNSUPPORTED DEFINITIVE: Does the question invite a definitive
   answer when the documents actually contain ambiguity or
   unresolved contradictions? If so, is UNSUPPORTED_DEFINITIVE_ANSWER
   in expected_failure_modes?

8. DOCUMENT REALISM: Do the documents look like real business
   documents? (Headers, numbering, dates, responsible parties, etc.)

9. STRUCTURED ANSWER: Does gold_answer_structured.primary_value
   match a specific string in the gold_evidence documents?

10. DISTRACTOR STRENGTH: Rate each distractor 1-5.
    1 = obvious trap, 5 = very convincing.
    Any distractor rated 1-2 should be flagged for strengthening.

OUTPUT FORMAT:
{
  "verdict": "PASS" | "REVISE" | "REJECT",
  "issues": [
    {"check": 1, "status": "YES|NO", "detail": "..."},
    ...
  ],
  "distractor_ratings": [{"value": "...", "rating": N, "note": "..."}],
  "revision_suggestions": ["...", "..."]
}
```
