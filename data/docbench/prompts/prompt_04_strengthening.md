# Prompt 4: Sample Strengthening

```
You are strengthening a weak TriMem-DocBench sample.

The following sample was flagged during review. Your job is to make
the trap MORE effective without changing the gold answer.

CURRENT SAMPLE:
{sample_json}

REVIEW FEEDBACK:
{review_feedback}

STRENGTHENING TECHNIQUES:
1. Add more technical detail to the WRONG document (old version,
   draft, FAQ) so retrieval systems prefer it.
2. Make the dates less prominent (bury version numbers in metadata,
   not titles).
3. Add a second distractor from a different document.
4. Add cross-references in the wrong document that make it look
   like the authoritative source.
5. Make the correct document's key section shorter and less keyword-rich.
6. Add a plausible-but-wrong calculation in the distractor document.

CONSTRAINTS:
- Do NOT change the gold answer.
- Do NOT change the question.
- Do NOT add new documents (modify existing ones).
- The strengthened sample must still be factually consistent
  (no internal contradictions in the "correct" documents).

OUTPUT: The full revised sample JSON.
```
