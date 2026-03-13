# Prompt 1: Document Draft Generation

```
You are generating evaluation data for a multi-document QA benchmark called TriMem-DocBench.

TASK: Generate a set of 3-5 documents for one evaluation sample.

SAMPLE TYPE: {type}
(One of: superseded_condition, overwritten_history, exception_in_notes,
table_text_inconsistency, unresolved_minutes, faq_vs_formal_spec,
interim_error, quantity_prose_conflict)

DIFFICULTY: {difficulty}
(easy / medium / hard)

DOMAIN: Choose from construction, procurement, quality management,
internal regulations, or product specifications. Pick whichever
fits the sample type naturally.

REQUIREMENTS:
1. Generate exactly {n_docs} documents (3-5).
2. Each document must have: doc_id (doc_01..doc_05), role, title, content, metadata.
3. Roles to use: current_main, previous_version, change_notice, supplementary,
   minutes, table_data, faq, draft, formal_spec.
4. Each document content: 150-500 tokens. Use section headers, numbered items,
   or table format as appropriate for the document type.
5. Documents must have CLEAR version/date relationships.
   Dates must be chronologically consistent.
6. The document set MUST contain a TRAP:
   - An old value that is MORE DETAILED than the new value
     (so retrieval systems prefer the old document)
   - OR a contradiction between two documents of similar authority
   - OR an exception buried in a supplementary document
7. The trap must be NON-OBVIOUS. Do not put "[OLD VERSION]" in the title.
   Make the old document look authoritative.
8. Use realistic document naming: "Rev.X", "DCN-YYYY-NNN",
   "第N回定例議事録", "付録X", etc.
9. Include specific numerical values, dates, and identifiers.
   Avoid vague language like "approximately" in key values
   (except when vagueness IS the trap, as in quantity_prose_conflict).

OUTPUT FORMAT:
Return a JSON array of document objects. Each object:
{
  "doc_id": "doc_01",
  "role": "current_main",
  "title": "...",
  "content": "...",
  "metadata": {
    "version": "...",
    "date": "YYYY-MM-DD",
    "status": "current|superseded|draft|provisional|archived",
    "references": []
  }
}

TYPE-SPECIFIC INSTRUCTIONS:

superseded_condition:
  - doc_01 = current version (less detail)
  - doc_02 = previous version (MORE detail, calculation notes)
  - doc_03 = change notice (explains why)
  - TRAP: doc_02 has richer technical detail, making retrieval prefer it

overwritten_history:
  - doc_01 = current version (value C)
  - doc_02 = first change record (A -> B)
  - doc_03 = second change record (B -> C)
  - doc_04 = intermediate version (value B)
  - TRAP: intermediate value B looks current if only one change record is found

exception_in_notes:
  - doc_01 = main specification (general rule)
  - doc_02 = notes/appendix (exception rule)
  - doc_03 = context document (shows which exception applies)
  - TRAP: main spec states a general value; notes contain an exception
    that applies to the specific case in the question

table_text_inconsistency:
  - doc_01 = prose document (value X in text)
  - doc_02 = table/schedule (value Y, different from X)
  - doc_03 = design intent or reasoning document
  - TRAP: text and table disagree; design intent clarifies which is correct

unresolved_minutes:
  - doc_01 = formal specification (old value, still officially current)
  - doc_02 = meeting minutes (decision to change, but not yet enacted)
  - doc_03 = provisional calculation (new value, marked provisional)
  - TRAP: minutes decision is not yet formal

faq_vs_formal_spec:
  - doc_01 = FAQ (simplified answer, incomplete)
  - doc_02 = formal specification (complete with conditions)
  - doc_03 = context document
  - TRAP: FAQ omits important conditions

interim_error:
  - doc_01 = final version (correct value)
  - doc_02 = draft version (WRONG value, clearly labeled draft)
  - doc_03 = review comment or errata
  - TRAP: draft has a plausible-looking wrong value

quantity_prose_conflict:
  - doc_01 = prose description (approximate value)
  - doc_02 = calculation sheet or quantity table (exact value)
  - doc_03 = summary or cover sheet
  - TRAP: prose rounds or approximates; table has precise value
```
