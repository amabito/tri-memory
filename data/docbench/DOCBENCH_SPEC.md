# TriMem-DocBench Specification

Version: 1.0 (2026-03-13)
Purpose: TriMemory real-data PoC evaluation dataset

---

## 1. Final JSON Schema

### Schema Definition

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": [
    "sample_id", "type", "difficulty", "documents", "question",
    "gold_answer_text", "gold_answer_structured", "gold_evidence",
    "distractors", "expected_failure_modes", "required_capabilities",
    "evaluator_notes", "split"
  ],
  "properties": {
    "sample_id": {
      "type": "string",
      "pattern": "^TDB-[0-9]{3}$",
      "description": "Unique sample identifier"
    },
    "type": {
      "type": "string",
      "enum": [
        "superseded_condition",
        "overwritten_history",
        "exception_in_notes",
        "table_text_inconsistency",
        "unresolved_minutes",
        "faq_vs_formal_spec",
        "interim_error",
        "quantity_prose_conflict"
      ]
    },
    "difficulty": {
      "type": "string",
      "enum": ["easy", "medium", "hard"]
    },
    "documents": {
      "type": "array",
      "minItems": 3,
      "maxItems": 6,
      "items": {
        "type": "object",
        "required": ["doc_id", "role", "title", "content", "metadata"],
        "properties": {
          "doc_id": {
            "type": "string",
            "pattern": "^doc_[0-9]{2}$"
          },
          "role": {
            "type": "string",
            "enum": [
              "current_main",
              "previous_version",
              "change_notice",
              "supplementary",
              "minutes",
              "table_data",
              "faq",
              "draft",
              "formal_spec"
            ]
          },
          "title": { "type": "string" },
          "content": {
            "type": "string",
            "description": "Plain text, 150-500 tokens. Sections delimited by headers."
          },
          "metadata": {
            "type": "object",
            "properties": {
              "version": { "type": "string" },
              "date": { "type": "string", "format": "date" },
              "status": {
                "type": "string",
                "enum": ["current", "superseded", "draft", "provisional", "archived"]
              },
              "references": {
                "type": "array",
                "items": { "type": "string" }
              }
            }
          }
        }
      }
    },
    "question": {
      "type": "string",
      "description": "1-3 sentences. Must require cross-document reasoning."
    },
    "gold_answer_text": {
      "type": "string",
      "description": "Natural language answer, 2-5 sentences."
    },
    "gold_answer_structured": {
      "type": "object",
      "required": ["primary_value", "source_version", "qualifications"],
      "properties": {
        "primary_value": {
          "type": "string",
          "description": "The key factual answer (e.g. '450kN', 'Fc=30')"
        },
        "source_version": {
          "type": "string",
          "description": "Which document version the primary value comes from"
        },
        "qualifications": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Conditions, caveats, or additional facts needed for completeness"
        },
        "superseded_value": {
          "type": "string",
          "description": "The old/wrong value that should NOT be given"
        },
        "inconsistency_flag": {
          "type": "boolean",
          "description": "True if documents contain unresolved contradictions"
        },
        "change_reason": {
          "type": "string",
          "description": "Why the value changed, if applicable"
        }
      }
    },
    "gold_evidence": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["doc_id", "span_id", "role"],
        "properties": {
          "doc_id": { "type": "string" },
          "span_id": {
            "type": "string",
            "description": "Section/item/row identifier within the document"
          },
          "role": {
            "type": "string",
            "enum": [
              "primary_source",
              "change_justification",
              "exception_rule",
              "contradiction_source",
              "status_indicator",
              "supporting_context"
            ]
          }
        }
      }
    },
    "distractors": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["value", "source_doc_id", "why_wrong"],
        "properties": {
          "value": { "type": "string" },
          "source_doc_id": { "type": "string" },
          "why_wrong": { "type": "string" },
          "retrieval_attractiveness": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "How likely a retrieval system picks this up"
          }
        }
      }
    },
    "expected_failure_modes": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["mode", "description"],
        "properties": {
          "mode": {
            "type": "string",
            "enum": [
              "STALE_FACT",
              "LATEST_ONLY",
              "MISSING_EXCEPTION",
              "INTERMEDIATE_VERSION_CONFUSION",
              "FORMAL_PROVISIONAL_CONFUSION",
              "TABLE_TEXT_CONFLICT_MISSED",
              "INCOMPLETE_JUSTIFICATION",
              "WRONG_SOURCE_PRIORITY",
              "UNSUPPORTED_DEFINITIVE_ANSWER"
            ]
          },
          "description": {
            "type": "string",
            "description": "What the wrong answer looks like"
          },
          "triggered_by": {
            "type": "string",
            "enum": [
              "retrieval_only",
              "latest_only",
              "no_cross_reference",
              "status_ignorance",
              "shallow_reading"
            ]
          }
        }
      }
    },
    "required_capabilities": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": [
          "version_resolution",
          "cross_document_evidence",
          "change_reason_retrieval",
          "exception_detection",
          "multi_condition_integration",
          "inconsistency_detection",
          "formal_provisional_distinction",
          "multi_version_tracking",
          "change_chain_reconstruction",
          "document_status_awareness",
          "source_priority_judgment",
          "quantitative_cross_check"
        ]
      }
    },
    "evaluator_notes": {
      "type": "string",
      "description": "Free-text notes for human reviewers about tricky aspects"
    },
    "split": {
      "type": "string",
      "enum": ["dev", "test"]
    }
  }
}
```

### Full Example

```json
{
  "sample_id": "TDB-001",
  "type": "superseded_condition",
  "difficulty": "hard",
  "documents": [
    {
      "doc_id": "doc_01",
      "role": "current_main",
      "title": "基礎設計仕様書 Rev.3 (2026-02-15)",
      "content": "3. 杭基礎設計\n3.1 設計条件\n支持層: 砂礫層 (GL-15m)\n3.2 杭の支持力\n設計支持力: 450kN/本\n杭径: phi600mm\n杭長: 12.0m\n安全率: 1.3\n3.3 配置\n杭本数: 48本 (6x8 grid, ピッチ 2.5D)",
      "metadata": {
        "version": "3.0",
        "date": "2026-02-15",
        "status": "current"
      }
    },
    {
      "doc_id": "doc_02",
      "role": "previous_version",
      "title": "基礎設計仕様書 Rev.2 (2025-11-01)",
      "content": "3. 杭基礎設計\n3.1 設計条件\n支持層: 砂礫層 (GL-15m)\n3.2 杭の支持力\n設計支持力: 380kN/本\n杭径: phi600mm\n杭長: 12.0m\n安全率: 1.5\n備考: 地盤調査報告書 GR-2025-018 に基づく。N値=35 (GL-15m) での極限支持力 570kN に安全率 1.5 を適用。\n3.3 配置\n杭本数: 48本 (6x8 grid, ピッチ 2.5D)",
      "metadata": {
        "version": "2.0",
        "date": "2025-11-01",
        "status": "superseded"
      }
    },
    {
      "doc_id": "doc_03",
      "role": "change_notice",
      "title": "設計変更通知 DCN-2026-003 (2026-02-10)",
      "content": "変更対象: 基礎設計仕様書 3.2 杭の支持力\n変更前: 380kN/本 (安全率 1.5)\n変更後: 450kN/本 (安全率 1.3)\n変更理由: 載荷試験 (LT-2026-001) の実測値 612kN が推定極限支持力 570kN を上回ったため、安全率を 1.5 から 1.3 に緩和。\n承認: 構造設計部長 2026-02-12\n備考: 杭径・杭長・配置は変更なし。",
      "metadata": {
        "version": "1.0",
        "date": "2026-02-10",
        "status": "current",
        "references": ["doc_01", "doc_02"]
      }
    }
  ],
  "question": "現行の杭の設計支持力はいくつか。また Rev.2 からの変更理由を述べよ。",
  "gold_answer_text": "現行の設計支持力は 450kN/本 (Rev.3)。Rev.2 では 380kN/本 (安全率 1.5) だったが、載荷試験 LT-2026-001 の実測値 612kN が推定極限支持力を上回ったため、安全率を 1.3 に緩和し 450kN に引き上げた (DCN-2026-003)。",
  "gold_answer_structured": {
    "primary_value": "450kN/本",
    "source_version": "Rev.3 (doc_01)",
    "qualifications": [
      "安全率 1.3",
      "載荷試験 LT-2026-001 に基づく緩和"
    ],
    "superseded_value": "380kN/本 (Rev.2)",
    "inconsistency_flag": false,
    "change_reason": "載荷試験実測値 612kN > 推定極限支持力 570kN のため安全率 1.5 -> 1.3 に緩和"
  },
  "gold_evidence": [
    {
      "doc_id": "doc_01",
      "span_id": "section_3.2",
      "role": "primary_source"
    },
    {
      "doc_id": "doc_03",
      "span_id": "変更理由",
      "role": "change_justification"
    }
  ],
  "distractors": [
    {
      "value": "380kN/本",
      "source_doc_id": "doc_02",
      "why_wrong": "Rev.2 の値。Rev.3 で 450kN に変更済み。",
      "retrieval_attractiveness": "high"
    }
  ],
  "expected_failure_modes": [
    {
      "mode": "STALE_FACT",
      "description": "Rev.2 の 380kN をそのまま回答。doc_02 の方が詳細な計算根拠を含むため retrieval スコアが高くなりやすい。",
      "triggered_by": "retrieval_only"
    },
    {
      "mode": "INCOMPLETE_JUSTIFICATION",
      "description": "450kN は答えるが、変更理由を DCN-2026-003 から引けず「安全率の見直し」程度で終わる。",
      "triggered_by": "latest_only"
    }
  ],
  "required_capabilities": [
    "version_resolution",
    "cross_document_evidence",
    "change_reason_retrieval"
  ],
  "evaluator_notes": "doc_02 の備考欄に N値=35, 極限支持力 570kN の詳細計算が含まれており、embedding 類似度で doc_02 が doc_01 より高スコアになりやすい。これが STALE_FACT trap の核心。",
  "split": "dev"
}
```

---

## 2. AI Generation Pipeline

### Step A: Type Selection

**AI task**: なし (人間が決定)
**Human task**: 16 samples の type 配分を決める。下記 Section 5 の計画に従う。
**Common mistake**: 得意な type に偏る。8 type を均等に割り当てること。

### Step B: Document Drafts

**AI task**: type を指定して 3-5 文書のドラフトを生成 (Prompt 1 使用)。
**Human confirms**:
  - 文書間の版関係が明確か (日付、Rev番号、ステータス)
  - content が 150-500 tokens に収まっているか
  - 「罠」になる旧情報が十分に詳細か (詳細な方が retrieval で引かれやすい)
  - 各 doc の role が正しいか
**Common generation mistakes**:
  - 旧版と最新版の差異が小さすぎる → 値を大きく変える (380 vs 450 のように 15%+ の差)
  - 全文書が同じ文体 → role ごとに書式を変えるよう指示 (仕様書は箇条書き、議事録は対話体、通知は公文書体)
  - metadata の日付が論理矛盾 (変更通知が最新版より後の日付) → 時系列を明示チェック
**Fix**: Prompt 1 にフォーマット制約を含め、生成後に日付順ソートで矛盾検出。

### Step C: Question Generation

**AI task**: 文書群を入力し、cross-document でないと答えられない質問を生成 (Prompt 2 の一部)。
**Human confirms**:
  - doc_01 (最新版) だけで完答できないこと
  - 質問が 1-3 文に収まっていること
  - 質問が曖昧でなく、gold_answer が一意に定まること
**Common mistakes**:
  - 単一文書で完答可能な質問を生成 → 「この質問に doc_01 だけで答えられるか?」テストを挟む
  - 複数の解釈が可能な質問 → 条件を明示 (「工区Bの」「現時点で有効な」)
**Fix**: 生成後に「doc_01 のみで回答」を試み、完答できたら reject。

### Step D: Gold Answer / Evidence

**AI task**: 文書群 + 質問から gold_answer_text, gold_answer_structured, gold_evidence を生成 (Prompt 2)。
**Human confirms**:
  - gold_answer_structured.primary_value が文書中の文字列と完全一致するか
  - gold_evidence の span_id が文書内に実在するか
  - qualifications が answer_text の内容と整合するか
  - superseded_value が実際に旧版に存在するか
**Common mistakes**:
  - span_id が曖昧 ("後半部分" のような指定) → セクション番号/項目番号を使う
  - gold_answer に文書にない情報を含む (hallucination) → 全 assertion を文書に traceback
  - structured answer と text answer が矛盾 → text を structured から機械生成すると安全
**Fix**: gold_evidence の各 span_id について、該当文書の content を grep して存在確認。

### Step E: Distractors / Failure Modes

**AI task**: 文書群 + gold_answer から distractor と failure mode を生成 (Prompt 2 の一部)。
**Human confirms**:
  - distractor.value が実際に文書中に存在する値か
  - retrieval_attractiveness の判定が妥当か (旧版が詳細 = high)
  - failure mode の triggered_by が論理的に正しいか
  - UNSUPPORTED_DEFINITIVE_ANSWER が適用可能なケースで漏れていないか
**Common mistakes**:
  - distractor が弱すぎる (明らかに古い日付の文書からの値 → 日付を目立たなくする)
  - failure mode が網羅的でない → type ごとに minimum failure modes を規定:
    - superseded_condition: STALE_FACT + INCOMPLETE_JUSTIFICATION
    - exception_in_notes: MISSING_EXCEPTION + LATEST_ONLY
    - table_text_inconsistency: TABLE_TEXT_CONFLICT_MISSED + UNSUPPORTED_DEFINITIVE_ANSWER
    - unresolved_minutes: FORMAL_PROVISIONAL_CONFUSION
**Fix**: Prompt 4 で「この distractor は弱い。retrieval システムが引っかかる理由を強化せよ」と指示。

### Step F: Human Review

**AI task**: なし
**Human task**: Section 4 のチェックリストに従いレビュー。pass/fail 判定。
**Common mistakes**: レビューが甘い (AI 生成物を信用しすぎる)
**Fix**: 最初の 8 件は 2 名レビュー。残りは 1 名。

### Step G: JSON Normalization

**AI task**: レビュー済みテキストを最終 JSON に変換 (Prompt 5)。
**Human confirms**:
  - JSON が schema に準拠するか (jsonschema validate)
  - enum 値が正しいか
  - sample_id が重複していないか
**Common mistakes**:
  - Unicode エスケープの不整合 → UTF-8 で保存
  - content 内の改行が `\n` になっていない → jq で正規化
**Fix**: `python -m json.tool < sample.json` で syntax check。schema validator script を用意。

### Step H: Final Acceptance

**AI task**: なし
**Human task**:
  1. JSON を読み込み、question に対して gold_answer を伏せた状態で自分で回答を試みる
  2. 自分の回答と gold_answer を比較
  3. 自分が間違えた場合: trap が効いている = good sample
  4. 自分が正答した場合: trap が弱い可能性。distractor の強化を検討
  5. 最終採用 / 修正 / reject を決定
**Common mistakes**: 全サンプルを accept してしまう
**Fix**: reject 率 20-30% を目標にする。弱いサンプルは Prompt 4 で強化して再提出。

---

## 3. Generation Prompts

### Prompt 1: Document Draft Generation

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

### Prompt 2: QA / Evidence / Distractor Generation

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

### Prompt 3: Quality Review

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

### Prompt 4: Sample Strengthening

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

### Prompt 5: JSON Normalization

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

---

## 4. Human Review Checklist

For each sample, check all of the following. Mark PASS / FAIL.

```
REVIEW CHECKLIST -- TriMem-DocBench Sample Review

Sample ID: ___________
Reviewer: ___________
Date: ___________

[STRUCTURE]
[ ] 1. Documents have clear version/date hierarchy
[ ] 2. metadata.status is correct for each document
[ ] 3. doc_id references in evidence/distractors match actual documents
[ ] 4. Each document content is 150-500 tokens
[ ] 5. Document titles follow realistic naming conventions

[TRAP QUALITY]
[ ] 6. Distractor document has HIGHER detail/keyword density
       than the correct source document
[ ] 7. Distractor value is a REAL value from a REAL document
       (not fabricated by the annotator)
[ ] 8. A naive retrieval system would plausibly rank the
       distractor document higher
[ ] 9. The trap targets a specific system weakness
       (retrieval-only, latest-only, etc.)

[ANSWER QUALITY]
[ ] 10. gold_answer_text and gold_answer_structured are consistent
[ ] 11. primary_value appears verbatim in the cited evidence document
[ ] 12. The answer is UNIQUE -- no alternative valid interpretation
[ ] 13. qualifications list all necessary caveats
[ ] 14. If inconsistency_flag is true, the contradiction is real

[EVIDENCE QUALITY]
[ ] 15. gold_evidence covers ALL documents needed for the answer
[ ] 16. span_id references exist in the document content
[ ] 17. evidence roles are correctly assigned

[FAILURE MODES]
[ ] 18. At least 1 failure mode per sample
[ ] 19. Each failure mode label is from the fixed set
[ ] 20. triggered_by is logically consistent with the mode
[ ] 21. UNSUPPORTED_DEFINITIVE_ANSWER is included when documents
        contain genuine ambiguity

[SINGLE-DOC TEST]
[ ] 22. Question CANNOT be fully answered from doc_01 alone
        (critical -- reject if answerable from single doc)

[REALISM]
[ ] 23. Documents read like actual business/technical documents
[ ] 24. No AI-isms ("it's worth noting", "comprehensive", etc.)
[ ] 25. Version numbers, dates, approval signatures are present

VERDICT: [ ] PASS  [ ] REVISE (list items)  [ ] REJECT (reason)
```

---

## 5. First 16 Samples Plan (Week 1)

### Type Distribution

| Type | Count | Difficulty | Docs/Sample | Priority |
|------|-------|-----------|-------------|----------|
| superseded_condition | 3 | 1 medium, 2 hard | 3 | Day 1-2 |
| exception_in_notes | 3 | 1 easy, 1 medium, 1 hard | 3-4 | Day 1-2 |
| overwritten_history | 2 | 1 medium, 1 hard | 4 | Day 3 |
| table_text_inconsistency | 2 | 1 medium, 1 hard | 3 | Day 3 |
| unresolved_minutes | 2 | 1 medium, 1 hard | 3 | Day 4 |
| faq_vs_formal_spec | 2 | 1 easy, 1 medium | 3 | Day 4 |
| interim_error | 1 | medium | 3 | Day 5 |
| quantity_prose_conflict | 1 | medium | 3 | Day 5 |
| **Total** | **16** | 2 easy, 8 medium, 6 hard | 3-4 avg | 5 days |

### Day-by-Day Plan

**Day 1-2 (6 samples)**: superseded_condition x3 + exception_in_notes x3
- AI generates document drafts via Prompt 1 (6 invocations)
- Human reviews structure, strengthens traps
- AI generates QA/evidence via Prompt 2 (6 invocations)
- Human does single-doc test on all 6

**Day 3 (4 samples)**: overwritten_history x2 + table_text_inconsistency x2
- Same pipeline
- overwritten_history requires 4 docs each -- more complex, slower

**Day 4 (4 samples)**: unresolved_minutes x2 + faq_vs_formal_spec x2
- Same pipeline
- Focus on formal/provisional distinction clarity

**Day 5 (2 samples + review)**: interim_error x1 + quantity_prose_conflict x1
- Generate remaining 2 samples
- Run Prompt 3 (quality review) on all 16
- Fix issues flagged in review
- Run Prompt 5 (JSON normalization) on all accepted samples
- Final human acceptance (Step H)

### Split Assignment
- dev: 10 samples (TDB-001 through TDB-010)
- test: 6 samples (TDB-011 through TDB-016)

### Domain Distribution (avoid monotony)
- Construction/civil: 5 samples
- Procurement/contracts: 3 samples
- Quality management/ISO: 3 samples
- Internal regulations/HR: 3 samples
- Product specifications: 2 samples

---

## 6. Directory Structure

```
D:\work\Projects\trn\data\docbench\
  DOCBENCH_SPEC.md          # This file
  schema.json               # JSON Schema for validation

  raw/                      # AI-generated drafts (pre-review)
    TDB-001_draft.json
    TDB-002_draft.json
    ...

  reviewed/                 # Post-review, pre-normalization
    TDB-001_reviewed.json   # Has reviewer annotations
    TDB-002_reviewed.json
    ...

  review_notes/             # Review checklists and feedback
    TDB-001_review.md
    TDB-002_review.md
    ...

  final/                    # Production-ready samples
    docbench_dev.jsonl       # 1 sample per line, dev split
    docbench_test.jsonl      # 1 sample per line, test split
    manifest.json            # {total, by_type, by_difficulty, by_split}

  prompts/                  # Generation prompts (versioned)
    prompt_01_doc_draft.md
    prompt_02_qa_generation.md
    prompt_03_quality_review.md
    prompt_04_strengthening.md
    prompt_05_json_normalize.md

D:\work\Projects\trn\scripts\
  validate_docbench.py      # JSON schema validation
  eval_docbench.py          # Run model evaluation
  score_docbench.py         # Compute metrics from predictions
```

### File Format Rules
- `raw/`, `reviewed/`: individual JSON files (1 per sample, pretty-printed)
- `final/`: JSONL (1 sample per line, compact)
- `review_notes/`: Markdown (checklist format)
- `prompts/`: Markdown (copy-pasteable prompts)
- `manifest.json`: auto-generated by `validate_docbench.py`

---

## 7. Summary

**Today's first 3 actions**:
1. Prompt 1 を使って superseded_condition の medium 難易度サンプル 1 件を生成し、手動で single-doc test を実施。trap が効くか確認。
2. `scripts/validate_docbench.py` を作成 -- schema.json で JSON validation + doc_id 参照整合性チェック + enum 値チェック。
3. TDB-001 を final JSON まで仕上げ、pipeline 全工程 (A-H) を 1 件通して手順の穴を洗い出す。

**AI generation's biggest pitfall**: distractor が弱い。AI は「正解を知った上で」distractor を作るため、「明らかに古い」「明らかに draft」な文書を生成しがち。retrieval system は日付やステータスを見ない -- keyword overlap と detail density で判断する。旧版の方が詳細であること、旧版のタイトルに "[旧版]" と書かないこと、を徹底的に指示する必要がある。

**Strict human review count**: 最初の 8 件は全工程を人手で厳密にレビュー。特に single-doc test (check 22) と distractor strength (check 6-8) は妥協しない。9 件目以降は AI review (Prompt 3) + 人手スポットチェックに移行可能。
