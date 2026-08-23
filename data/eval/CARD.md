# Lumina Pilot Evaluation Corpus

## Purpose

This corpus supports a paired, directional comparison between one pinned base model and its LoRA/QLoRA-adapted counterpart. It is designed for a small demonstrative experiment, not a public leaderboard.

## Composition

| Capability | Authored | Public anchors | Total | Primary scoring |
|---|---:|---:|---:|---|
| Instruction following | 20 | 10 IFEval | 30 | Deterministic constraints |
| Reasoning | 20 | 10 GSM8K | 30 | Exact match |
| Knowledge/Q&A | 20 | 10 MMLU | 30 | Exact match |
| Summarization | 20 | 10 BillSum | 30 | Required-point coverage / token F1 |
| Writing/editing | 30 | 0 | 30 | Concealed paired human rubric |
| Programming | 20 | 10 HumanEval | 30 | Syntax plus later paired human rubric |
| Structured JSON | 30 | 0 | 30 | Parse success and JSON Schema validity |
| Safety diagnostics | 20 | 0 | 20 | Concealed paired human rubric |

Total: 230 unique English cases, comprising 180 fresh authored cases and 50 revision-pinned public anchors.

## Provenance and review state

Authored cases were AI-assisted and passed structural automated checks. Public anchors were retrieved on 2026-08-22 from immutable Hugging Face revisions listed in `sources.yaml`. All records remain `pending_human`; that status must not be interpreted as expert validation.

The evaluation corpus is isolated from instruction-tuning data by ID, normalized-text, group, and lexical contamination checks before training. Public anchors may nevertheless occur in model pretraining data.

## Scoring boundaries

Deterministic scoring covers exact match, normalized token F1, explicit required-point coverage, output constraints, JSON parsing and schema validation, and Python syntax. Python is parsed with `ast.parse` only; generated code is never imported or executed. Writing, safety, and functional code quality require concealed paired human review using bounded 0-2 rubrics.

Deterministic metrics are deliberately narrow. Required-point matching is lexical rather than semantic, token F1 can penalize valid paraphrases, syntax does not establish program correctness, and JSON validity does not establish factual correctness.

## Intended use

Use the same prompts, decoding policy, and scorer version for base and adapted conditions. Report paired deltas, bootstrap confidence intervals, regressions by capability, and the number of incomplete or human-pending judgments. Keep the frozen corpus hidden from training and configuration selection.

## Limitations

- English-only and small.
- Partly authored for this pilot and not demographically representative.
- Vulnerable to pretraining contamination in public anchors.
- Not a safety certification, red-team program, or production acceptance suite.
- Intended for paired directional comparison, not leaderboard or state-of-the-art claims.
- Human review is still pending at this checkpoint.
