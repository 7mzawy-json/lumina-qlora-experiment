# Lumina Demonstration Authored Data Card

This directory contains 150 English instruction-tuning candidates created specifically for the Lumina demonstrative experiment: 75 structured JSON tasks and 75 multi-constraint instruction-following tasks.

## Provenance and review status

- Provenance: `lumina-demonstration-authored`
- License label: CC-BY-4.0
- Authorship: AI-assisted drafting for this candidate evaluation
- Automated review date: 2026-08-21
- Human review status: pending

Each record is marked `automated_checks_passed_pending_human`. That label means structural and constraint checks may accept the row for engineering tests, but it must not be represented as independently human-reviewed. Before an 8B measured run, the candidate should manually spot-check every record and change approved rows to `human_approved`, or report the remaining limitation explicitly.

## Scope and privacy

The prompts use synthetic, generic scenarios. They contain no Parallax Horizon private information, user data, credentials, or claims about Lumina's undisclosed architecture.

## Limitations

The corpus is intentionally small and template-balanced. Its purpose is to test whether targeted supervised fine-tuning can improve JSON reliability and multi-constraint instruction following without making broader capability claims. Template regularity may inflate in-domain gains, so evaluation prompts must remain independently authored and contamination-checked.
