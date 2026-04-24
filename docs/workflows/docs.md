---
description: Documentation-only change
---

Use when updating docstrings, README/docs, or comments.

### Standards
- Adhere to **Docstring & Typing** rules in [`typing-docstrings.md`](../standards/typing-docstrings.md).
- Follow **Markdown heading capitalization** in [`code-clarity.md`](../standards/code-clarity.md#header-and-title-capitalization): use sentence case (capitalize only the first word, except proper nouns/acronyms).
- Follow **Markdown link path rules** in [`code-clarity.md`](../standards/code-clarity.md#markdown-link-paths): use relative links for repository-internal targets.
- Focus on "why" (rationale) rather than just "what" (implementation).

1. Make doc changes (focus on "why" and "how to use").
// turbo
2. Run formatting/lint gate if Python files were touched:
   ```bash
   uv run pre-commit run --all-files
   ```
3. If docs describe behavior that is testable, ensure tests exist or add them.

**Done criteria:**
- Docs are consistent with code
- Markdown headings use sentence-case capitalization
- Repository-internal links are relative and resolve correctly
- Any touched code still passes `pre-commit`
