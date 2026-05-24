<!--
Be synthetic: bullet points are preferred and 1–3 short sentences per section.
Delete any guidance comments before submitting.
-->

## TL;DR
<!-- What does this PR do, in one or two sentences? -->
- Short summary: ...

## Breaking Changes
<!-- List anything that requires user action or could break existing behavior. Link to migration steps. -->
- No breaking changes.
<!-- If any:
- [ ] ... (describe)
- Migration notes: ...
-->

## Highlights
<!-- Top 3–6 notable changes, user-visible first. Keep each to one line. -->
- ...

## Examples / Demos
<!-- Minimal examples, commands, or screenshots/gifs. Prefer concrete inputs/outputs. -->
- Example command: `...`
- Output: `...`
- Screenshots: (optional) ...

## Why
<!-- The rationale/problem this solves. 1–3 sentences. -->
- ...

## How
<!-- Brief approach/implementation notes. Keep it high-level; link to code as needed. -->
- ...

## Tests
<!-- What’s covered? How to run them? -->
- Added/updated tests: ...
- How to run: `uv run pytest -k ...`
- Current status: all passing / flaky test noted in ...

## Documentation
<!-- User docs, README, changelog. -->
- Docs updated: yes/no (link: ...)
- Changelog entry: added/not needed

## Performance
<!-- Any perf impact or measurements. -->
- No significant performance impact.
<!-- If applicable:
- Benchmark: before ... / after ...
-->

## Compatibility
<!-- Supported versions, APIs, platforms affected. -->
- No compatibility issues identified.

## Migration Steps (if any)
<!-- Only needed if there are breaking changes. -->
- Not applicable.

## Rollback Plan
<!-- How to safely revert if things go wrong. -->
- Revert via `git revert <sha>`; no data migration required.
<!-- If data/schema changes:
- Provide rollback script/steps: ...
-->

## Related
<!-- Link issues, PRs, design docs, tickets. -->
- Closes #...
- Relates to #...

## Checklist
Before submitting the PR, make sure to check every entry of the following checklist (`[x]`), or move them to N/A.

#### Done:
- [ ] Tests added/updated
- [ ] Docs updated (README/CHANGELOG)
- [ ] Backwards compatible (or migration steps provided)
- [ ] Self-reviewed and linted
- [ ] Examples/screenshots/recordings for usage changes

#### N/A:
- None
