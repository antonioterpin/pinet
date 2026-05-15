# Exploration, validation, and temporary scripts

## When unsure about design

When unsure about an idea, behavior, or design:
- **validate it with a temporary script or experiment**
- do *not* pollute the main codebase with exploratory logic

## After validation

After validation:
- **delete the temporary code**
- encode the conclusion in one of:
  - a concise comment (for local decisions)
  - a docstring (for API-level behavior)
  - a document under `docs/` (for important design choices)
