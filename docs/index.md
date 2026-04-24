# PiNet documentation hub

This is the single source of truth for all PiNet documentation related to agent-driven development. Start here.

---

## Guides

| Document | What it covers |
|---|---|
| [agent-development.md](guides/agent-development.md) | Using PiNet with Claude Code, GitHub Copilot, Antigravity, and custom agents |

The top-level [CONTRIBUTING.md](../CONTRIBUTING.md) covers human-facing development setup, branch workflow, PR preparation, and commit style.

---

## Standards

One file per topic -- all code must satisfy these:

| Standard | Topic |
|---|---|
| [environment-tooling.md](standards/environment-tooling.md) | Always use `uv run`; virtual-env discipline |
| [code-quality.md](standards/code-quality.md) | Pre-commit gates, quality checklist |
| [code-clarity.md](standards/code-clarity.md) | Naming, comments, markdown & prose conventions |
| [code-organization.md](standards/code-organization.md) | Module layout and ownership rules |
| [typing-docstrings.md](standards/typing-docstrings.md) | Modern type hints (PEP 585/604), Google-style docstrings |
| [testing.md](standards/testing.md) | TDD, Arrange/Act/Assert, coverage, hypothesis |
| [ml-numerical.md](standards/ml-numerical.md) | JAX, determinism, jit/vmap guidelines |
| [api-design.md](standards/api-design.md) | Public API contracts, simplicity |
| [change-scope.md](standards/change-scope.md) | What belongs in a single PR |
| [version-control.md](standards/version-control.md) | Commit messages, branch naming |
| [linting-formatting.md](standards/linting-formatting.md) | Ruff, pydoclint, basedpyright config |
| [exploration-validation.md](standards/exploration-validation.md) | Scratch scripts and API validation |
| [device-utilization.md](standards/device-utilization.md) | GPU selection, JAX device placement |

---

## Workflows

Step-by-step procedures for common tasks:

| Workflow | When to use |
|---|---|
| [orientation.md](workflows/orientation.md) | First time on the project, or returning after a long absence |
| [feature.md](workflows/feature.md) | Implementing a new feature (TDD approach) |
| [bugfix.md](workflows/bugfix.md) | Fixing a reported bug |
| [refactor.md](workflows/refactor.md) | Improving structure without changing behavior |
| [api-validation.md](workflows/api-validation.md) | Exploring external APIs or uncertain design |
| [docs.md](workflows/docs.md) | Documentation-only changes |

---

## Agent personas

| Persona | Purpose |
|---|---|
| [agents/implementer.md](agents/implementer.md) | Guidance for the implementing agent role |
| [agents/reviewer.md](agents/reviewer.md) | Guidance for the reviewing agent role |

---

## Quick reference

```bash
# Run tests
uv run pytest

# Lint and format
uv run pre-commit run --all-files

# Type check
uv run pre-commit run --hook-stage push --all-files

# Add a dependency
uv add <package>          # runtime
uv add --dev <package>    # dev-only
```
