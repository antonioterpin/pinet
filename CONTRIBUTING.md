# Contributing

- [Development setup](#development-setup)
- [Development workflow](#development-workflow)
- [Testing a feature](#testing-a-feature)
- [Quality gates](#quality-gates)
- [Preparing for a PR](#preparing-for-a-pr)

All project rules, standards, and step-by-step workflows live in
[`docs/`](docs/) — start at [`docs/index.md`](docs/index.md). This file
covers human-facing setup and the contribution workflow only.

## Development setup

We use [`uv`](https://docs.astral.sh/uv/) to manage the virtual environment
and dependencies. [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/),
then sync the environment from the lockfile:

```sh
# Runtime + dev dependencies (pre-commit, pytest, basedpyright, ruff, ...)
uv sync --extra dev

# Add the CUDA 12 wheels as well if you have a GPU
uv sync --extra dev --extra cuda12
```

`uv run <command>` runs a command inside the managed environment; never
assume a global install. Add dependencies with `uv add <package>` (runtime)
or `uv add --dev <package>` (dev only) — do not hand-edit `pyproject.toml`,
and commit the updated `uv.lock`.

Install the git hooks (pre-commit, pre-push, and commit-msg are all
configured):

```sh
uv run pre-commit install --install-hooks
```

To configure the Conventional Commits message template:

```sh
git config commit.template .gitmessage
```

## Development workflow

We follow a [Git feature-branch](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow)
workflow with test-driven development:

1. Open an issue describing the goal and the tests that must pass for it to
   be considered done.
2. Branch from `dev`:
   ```sh
   git checkout dev
   git checkout -b <type>-<short-description>
   ```
3. Write the tests first (see [Testing a feature](#testing-a-feature) and
   [docs/standards/testing.md](docs/standards/testing.md)).
4. Implement until the tests pass and all [quality gates](#quality-gates)
   are green.
5. Open a PR to `dev` (squashed to a single Conventional-Commits commit).
6. Delete the branch once merged.

- `main` and `dev` are protected and require a PR.
- CI runs the [code-style](.github/workflows/code-style.yaml),
  [commit-lint](.github/workflows/commitlint.yaml), and
  [tests](.github/workflows/test.yaml) workflows on every push, so the state
  of each feature is visible without polling the author.
- A PR to `main` is opened only for milestones (a `dev → main` release).

Pick the workflow that matches your task from
[`docs/workflows/`](docs/workflows/) (feature, bugfix, refactor,
api-validation, docs); first-timers should start with
[`docs/workflows/orientation.md`](docs/workflows/orientation.md).

## Testing a feature

Add a `test_<area>.py` file under `src/test/` (flat layout) with a
module-level docstring describing the behavior under test. See
[docs/standards/testing.md](docs/standards/testing.md) for the full policy.

Run the suite:

```sh
uv run pytest

# A single file
uv run pytest src/test/test_box.py

# With coverage
uv run pytest --cov=src/pinet --cov-report=term-missing
```

## Quality gates

All of these must pass before a PR (the pre-commit/pre-push hooks run them
automatically; you can also run them by hand):

```sh
# Lint and format
uv run pre-commit run --all-files

# Type check (basedpyright; strict on both src/ and tests/)
uv run pre-commit run --hook-stage push --all-files

# Tests
uv run pytest
```

If a hook keeps you from committing or pushing and you understand why it is
safe to bypass it, add `--no-verify` — but CI will still enforce the gate.

## Preparing for a PR

Squash your branch into a single, well-formed Conventional-Commits commit
before opening the PR so reviewers see one concise change:

```sh
git rebase -i HEAD~<number-of-commits>
```

Keep the first commit as `pick`, mark the rest `squash` (or `s`), then edit
the combined message to follow the
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary)
standard (see `.gitmessage`). Resolve any conflicts and
`git rebase --continue`. If the branch was already pushed, force-push the
squashed history (`git push --force-with-lease`). Then open the PR from your
branch to `dev`.
